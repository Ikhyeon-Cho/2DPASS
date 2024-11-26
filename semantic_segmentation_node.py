#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import yaml
import importlib
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from easydict import EasyDict
import os
import struct

class SemanticSegmentationNode:
    def __init__(self, config_path, checkpoint_path):
        rospy.init_node('semantic_segmentation_node')
        
        # Initialize color map first
        self.color_map = self._load_color_map()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        try:
            self._initialize_model(checkpoint_path)
        except Exception as e:
            rospy.logerr(f"Failed to initialize model: {str(e)}")
            raise
        
        # ROS subscribers and publishers
        self.pc_sub = rospy.Subscriber('/voxel_cloud', PointCloud2, self.pointcloud_callback, queue_size=1)
        self.pred_pub = rospy.Publisher('/segmented_pointcloud', PointCloud2, queue_size=1)
        
        rospy.loginfo("Semantic segmentation node initialized!")

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add required default configurations
        default_config = {
            'submit_to_server': False,
            'baseline_only': False,
            'test': True,
            'debug': False,
            'gpu': [0],
            'num_vote': 1,
            'save_prediction': False,
            'batch_size': 1,
            'dataset_params': {
                'val_data_loader': {
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1,
                    'rotate_aug': False,
                    'flip_aug': False,
                    'scale_aug': False,
                    'transform_aug': False,
                    'dropout_aug': False
                },
                'ignore_label': 0,
                'return_test': True,
                'pc_dataset_type': 'SemanticKITTI'
            },
            'model_params': {
                'model_architecture': '2dpass',
                'input_dims': 4,
                'num_classes': 20,
                'hiden_size': 64,
                'spatial_shape': [1000, 1000, 60],
                'scale_list': [2, 4, 8, 16]
            },
            'train_params': {
                'max_num_epochs': 64,
                'learning_rate': 0.24,
                'optimizer': 'SGD',
                'lr_scheduler': 'CosineAnnealingWarmRestarts',
                'momentum': 0.9,
                'nesterov': True,
                'weight_decay': 1.0e-4
            }
        }
        
        # Update config with default values (recursive)
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    if k not in d:
                        d[k] = v
            return d
        
        config = update_dict(config, default_config)
        return EasyDict(config)

    def _initialize_model(self, checkpoint_path):
        """Initialize model with proper error handling"""
        model_file = importlib.import_module('network.' + self.config['model_params']['model_architecture'])
        self.model = model_file.get_model(self.config)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Disable criterion during inference
        self.model.criterion = lambda x: x

    def preprocess_pointcloud(self, points_msg):
        # Convert ROS PointCloud2 to numpy array
        points_list = []
        for p in point_cloud2.read_points(points_msg, skip_nans=True):
            points_list.append([p[0], p[1], p[2], p[3]])  # x, y, z, intensity
        points = np.array(points_list, dtype=np.float32)
        
        # Convert numpy array to torch tensor and move to device
        points_tensor = torch.from_numpy(points).to(self.device)  # Shape: [N, 4]
        
        # Create data dictionary for model with batch structure
        data_dict = {
            'points': points_tensor,  # Shape: [N, 4]
            'lidar': points_tensor,   # Shape: [N, 4]
            'batch_size': torch.tensor([1]).to(self.device),  # Single tensor for batch size
            'batch_idx': torch.zeros(points.shape[0], dtype=torch.long).to(self.device),  # Shape: [N]
            'labels': torch.zeros(points.shape[0], dtype=torch.long).to(self.device),  # Keep dummy labels
            'is_test': True  # Flag to indicate test/inference mode
        }
        
        return data_dict

    def _load_color_map(self):
        """Load SemanticKITTI color map in BGR format"""
        color_map = {
            0: [0, 0, 0],         # unlabeled - black
            1: [100, 150, 245],   # car - reddish
            2: [100, 230, 245],   # bicycle
            3: [30, 60, 150],     # motorcycle
            4: [80, 30, 180],     # truck
            5: [0, 0, 255],       # other-vehicle - red
            6: [255, 30, 30],     # person - blue
            7: [255, 40, 200],    # bicyclist
            8: [150, 30, 90],     # motorcyclist
            9: [255, 0, 255],     # road - magenta
            10: [255, 150, 255],  # parking
            11: [75, 0, 75],      # sidewalk
            12: [175, 0, 75],     # other-ground
            13: [255, 200, 0],    # building - light blue
            14: [255, 120, 50],   # fence
            15: [0, 175, 0],      # vegetation - green
            16: [135, 60, 0],     # trunk - dark brown
            17: [150, 240, 80],   # terrain
            18: [255, 240, 150],  # pole
            19: [255, 0, 0]       # traffic-sign - blue
        }
        return color_map

    def postprocess_predictions(self, predictions, original_msg):
        pred_labels = predictions['logits'].argmax(dim=1).cpu().numpy()
        
        # Create new point cloud with labels and colors
        points_list = []
        for i, p in enumerate(point_cloud2.read_points(original_msg, skip_nans=True)):
            label = pred_labels[i]
            # Get BGR color and convert to RGB for visualization
            bgr = self.color_map[label]
            rgb = (bgr[2], bgr[1], bgr[0])  # Convert BGR to RGB
            
            # Pack RGB into float32 using PCL's method
            rgb_packed = struct.unpack('f', struct.pack('I', 
                (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])))[0]
            
            points_list.append([p[0], p[1], p[2], rgb_packed])
        
        # Create PointCloud2 message with color field
        fields = [
            point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('rgb', 12, point_cloud2.PointField.FLOAT32, 1)
        ]
        
        pc2_msg = point_cloud2.create_cloud(original_msg.header, fields, points_list)
        return pc2_msg

    def pointcloud_callback(self, msg):
        try:
            with torch.no_grad():
                data_dict = self.preprocess_pointcloud(msg)
                predictions = self.model(data_dict)
                rospy.logdebug(f"Model output keys: {predictions.keys()}")
                predictions_msg = self.postprocess_predictions(predictions, msg)
                self.pred_pub.publish(predictions_msg)
                
        except Exception as e:
            rospy.logerr(f"Error in pointcloud processing: {str(e)}")
            rospy.logerr(f"Available prediction keys: {predictions.keys() if 'predictions' in locals() else 'N/A'}")

if __name__ == '__main__':
    try:
        config_path = rospy.get_param('~config_path', 'config/2DPASS-semantickitti.yaml')
        checkpoint_path = rospy.get_param('~checkpoint_path', 'checkpoints/best_model_64.ckpt')
        
        node = SemanticSegmentationNode(config_path, checkpoint_path)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 