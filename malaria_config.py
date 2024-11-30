import os
import sys
import json
import datetime
import numpy as np
import tensorflow as tf
import skimage.draw
import skimage.io
import skimage.color
import matplotlib.pyplot as plt

# Compatibility imports
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam

# Updated import for Mask R-CNN (assuming you've installed the right version)
try:
    import mrcnn
    from mrcnn.config import Config
    from mrcnn import model as modellib
    from mrcnn import utils
except ImportError:
    print("Mask R-CNN library not found. Please install from GitHub.")
    sys.exit(1)

# Configuration for Google Colab
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print("Error configuring GPU memory growth.")
# Validate TensorFlow version
assert tf.__version__.startswith('2'), "TensorFlow 2.x is required"

class MalariaConfig(Config):
    """Configuration for training on the malaria dataset."""
    NAME = "malaria"
    
    # GPU config for Colab
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2  # Adjust based on your GPU memory
    
    # Model hyperparameters
    NUM_CLASSES = 1 + 4  # Background + cell types
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    
    # Improved detection parameters
    DETECTION_MIN_CONFIDENCE = 0.85
    DETECTION_NMS_THRESHOLD = 0.3
    
    # Image processing
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024
    IMAGE_RESIZE_MODE = "pad64"
    
    # Training hyperparameters
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    # Augmentation
    ROTATION_RANGE = 45
    ZOOM_RANGE = [0.8, 1.2]

class CellDataset(utils.Dataset):
    """Dataset for malaria cell detection with enhanced loading."""
    def load_cell(self, dataset_dir, subset):
        """Load dataset with robust error handling."""
        self.add_class("cell", 1, "uninfectedretic")
        self.add_class("cell", 2, "ring")
        self.add_class("cell", 3, "trophozoite")
        self.add_class("cell", 4, "schizont")

        assert subset in ["train", "val"], "Subset must be 'train' or 'val'"
        
        # Enhanced path handling
        dataset_path = os.path.join(dataset_dir, subset)
        
        # Robust JSON loading with error handling
        try:
            with open(os.path.join(dataset_path, "via_region_data.json"), 'r') as f:
                annotations = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading annotations: {e}")
            return

        # Filter annotations with regions
        annotations = [a for a in annotations.values() if a.get('regions')]

        for a in annotations:
            # Safety checks
            if not a.get('filename') or not os.path.exists(os.path.join(dataset_path, a['filename'])):
                continue

            # Extract polygon and object information
            polygons = [r['shape_attributes'] for r in a.get('regions', {}).values()]
            objects = [s.get('region_attributes', {}) for s in a.get('regions', {}).values()]
            
            # Safely extract cell types
            num_ids = [int(n.get('cell', 0)) for n in objects if n.get('cell')]

            # Image loading with error handling
            try:
                image_path = os.path.join(dataset_path, a['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
            except Exception as e:
                print(f"Error processing image {a['filename']}: {e}")
                continue

            self.add_image(
                "cell",
                image_id=a['filename'],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks with improved handling."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "cell":
            return super(self.__class__, self).load_mask(image_id)
        
        num_ids = image_info['num_ids']
        info = self.image_info[image_id]
        
        # Create mask with safe dimensions
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            # Ensure polygon points are within image boundaries
            y = np.clip(p['all_points_y'], 0, info["height"] - 1)
            x = np.clip(p['all_points_x'], 0, info["width"] - 1)
            
            rr, cc = skimage.draw.polygon(y, x)
            mask[rr, cc, i] = 1

        return mask, np.array(num_ids, dtype=np.int32)

def train(model, dataset_dir, config):
    """Enhanced training function with more logging and checkpointing."""
    # Prepare datasets
    dataset_train = CellDataset()
    dataset_train.load_cell(dataset_dir, "train")
    dataset_train.prepare()

    dataset_val = CellDataset()
    dataset_val.load_cell(dataset_dir, "val")
    dataset_val.prepare()

    # Callbacks for better training
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='malaria_best_model.h5', 
            save_best_only=True, 
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10, 
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1, 
            patience=5
        )
    ]

    # Train
    print("Training network heads")
    model.train(
        dataset_train, 
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=50,  # Increased epochs
        layers='heads',
        custom_callbacks=callbacks
    )

def color_splash(image, mask):
    """Apply color splash effect with more robust handling."""
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Convert to grayscale
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    
    # Handle mask
    if mask is not None and mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    
    return splash

def main():
    """Main execution function with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Malaria Cell Detection with Mask R-CNN')
    parser.add_argument('--mode', required=True, choices=['train', 'detect'],
                        help='Mode of operation: train or detect')
    parser.add_argument('--dataset', required=True,
                        help='Path to malaria dataset')
    parser.add_argument('--weights', default='coco',
                        help='Path to weights: coco, last, or specific .h5 file')
    parser.add_argument('--image', help='Path to image for detection')
    
    args = parser.parse_args()

    # Configuration
    config = MalariaConfig()
    config.display()

    # Model initialization
    model = modellib.MaskRCNN(
        mode="training" if args.mode == 'train' else "inference", 
        config=config,
        model_dir='./logs'
    )

    # Weight loading strategy
    if args.weights.lower() == 'coco':
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif args.weights.lower() == 'last':
        model.load_weights(model.find_last(), by_name=True)
    else:
        model.load_weights(args.weights, by_name=True)

    # Execute based on mode
    if args.mode == 'train':
        train(model, args.dataset, config)
    elif args.mode == 'detect':
        image = skimage.io.imread(args.image)
        results = model.detect([image], verbose=1)
        r = results[0]
        splash = color_splash(image, r['masks'])
        plt.imsave('splash_output.png', splash)

if __name__ == '__main__':
    main()
