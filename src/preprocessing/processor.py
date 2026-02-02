import os
import numpy as np
from pathlib import Path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image loading, resizing, and preprocessing for vehicle classification.
    Converts images to greyscale and resizes to 100x90.
    """
    
    TARGET_SIZE = (100, 90)  # (width, height)
    
    def __init__(self, dataset_root: str):
        """
        Initialize preprocessor with dataset root directory.
        
        Args:
            dataset_root: Path to root directory containing subdirectories with images
                         e.g., 'C:\\Users\\charl\\Documents\\datasets\\image-data\\vehicle-images'
        """
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        
        self.class_dirs = self._discover_class_directories()
        self.class_mapping = {class_name: idx for idx, class_name in enumerate(self.class_dirs)}
    
    def _discover_class_directories(self) -> list:
        """
        Discover all subdirectories in dataset root (each represents a vehicle type/class).
        
        Returns:
            List of class directory names sorted alphabetically
        """
        class_dirs = [
            d.name for d in self.dataset_root.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        class_dirs.sort()
        logger.info(f"Discovered {len(class_dirs)} classes: {class_dirs}")
        return class_dirs
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image, convert to greyscale, and resize to target size.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Numpy array of shape (100, 90, 1) with pixel values in [0, 255]
        
        Raises:
            Exception: If image cannot be loaded or processed
        """
        try:
            # Open image
            img = Image.open(image_path)
            
            # Convert to greyscale
            img_grey = img.convert('L')
            
            # Resize to target size (width, height)
            img_resized = img_grey.resize(self.TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and add channel dimension
            img_array = np.array(img_resized, dtype=np.uint8)
            img_array = np.expand_dims(img_array, axis=-1)  # (100, 90) -> (100, 90, 1)
            
            return img_array
        
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            raise
    
    def load_class_images(self, class_name: str) -> tuple:
        """
        Load all images from a specific class directory.
        
        Args:
            class_name: Name of the class (subdirectory name)
        
        Returns:
            Tuple of (images_array, class_label_array)
            - images_array: Shape (N, 100, 90, 1), dtype uint8
            - class_label_array: Shape (N,), dtype int, 0-indexed class labels
        
        Raises:
            ValueError: If class_name not found in discovered classes
        """
        if class_name not in self.class_mapping:
            raise ValueError(f"Unknown class: {class_name}. Available classes: {self.class_dirs}")
        
        class_path = self.dataset_root / class_name
        class_label = self.class_mapping[class_name]
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [
            f for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"No images found in {class_path}")
            return np.array([], dtype=np.uint8).reshape(0, 100, 90, 1), np.array([], dtype=np.int32)
        
        images = []
        labels = []
        
        logger.info(f"Loading {len(image_files)} images from class '{class_name}'...")
        for idx, img_file in enumerate(image_files):
            try:
                img_array = self.load_image(str(img_file))
                images.append(img_array)
                labels.append(class_label)
            except Exception as e:
                logger.warning(f"Skipping {img_file}: {e}")
                continue
        
        if not images:
            logger.warning(f"Failed to load any images from class '{class_name}'")
            return np.array([], dtype=np.uint8).reshape(0, 100, 90, 1), np.array([], dtype=np.int32)
        
        images_array = np.stack(images, axis=0)
        labels_array = np.array(labels, dtype=np.int32)
        
        logger.info(f"Loaded {len(images)} images from '{class_name}'")
        return images_array, labels_array
    
    def load_all_images(self) -> tuple:
        """
        Load all images from all classes in the dataset.
        
        Returns:
            Tuple of (all_images, all_labels, class_mapping_dict)
            - all_images: Shape (N, 100, 90, 1), dtype uint8
            - all_labels: Shape (N,), dtype int, 0-indexed class labels
            - class_mapping_dict: Dict mapping class index to class name
        """
        all_images = []
        all_labels = []
        
        logger.info(f"Loading images from {len(self.class_dirs)} classes...")
        
        for class_name in self.class_dirs:
            try:
                images, labels = self.load_class_images(class_name)
                if len(images) > 0:
                    all_images.append(images)
                    all_labels.append(labels)
            except Exception as e:
                logger.error(f"Error loading class '{class_name}': {e}")
                continue
        
        if not all_images:
            raise RuntimeError("No images loaded from any class")
        
        # Concatenate all images and labels
        all_images_array = np.concatenate(all_images, axis=0)
        all_labels_array = np.concatenate(all_labels, axis=0)
        
        # Create reverse mapping (index -> class_name)
        reverse_mapping = {idx: name for name, idx in self.class_mapping.items()}
        
        logger.info(f"Total images loaded: {len(all_images_array)}")
        logger.info(f"Image shape: {all_images_array.shape}")
        logger.info(f"Labels shape: {all_labels_array.shape}")
        
        return all_images_array, all_labels_array, reverse_mapping
    
    def get_class_info(self) -> dict:
        """
        Get information about classes in the dataset.
        
        Returns:
            Dict with class names and indices
        """
        return self.class_mapping


class BatchProcessor:
    """
    Handles batching and splitting of preprocessed image data.
    """
    
    @staticmethod
    def split_train_test(images: np.ndarray, labels: np.ndarray, 
                        test_ratio: float = 0.2, random_state: int = 42) -> tuple:
        """
        Split data into train and test sets while maintaining class distribution.
        
        Args:
            images: Image array of shape (N, 100, 90, 1)
            labels: Label array of shape (N,)
            test_ratio: Proportion of data to use for testing (0.0 to 1.0)
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (x_train, x_test, y_train, y_test)
        """
        np.random.seed(random_state)
        
        # Get unique classes and their indices
        unique_labels = np.unique(labels)
        train_indices = []
        test_indices = []
        
        # Stratified split: maintain class distribution
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            np.random.shuffle(label_indices)
            
            split_point = int(len(label_indices) * (1 - test_ratio))
            train_indices.extend(label_indices[:split_point])
            test_indices.extend(label_indices[split_point:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        x_train = images[train_indices]
        y_train = labels[train_indices]
        x_test = images[test_indices]
        y_test = labels[test_indices]
        
        logger.info(f"Train set: {len(x_train)} samples")
        logger.info(f"Test set: {len(x_test)} samples")
        
        return x_train, x_test, y_train, y_test
    
    @staticmethod
    def create_batches(images: np.ndarray, labels: np.ndarray, 
                       batch_size: int = 32) -> tuple:
        """
        Create mini-batches from data.
        
        Args:
            images: Image array of shape (N, 100, 90, 1)
            labels: Label array of shape (N,)
            batch_size: Size of each batch
        
        Returns:
            Tuple of (image_batches, label_batches) as lists
        """
        num_samples = len(images)
        num_batches = int(np.ceil(num_samples / batch_size))
        
        image_batches = []
        label_batches = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            image_batches.append(images[start_idx:end_idx])
            label_batches.append(labels[start_idx:end_idx])
        
        logger.info(f"Created {num_batches} batches of size {batch_size}")
        return image_batches, label_batches


class DataAugmentation:
    """
    Provides data augmentation techniques for training (optional).
    """
    
    @staticmethod
    def random_rotation(image: np.ndarray, max_angle: int = 15) -> np.ndarray:
        """
        Apply random rotation to image.
        
        Args:
            image: Image array of shape (100, 90, 1)
            max_angle: Maximum rotation angle in degrees
        
        Returns:
            Rotated image array
        """
        from PIL import Image as PILImage
        
        angle = np.random.uniform(-max_angle, max_angle)
        img_pil = PILImage.fromarray(image.squeeze(), mode='L')
        img_rotated = img_pil.rotate(angle, expand=False, fillcolor=0)
        return np.expand_dims(np.array(img_rotated), axis=-1)
    
    @staticmethod
    def random_brightness(image: np.ndarray, factor: float = 0.2) -> np.ndarray:
        """
        Apply random brightness adjustment.
        
        Args:
            image: Image array of shape (100, 90, 1)
            factor: Brightness adjustment factor (0.0 to 1.0)
        
        Returns:
            Brightness-adjusted image array
        """
        adjustment = np.random.uniform(1 - factor, 1 + factor)
        adjusted = np.clip(image.astype(float) * adjustment, 0, 255)
        return adjusted.astype(np.uint8)
    
    @staticmethod
    def random_horizontal_flip(image: np.ndarray) -> np.ndarray:
        """
        Apply random horizontal flip.
        
        Args:
            image: Image array of shape (100, 90, 1)
        
        Returns:
            Flipped image array (50% chance)
        """
        if np.random.rand() > 0.5:
            return np.fliplr(image)
        return image
    
    @staticmethod
    def augment_batch(images: np.ndarray, augmentation_fn=None) -> np.ndarray:
        """
        Apply augmentation to a batch of images.
        
        Args:
            images: Image batch of shape (N, 100, 90, 1)
            augmentation_fn: Function to apply to each image (or None for random augmentation)
        
        Returns:
            Augmented image batch
        """
        augmented = []
        
        for img in images:
            if augmentation_fn:
                aug_img = augmentation_fn(img)
            else:
                # Random combination of augmentations
                aug_img = DataAugmentation.random_horizontal_flip(img)
                if np.random.rand() > 0.5:
                    aug_img = DataAugmentation.random_brightness(aug_img)
                if np.random.rand() > 0.7:
                    aug_img = DataAugmentation.random_rotation(aug_img)
            
            augmented.append(aug_img)
        
        return np.array(augmented)


# Example usage and convenience function
def prepare_dataset(*, dataset_root: str, test_ratio: float = 0.2) -> tuple:
    """
    Convenience function to load and prepare entire dataset.
    
    Args:
        dataset_root: Path to root directory with image subdirectories
        test_ratio: Proportion of data for testing
    
    Returns:
        Tuple of (x_train, x_test, y_train, y_test, class_mapping)
    """
    # Load all images
    preprocessor = ImagePreprocessor(dataset_root)
    images, labels, class_mapping = preprocessor.load_all_images()
    
    # Split into train/test
    x_train, x_test, y_train, y_test = BatchProcessor.split_train_test(
        images, labels, test_ratio=test_ratio
    )
    
    logger.info(f"Dataset preparation complete. Classes: {class_mapping}")
    
    return x_train, x_test, y_train, y_test, class_mapping
