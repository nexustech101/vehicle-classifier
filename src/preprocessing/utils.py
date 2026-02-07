"""Legacy preprocessing utilities for image manipulation."""

from typing import List
import os
import numpy as np
from PIL import Image


def image_to_nparray(img_path: str) -> np.ndarray:
    """
    Load an image from the specified path and convert it to a numpy array.
    
    Args:
        img_path: Path to the image file.
    
    Returns:
        Image as a numpy array with shape (H, W, 1).
    """
    img = Image.open(img_path).convert('L')
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=-1)


def flatten_image(img: np.ndarray) -> np.ndarray:
    """
    Flatten a 2D or 3D image array into a 1D array.
    Args:
        img (np.ndarray): Input image array.
    Returns:
        np.ndarray: Flattened image array.
    """
    return img.flatten()


def crop_center(img: np.ndarray, cropx: int, cropy: int) -> np.ndarray:
    """
    Crop the center of an image to the specified size.
    Args:
        img (np.ndarray): Input image array.
        cropx (int): Width of the cropped image.
        cropy (int): Height of the cropped image.
    Returns:
        np.ndarray: Cropped image array.
    """
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty+cropy, startx:startx+cropx]


def resize_image(img: np.ndarray, size: tuple) -> np.ndarray:
    """
    Resize an image to the specified size.
    
    Args:
        img: Input image array.
        size: Desired size (width, height).
    
    Returns:
        Resized image array.
    """
    # Handle channel dimension
    squeeze = False
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(axis=-1)
        squeeze = True
    
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = pil_img.resize(size, Image.LANCZOS)
    result = np.array(pil_img, dtype=np.float32)
    
    if squeeze:
        result = np.expand_dims(result, axis=-1)
    
    return result


def save_model_metadata(proj_name: str, classes: List[str], clf) -> bool:
    """
    Save model metadata to disk using JSON.
    
    Args:
        proj_name: Name of the project.
        classes: List of class names.
        clf: Classifier object (config will be extracted if available).
    
    Returns:
        True if saved successfully, False otherwise.
    """
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        metadata = {
            'proj_name': proj_name,
            'classes': classes,
            'classifier_type': type(clf).__name__,
        }
        
        # Extract config if available
        if hasattr(clf, 'get_config'):
            metadata['config'] = clf.get_config()
        
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', f'{proj_name}_metadata.json')
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")
        return False