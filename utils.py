from typing import List, Dict, Optional
import os
import pickle
import numpy as np

def image_to_nparray(img_path: str) -> np.ndarray:
    """
    Load an image from the specified path and convert it to a numpy array.
    Args:
        img_path (str): Path to the image file.
    Returns:
        np.ndarray: Image as a numpy array.
    from keras.preprocessing.image import load_img, img_to_array
    """
    from keras.preprocessing.image import load_img, img_to_array
    img = load_img(img_path, color_mode='grayscale')
    img_array = img_to_array(img)
    return img_array


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
        img (np.ndarray): Input image array.
        size (tuple): Desired size (width, height).
    Returns:
        np.ndarray: Resized image array.
    """
    try:
        from keras.preprocessing.image import img_to_array, array_to_img
        pil_img = array_to_img(img)
        pil_img = pil_img.resize(size)
        return img_to_array(pil_img)
    except ImportError:
        raise ImportError("Keras is required for image resizing.")
    except Exception as e:
        raise RuntimeError(f"Error resizing image: {str(e)}")


def save_model_metadata(proj_name: str, classes: List[str], clf) -> bool:
    """
    Save model metadata to disk using pickle.
    Args:
        proj_name (str): Name of the project.
        classes (List[str]): List of class names.
        clf: Classifier object.
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        metadata = {'proj_name': proj_name, 'classes': classes, 'classifier': clf}
        filepath = os.path.join('models', f'{proj_name}_metadata.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        return True
    except Exception as e:
        print(f"Error saving model metadata: {str(e)}")
        return False