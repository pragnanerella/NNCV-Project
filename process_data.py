# process_data.py
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def preprocess(img: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for the model.
    
    Args:
        img (PIL.Image): Input image in original shape.
    
    Returns:
        torch.Tensor: Processed image tensor of shape [B, C, H, W].
    """
    transform = Compose([
        ToTensor(),  # Converts PIL image to tensor and scales it to [0,1]
        Resize((256, 512)),  # Resize image
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1,1]
    ])
    
    img_tensor = transform(img).to(torch.float32)  # Ensure dtype is float32
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def postprocess(prediction: torch.Tensor, shape: tuple) -> np.ndarray:
    """
    Postprocess the model's prediction to the original image shape with train IDs.
    
    Args:
        prediction (torch.Tensor): Model output tensor of shape [B, n_classes, H, W].
        shape (tuple): Original image shape (X, Y).
    
    Returns:
        np.ndarray: Segmentation map of shape [X, Y, 1] with train IDs.
    """
    # Remove batch dimension and apply argmax to get class indices (train IDs)
    prediction = prediction.squeeze(0)  # [n_classes, H, W]
    pred_ids = torch.argmax(prediction, dim=0)  # [H, W]
    print("pred_ids", pred_ids)  # Debug
    
    # Resize to original shape
    pred_ids = pred_ids.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    pred_ids = torch.nn.functional.interpolate(
        pred_ids.float(), size=shape, mode='nearest'
    ).squeeze(0).squeeze(0).long()  # [X, Y]
    
    # Model outputs train IDs directly, so no additional mapping needed
    train_ids = pred_ids.cpu().numpy()
    print("train_ids", train_ids)  # Debug
    
    # Ensure values are valid train IDs (0-18, 255 for out-of-range)
    train_ids = np.where((train_ids >= 0) & (train_ids <= 18), train_ids, 255)
    
    # Shape to [X, Y, 1]
    train_ids = train_ids[..., np.newaxis]
    return train_ids.astype(np.uint8)