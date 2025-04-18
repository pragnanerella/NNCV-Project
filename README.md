# Cityscapes Segmentation Model

This repository contains an enhanced U-Net architecture for Cityscapes segmentation with Atrous Spatial Pyramid Pooling (ASPP) and improved upsampling. The model is built using PyTorch and includes preprocessing and postprocessing utilities.

## Required Libraries and Installation

This project depends on the following libraries:

- Python 3.7 or higher
- PyTorch
- torchvision
- NumPy
- Pillow
- tqdm

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install torch torchvision numpy pillow tqdm
```

## Model Overview

The model architecture is based on U-Net with the following key components:

- **DoubleConv**: Two consecutive convolutional layers followed by batch normalization, ReLU activation, and dropout.
- **Down**: Max pooling followed by a DoubleConv block.
- **Up**: Upsampling with transposed convolutions and concatenation with corresponding skip connections.
- **ASPP (Atrous Spatial Pyramid Pooling)**: Multi-scale context aggregation for better segmentation performance, including a global context branch.
- **OutConv**: Final output convolution for producing segmentation maps.

### Model Parameters

The `Model` class accepts the following arguments:

- `in_channels` (_int_): Number of channels in the input image (default: 3).
- `n_classes` (_int_): Number of segmentation classes (default: 19 for the Cityscapes dataset).

## Data Processing

The following preprocessing and postprocessing functions are available for handling input data:

### `preprocess(img: Image.Image) -> torch.Tensor`

Preprocesses a given image for model input:

1. Converts the image to a tensor.
2. Resizes the image to `256×512`.
3. Normalizes the image to the range `[-1, 1]`.

### `postprocess(prediction: torch.Tensor, shape: tuple) -> np.ndarray`

Postprocesses the model's prediction to map it back to the original image shape and convert the output to class IDs.

## Example Usage

```python
from model import Model
from process_data import preprocess, postprocess
from PIL import Image
import torch

# Load model
model = Model()

# Load and preprocess an image
image_path = 'path_to_image.jpg'
image = Image.open(image_path)
input_tensor = preprocess(image).unsqueeze(0)  # add batch dimension

# Run inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# Postprocess the prediction
output_shape = image.size[::-1]  # (height, width)
segmentation_map = postprocess(output, output_shape)

# Save or visualize the segmentation map
```

## Model Weights

The model weights (`model.pth`) are too large to store in this repository. Please download the pretrained model from the following link:

- [Download `model.pth` from Google Drive](https://drive.google.com/your-link-here)

## Contact Information for Codalab Mapping

To ensure correct mapping across Codalab and other systems, please use the following credentials:

- **Codalab Username**: pragnanerella
- **TU/e Email Address**: v.p.nerella@student.tue.nl

