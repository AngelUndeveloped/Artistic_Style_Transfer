# This project is an attempt to Neural Transfer Using PyTorch
# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#neural-transfer-using-pytorch

"""torch, torch.nn, numpy (indispensables packages for neural networks with PyTorch)
torch.optim (efficient gradient descents)
PIL, PIL.Image, matplotlib.pyplot (load and display images)
torchvision.transforms (transform PIL images into tensors)
torchvision.models (train or load pretrained models)
copy (to deep copy the models; system package)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

import copy

# Check what gpus are available, if any available get gpu, otherwise get cpu
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(gpu_device)

# Desired size of the output image
output_image_size = 512 if torch.cuda.is_available() else 128  # Smaller sizes is for cpu

# Define a transform pipeline for loading images
loader = transforms.Compose([
    # Resize the image to the specified size
    transforms.Resize(output_image_size),
    # Crop the center of the image to match the desired size
    transforms.CenterCrop(output_image_size),
    # Convert the image to a tensor
    transforms.ToTensor()
])


def image_loader(image_name):
    # Open the image
    image = Image.open(image_name)
    # Apply the transform pipeline to the image
    image = loader(image)
    # Add a fake batch dimension to fit the network's input dimensions
    image = image.unsqueeze(0)
    # Move the image to the GPU and convert it to a float tensor
    image = image.to(gpu_device, torch.float)

    return image


style_img = image_loader("./images/picasso.jpg")
content_img = image_loader("./images/dancing.jpg")

assert style_img.size == content_img.size()
print("we need to import style and content images of the same size")

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')