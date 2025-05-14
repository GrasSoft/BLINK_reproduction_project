import torch
import torch.optim as optim
from helper_functions import *
# WARNING: Do not import any other libraries or files

def normalize(img, mean, std):
    """ Z-normalizes an image tensor.

    # Parameters:
        @img, torch.tensor of size (b, c, h, w)
        @mean, torch.tensor of size (c)
        @std, torch.tensor of size (c)

    # Returns the normalized image
    """
    
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    
    # Apply normalization
    img = (img - mean) / std
    
    return img 

def content_loss(input_features, content_features, content_layers):
    """ Calculates the content loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @content_features, VGG features of the content image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @content_layers, a list containing which layers to consider for calculating
            the content loss.
    
    # Returns the content loss, a torch.tensor of size (1)
    """
    total_loss = 0.0

    for layer in content_layers:
        input_layer_features = input_features[layer]
        content_layer_features = content_features[layer]
        
        layer_loss = torch.nn.functional.mse_loss(input_layer_features, content_layer_features)
        
        total_loss += layer_loss
    
    normalized_loss = total_loss / len(content_layers)

    
    return normalized_loss

def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.

    # NOTE: Normalize by number of number of dimensions of the feature matrix.
    
    # Parameters:
        @x, torch.tensor of size (b, c, h, w) 

    # Returns the gram matrix
    """
    b, c, h, w = x.size()

    # reshape to (b, c, h * w)
    features = x.view(b, c, -1)
    
    gram = torch.bmm(features, features.transpose(1, 2))
    
    return gram / (c * h * w)

def style_loss(input_features, style_features, style_layers):
    """ Calculates the style loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @style_features, VGG features of the style image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @style_layers, a list containing which layers to consider for calculating
            the style loss.
    
    # Returns the style loss, a torch.tensor of size (1)
    """
    
    total_loss = 0.0
    for layer in style_layers:
        input_layer_features = gram_matrix(input_features[layer])
        style_layer_features = gram_matrix(style_features[layer])
        
        layer_loss = torch.nn.functional.mse_loss(input_layer_features, style_layer_features)
        
        total_loss += layer_loss
        
    normalized_loss = total_loss / len(style_layers)

    return normalized_loss

def total_variation_loss(y):
    """ Calculates the total variation across the spatial dimensions.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)
    
    # Returns the total variation, a torch.tensor of size (1)
    """
    # Calculate the difference along the height dimension (I_k,j,c - I_k+1,j,c)
    tv_h = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    
    # Calculate the difference along the width dimension (I_k,j,c - I_k,j+1,c)
    tv_w = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
    
    total_variation = torch.sum(tv_h) + torch.sum(tv_w)
    
    b, c, h, w = y.size()
    normalized_tv_loss = total_variation / (c * h * w)
    
    return normalized_tv_loss

def get_gradient_imgs(img):
    """ Calculates the gradient images based on the sobel kernel.

    # NOTE: 
      1. The gradient image along the x-dimension should be at first position,
         i.e. at out[:,0,:,:], and the gradient image calulated along the y-dimension
         should be at out[:,1,:,:].
      2. Do not use padding for the convolution.
      3. When defining the Sobel kernel, use the finite element approximation of the gradient and approximate the derivative in x-direction according to:
            df / dx  =  f(x+1,y) - f(x-1,y)   (value of left neighbor pixel is subtracted from the value of the right neighbor pixel)
         and the derivative in y-direction according to:
            df / dy  =  f(x,y+1) - f(x,y-1)   (value of bottom neighbor pixel is subtracted from the value of the top neighbor pixel)

    # Parameters:
        @img grayscale image, tensor of size (1,1,H,W)
    
    # Returns the gradient images, concatenated along the second dimension. 
      Size (1,2,H-2,W-2)
    """
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    gradient_x = torch.nn.functional.conv2d(img, sobel_x, stride=1, padding=0)
    gradient_y = torch.nn.functional.conv2d(img, sobel_y, stride=1, padding=0)
    
    gradients = torch.cat((gradient_x, gradient_y), dim=1)
    
    return gradients

def edge_loss(img1, img2):
    """ Calculates the edge loss based on the mean squared error between the two images.

    # Parameters:
        @img1 (1,2,H,W)
        @img2 (1,2,H,W)
    
    # Returns the edge loss, a torch.tensor of size (1)
    """
    loss = torch.nn.functional.mse_loss(img1, img2)
    return loss