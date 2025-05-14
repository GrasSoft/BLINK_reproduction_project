import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

from helper_functions import *
from assses_style_differences import *

def run_style_transfer(vgg_mean, vgg_std, content_img, style_img):
    """ Neural Style Transfer optmization procedure for a single style image.
    
    # Parameters:
        @vgg_mean, VGG channel-wise mean, torch.tensor of size (c)
        @vgg_std, VGG channel-wise standard deviation, detorch.tensor of size (c)
        @content_img, torch.tensor of size (1, c, h, w)
        @style_img, torch.tensor of size (1, c, h, w)
        @num_steps, int, iteration steps
        @random_init, bool, whether to start optimizing with based on a random image. If false,
            the content image is as initialization.
        @w_style, float, weight for style loss
        @w_content, float, weight for content loss 
        @w_tv, float, weight for total variation loss
        @w_edge, float, weight for edge loss
        @print_iter, int, iteration interval for printing the losses

    # Returns the style-transferred image
    """

    # Initialize Model
    model = Vgg19(content_layers, style_layers, device)

    normed_style_img = normalize(style_img, vgg_mean, vgg_std)

    loss = torch.tensor([0.], device=device)

    # Retrieve feature maps for content and style image
    # We do not need to calculate gradients for these feature maps
    with torch.no_grad():
        style_features = model(normed_style_img)
    
        # Retrieve features of image that is being optimized
        normed_img = normalize(content_img, vgg_mean, vgg_std)
        input_features = model(normed_img)

        s_loss = style_loss(input_features, style_features, style_layers)
    
        # Sum up the losses
        loss = loss + s_loss

    return loss

if __name__ == '__main__':
    seed_everything(101)

    device = 'cpu' # NOTE: Make sure that if you use cuda that it also runs on CPU
    img_size = 256 # Image will be resized to fit the model

    # Choose what feature maps to extract for the content and style loss
    # We use the ones as mentioned in Gatys et al. 2016
    content_layers = ['conv4_2']
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    # Paths    
    style_B_path = "./style_B_images/"
    style_A_path = "./style_A_images/"
    style_original_path = "./style_original_images/"
    
    results = []
    
    for index, item in enumerate(os.listdir(style_original_path)):
        # Define the channel-wise mean and standard deviation used for VGG training
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        
        # for A
        # Load style and content images as resized (spatially square) tensors
        style_img = image_loader(style_A_path + item, device=device, img_size=img_size)
        content_img = image_loader(style_original_path + item, device=device, img_size=img_size)

        loss_A = run_style_transfer(vgg_mean, vgg_std, content_img, style_img)

        # for B
        # Load style and content images as resized (spatially square) tensors
        style_img = image_loader(style_B_path + item, device=device, img_size=img_size)
        
        loss_B = run_style_transfer(vgg_mean, vgg_std, content_img, style_img)

        results.append(
            {
                "idx": index,
                "loss_A": loss_A.numpy()[0],
                "loss_B": loss_B.numpy()[0],
                "path_original": style_original_path + item,
                "path_A": style_A_path + item,
                "path_B": style_B_path + item,
            }
        )
    
    df = pd.DataFrame(results)
    df_questions = pd.read_parquet('../BLINK/Art_Style/val-00000-of-00001.parquet', engine='pyarrow')
    
    df_questions['idx'] = df_questions['idx'].str.extract(r'(\d+)').astype(int)

    merged_df = pd.merge(df, df_questions, on='idx', how='inner')

    merged_df[["idx", "path_original", "path_A", "path_B", "question", "sub_task", "choices", "answer", "loss_A", "loss_B" ]].to_csv("val_art_style_with_losses.csv")
    
