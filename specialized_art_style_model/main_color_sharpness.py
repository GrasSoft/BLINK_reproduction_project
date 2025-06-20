import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2

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
        
        w_style = 5e5
        w_content = 1

        s_loss = style_loss_mse(input_features, style_features, style_layers)
        c_loss = content_loss(input_features, style_features, content_layers)
        
        # Sum up the losses
        # loss = loss + w_style * s_loss + w_content * c_loss
        loss = loss + s_loss

    return loss

if __name__ == '__main__':
    seed_everything(101)

    from PIL import Image
    from collections import Counter
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    
    def get_dominant_colors(image_path, top_n=10, resize_dim=(100, 100)):
        """Return the top N dominant RGB colors from an image."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize(resize_dim)
        pixels = np.array(img).reshape(-1, 3)
        counter = Counter(map(tuple, pixels))
        return np.array([color for color, _ in counter.most_common(top_n)])

    def color_similarity_score(colors1, colors2):
        """Compute average minimum distance between dominant colors."""
        distances = euclidean_distances(colors1, colors2)
        
        # For each color in image1, find the closest color in image2
        min_dist1 = distances.min(axis=1)
        min_dist2 = distances.min(axis=0)

        # Average both directions
        score = (min_dist1.mean() + min_dist2.mean()) / 2
        return score
    
    def compute_sharpness(image_path):
        """Compute sharpness score using the variance of the Laplacian."""
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        img_np = np.array(img)
        laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
        sharpness = laplacian.var()
        return sharpness


    # Paths    
    style_B_path = "./style_B_images/"
    style_A_path = "./style_A_images/"
    style_original_path = "./style_original_images/"
    
    results = []
    
    for index, item in enumerate(os.listdir(style_original_path)):
        # Example usage
        # colors_img_orig = get_dominant_colors(style_original_path + item)
        
        # colors_img1 = get_dominant_colors(style_A_path + item)
        # colors_img2 = get_dominant_colors(style_B_path + item)

        # similarity_score1 = color_similarity_score(colors_img_orig, colors_img1)
        # similarity_score2 = color_similarity_score(colors_img_orig, colors_img2)


        sharpness_img_orig = compute_sharpness(style_original_path + item)
        
        sharpness_img_1 = compute_sharpness(style_A_path + item)
        sharpness_img_2 = compute_sharpness(style_B_path + item)
        
        diff1 = abs(sharpness_img_orig - sharpness_img_1)
        diff2 = abs(sharpness_img_orig - sharpness_img_2)

        name, extension = os.path.splitext(item)

        import re
        # extract number from name, e.g. "image_0" → 0
        match = re.search(r'(\d+)', name)
        

        if not match:
            continue  # skip files with no number
        idx = int(match.group(1))


        results.append(
            {
                "idx": idx + 1,
                "loss_A": diff1,
                "loss_B": diff2,
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
    
