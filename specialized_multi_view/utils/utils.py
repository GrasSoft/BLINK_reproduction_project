import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch


def show_question_with_images(data, index):
    example = data[index]
    image_keys = ['image_1', 'image_2', 'image_3', 'image_4']
    
    images = [(key, example[key]) for key in image_keys if example[key] is not None]
    
    # display images in a row
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]  
    for ax, (label, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    
    plt.suptitle(f"Question: {example['question']}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

    # show choices and correct answer
    print("Prompt:", example["prompt"])
    print("Choices:")
    for i, choice in enumerate(example['choices']):
        print(f"  {chr(65 + i)}. {choice}")
    print("\nCorrect Answer:", example['answer'])

    if example.get('explanation'):
        print("\nExplanation:", example['explanation'])

def load_and_transform(image_path, device, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

@torch.no_grad()
def estimate_motion_direction(img0_tensor, img1_tensor, matcher):
    batch = {
        'image0': img0_tensor,
        'image1': img1_tensor
    }

    correspondences = matcher(batch)

    if correspondences['keypoints0'].shape[0] < 1:
        return 'unknown'

    # Average x displacement
    x0 = correspondences['keypoints0'][:, 0]
    x1 = correspondences['keypoints1'][:, 0]
    avg_dx = (x1 - x0).mean().item()

    # (B) stands for right, (A) stands for left
    return '(B)' if avg_dx > 0 else '(A)'
