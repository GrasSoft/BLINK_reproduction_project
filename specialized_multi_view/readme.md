# LoFTR Evaluation on BLINK Multi-view Reasoning

This repository contains an implementation for evaluating the LoFTR model on the Multi-view Reasoning subtask of the BLINK benchmark.

## Overview

LoFTR (Local Feature Matching with Transformers) is a deep learning model designed for local feature matching between images. It establishes dense correspondences of keypoints across the entire image, making it well-suited for analyzing camera motion between video frames.

## Dataset Description

The BLINK Multi-view Reasoning dataset consists of image pairs extracted from videos showing static scenes with camera movement. Each sample includes:

- Two images: one from the beginning and one from the end of a video
- A task to determine camera movement direction (clockwise/left or counter-clockwise/right)
- Multiple choice answers (A) left or (B) right

### Example Data Format

```
Prompt: The images are frames from a video. The video is shooting a static scene. 
The camera is either moving clockwise (left) or counter-clockwise (right) around the object. 
The first image is from the beginning of the video and the second image is from the end. 
Is the camera moving left or right when shooting the video? Select from the following options.
(A) left
(B) right

Choices:
A. left
B. right

Correct Answer: (A)
```

## Model Implementation

### LoFTR Model Loading

The LoFTR model is loaded using the kornia library with two available versions:

- **"outdoor"**: Optimized for outdoor scenes
- **"indoor"**: Optimized for indoor scenes

### Evaluation Approach

Since LoFTR is not a multimodal language model, our evaluation approach focuses solely on the images, ignoring the textual prompts and answers. The evaluation pipeline works as follows:

1. **Feature Extraction**: Input both images to the LoFTR model to extract dense keypoint correspondences
2. **Motion Analysis**: Calculate the average difference between all matched keypoints
3. **Direction Classification**: 
   - If average difference < 0: Camera moved left (clockwise)
   - If average difference > 0: Camera moved right (counter-clockwise)
4. **Answer Mapping**: Map the directional predictions to the corresponding multiple choice answers

### Custom Evaluation Function

The core evaluation function computes the average horizontal displacement of keypoint matches between the two images. This displacement serves as an indicator of camera rotation direction.

## Results

Performance on the validation set:

| Model Version | Accuracy |
|---------------|----------|
| Indoor        | 89.47%   |
| Outdoor       | 92.48%   |

The outdoor model demonstrates superior performance, likely due to better generalization across various scene types in the BLINK dataset.

All of the code is written under multi_view.ipynb