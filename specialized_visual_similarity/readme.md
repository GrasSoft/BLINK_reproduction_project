# DinoV2 Evaluation on BLINK Visual Similarity

This repository contains an implementation for evaluating the DinoV2 model on the Visual Similarity subtask of the BLINK benchmark.

## Overview

DinoV2 (Self-Supervised Vision Transformers) is a vision foundation model that learns robust visual representations without labeled data. It produces high-quality image features that capture semantic similarities between images, making it well-suited for visual similarity tasks.

## Dataset Description

The BLINK Visual Similarity dataset consists of 3 images per sample, one reference image, and two other images with some similarities.  
The purpose is to find the image more similar to the reference one.
Each sample includes:

- Three images
- A task to determine which image is the most similar to the reference image
- Multiple choice answers (A) second image or (B) third image 

### Example Data Format

```
Prompt: Given three similar but different images, take the first image as reference. Can you tell which one of the latter two images is most similar to the first one?
Select from the following choices.

(A) the second image
(B) the third image

Choices:
  A. the second image
  B. the third image

Correct Answer: (A)
```

## Model Implementation

### DinoV2 Model Loading

The DinoV2 model is loaded using weights from torch hub library. There are 8 versions available:

| Variant        | Parameters | Register Tokens? | Classifier Head? |
|----------------|------------|------------------|------------------|
| *_vits14       | 21 M       | No               | No               |
| *_vits14_reg   | 21 M       | Yes              | No               |
| *_vitb14       | 86 M       | No               | No               |
| *_vitb14_reg   | 86 M       | Yes              | No               |
| *_vitl14       | 300 M      | No               | No               |
| *_vitl14_reg   | 300 M      | Yes              | No               |
| *_vitg14       | 1.1 B      | No               | No               |
| *_vitg14_reg   | 1.1 B      | Yes              | No               |

We tested all models, saving their results. They are showed on the last table

### Evaluation Approach

Since DinoV2 is not a multimodal language model, our evaluation approach focuses solely on the images, ignoring the textual prompts and answers. The evaluation pipeline works as follows:

1. **Feature Extraction**: Input all 3 images to the DinoV2 model to extract features
2. **Cosine Similarity**: Calculate the cosine similarity between the reference image and the other 2 candidate images
3. **Visual Similarity Classification**: 
   - We choose the second image if it has a higher similarity score with the reference image, and we choose the third otherwise 
4. **Answer Mapping**: Map the correct image prediction to the corresponding multiple choice answers

## Results

Performance on the validation set:

| Model Version | Accuracy |
|---------------|----------|
| *_vits14       | 82.96%  |
| *_vits14_reg   | 83.70%  |
| *_vitb14       | 81.48%  | 
| *_vitb14_reg   | 79.26%  |   
| *_vitl14       | 76.30%  |   
| *_vitl14_reg   | 75.56%  |   
| *_vitg14       | 74.82%  |   
| *_vitg14_reg   | 73.34%  |   


We can deduce from this table that the samlles model with registers (learned positional tokens) demonstrates superior performance, with an accuracy of 82.96%, surpassing the highest accuracy of the best multimodal language model (GPT-4 Turbo with 80.74%). This is likely due to lower resolution of images in the BLINK dataset.

All of the code is written under visual_similarity.ipynb