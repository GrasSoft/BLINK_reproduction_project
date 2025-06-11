This implementation is for extending the Table 2 of the BLINK paper to the Jigsaw category.
Here is an example of the data in this subset :
---------------------------
Prompt: Given the first image with the lower right corner missing, can you tell which one of the second image or the third image is the missing part? Imagine which image would be more appropriate to place in the missing spot. You can also carefully observe and compare the edges of the images.
Select from the following choices.

(A) the second image
(B) the third image

Choices:
  A. the second image
  B. the third image

Correct Answer: (B)
---------------------------

There is no specialist model for this jigsaw task in particular. That explains the absence of this category in Table 2 of the paper.

My idea to still try to compare VLM and specialist performances is to use a specialist inpainting model that can reconstruct the missing region, and then compare the result to each of the two candidates to see which fits better. The issue is that there is no real specialist model for that, in the sense that reconstructing a quarter of an image is still an incredibly hard task for current state-of-the-art models.

I tried to use LaMa (https://github.com/advimman/lama?tab=readme-ov-file), which leverages fast Fourier convolutions with image-wide receptive field allowing to complete large missing areas.
Their method improved the state-of-the-art across a range of datasets when the paper was published in 2021.

I modified their Colab notebook (available in their GitHub repository) to automatically take the bottom right of the images as the mask, and I ran the inference for around 150 images from the BLINK "Jigsaw" category.

Once I have the reconstructed images, I crop the bottom right part and compute the similarity with the 2 candidates from the dataset. The one with higher similarity is considered as the better fit by the "specialized" model.

In the end, I compute the accuracy of all these reconstructions and get an accuracy of 72.8 %. For comparison, the best VLM score reported in Table 1 is 66 % for GPT-4 Turbo.