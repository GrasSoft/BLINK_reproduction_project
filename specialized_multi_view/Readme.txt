This is an implementation for evaluating LoFTR model on the Multi-view Reasoning subtask of the BLINK benchmark
Here is an example of how data looks in this dataset
------------------------
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
------------------------

The LoFTR model is loaded using the koria library. There are 2 versions available "outdoor" (which works better for outdoor scenes) and "indoor" 
(which works better for indoor scenes). Because LoFTR is not a multimodal language model, we are not using the prompts and the answers, but only
the images. 

LoFTR is a deep learning model designed for local feature matching between images. It establishes dense correspondences of keypoints across the entire image.
Because of this, we give the network the 2 images and retrieve their keypoints. In order to verify whether the camera moved left or right, we create
a custom evaluation function. This returns the average difference between all keypoints. If this value is below 0, we conclude that the camera moved
to the left and if it's above 0 to the right. We map those answers in order to correctly evaluate the model on the multi-view reasoning task.

On the validation set, the indoor model gets a 89.47% accuracy, while the outdoor model has a better 92.48% accuracy.