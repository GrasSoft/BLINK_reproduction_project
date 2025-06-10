This implementation is for reproducing the Forensic Detection results from Table 2 of the BLINK paper.
Here is an example of the data in this subset : 
---------------------
Prompt: You are a judge in a photography competition, and now you are given the four images. Please examine the details and tell which one of them is most likely to be a real photograph.
Select from the following choices.
(A) the first image
(B) the second image
(C) the third image
(D) the fourth image

Choices:
  A. the first image
  B. the second image
  C. the third image
  D. the fourth image

Correct Answer: (A)
---------------------

The specialist model chosen by the authors is DIRE.
Some scripts are from the DIRE GitHub repository : https://github.com/ZhendongWang6/DIRE/tree/main?tab=readme-ov-file
The demo.py script was adapted for infering with this specific subset of BLINK.
The utils scripts were used directly.

I implemented a script for processing the results of DIRE. 
The BLINK dataset is not very convenient because for some images, the prompt asks for the fake image (which is easy : we take the highest probably outputed by DIRE for the 4 corresponding images). However, for some other images, the prompt asks for the real image (3 are fake).

Inference take 1'30 for the 132 samples in the forensic detection validation subset (sadly we don't have the groundtruth values for the test set).

Final accuracy is 68.94 % which is exactly the one mention in Table 2 of the BLINK paper.
