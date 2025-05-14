import os 
import pandas as pd
import cv2
import numpy as np
import ast

df = pd.read_csv("val_art_style_with_losses.csv")


count = 0
count_correct = 0
tc = 0

for index, row in df.iterrows():
    if ("A" in row["answer"] and row["loss_A"] < row["loss_B"]) or ("B" in row["answer"] and row["loss_A"] < row["loss_B"]):
        count_correct += 1
        
    count += 1
    tc += 1
    
    
print(f"There were this many correct guesses: {count_correct}")
print(f"There were this many items (to assess): {count}")
print(f"There were this many items (total): {tc}")
print(f"Accuracy: {count_correct / count}")