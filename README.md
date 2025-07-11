# BLINK_reproduction_project

This repo contains the pieces of code and tools for the DSAIT4030 Generative Modeling reproduction project for the BLINK paper.

## Specialized Depth Model results reproduction

In the folder **specialized_depth_model** there are the tools used to reproduce the results for the specialized depth model (DepthAnything) with a claimed accuracy of **97.58** (close to human performance)

### Tools and how to use them (in the order used to reproduce results):

-   `open_parquet_depth.py`: this script opens the `.parquet` file that holds all the data in the benchmark. The images are stored in base64 encoded strings. Saves them to a folder called `depth_val_iamges`
-   `detect_red_circles.py`: this script finds the red circles that denote where the visual model should look, along with the letters A and B in their vicinity and saves a new dataframe with the results, and circle coordinates in `val_depth_with_centers.csv`, along with the path to the computed depth images. For creating depth images follow the tutorial on https://github.com/LiheYoung/Depth-Anything?tab=readme-ov-file
-   `accuracy_depth.py`: this script takes the aforementioned `.csv` and computes the accuracy of the prediction. To note, some of the images could not be properly assessed due to invisible letters and/or incomplete circles

## Specialized Art Style model creation

In the folder **specialized_art_style_model** there are the tools used to create the results for the specialized art style model (using GRAM matrices).

### Tools and how to use them (in the order used to reproduce results):

-   `open_parquet_style.py`: this script opens the `.parquet` file that holds all the data in the benchmark. The images are stored in base64 encoded strings. Saves them to folders called `style_original_images` for the images the two options need to be compared against, (which of A and B is most similar to this original image). `style_A_images` for the images that serve as the first option, A. `style_B_images` for the images that serve as the second option, B.
-   `main.py`: this script links up two other that hold helper functions (`helper_functions.py and assess_style_differences.py`) to create a `.csv` file that holds the image paths and the losses for the (dis)similarity metric score. Results are stored in `val_art_style_with_centers.csv`
-   `accuracy_art_style.py`: this script takes the aforementioned `.csv` and computes the accuracy of the prediction.

## LLaVA and BakLLaVA Experiments

In the folder **reproducing_table_1** you will find the tools to reproduce the evaluation on LLaVA and BakLLaVA models.  
Note: the experiment can be run on all tasks, subset of the tasks, or on a single task by adjusting the `--task_name` argument.

### Usage:

1. Run `test_benchmark.py` to generate predictions for either LLaVA or BakLLaVA on a subset of the tasks:

    ```bash
    python test_benchmark.py --model_name LLaVA --task_name subset

    ```

2. Run `evaluate_models.py` to evaluate predictions for either LLaVA or BakLLaVa:
    ```bash
    python evaluate_models.py --model_name LLaVA
    ```

Results will be printed to the console and saved in:

-   `saved_val_scores/`
-   `saved_val_predictions/`
-   `saved_test_predictions/`

## Gemma-3 and MiniCPM-V Experiments

The `multimodal_extended/` directory contains everything required to benchmark **Gemma-3** and **MiniCPM-V** on the BLINK validation / test splits.

### 1 · Prerequisites

Pull the models locally via _Ollama_ (or your preferred backend)

```bash
ollama pull gemma3:4b         # Gemma-3-4B
ollama pull minicpm-v:8b     # MiniCPM-V-8B
```

### 2 · Set-up

```bash
# clone / cd into multimodal_extended if not already there
cd multimodal_extended

# create an isolated environment
conda create -n blink python=3.12
conda activate blink
pip install -r requirements.txt
```

### 3 · Running an evaluation

Evaluate any checkpoint on either split:

```bash
# run the BLINK pipeline (image → choice letters)
python test_benchmark.py      \
       --model_name minicpm-v:8b \
       --task_name all           # or a task name, e.g. "Counting"

# extract the final prediction and dump a JSON file
python evaluate.py \
       --model_name minicpm-v:8b \
       --split val               # use "test" for the hidden split
```

### 4 · Outputs

Results will be printed to the console and saved in:

-   `outputs/{model_name}/`
    └─ JSON files `{task}.json` with fields
    `{idx, answer, prediction, full_prediction}`
-   `val_predictions/` and `test_predictions/`
    └─ JSON files `{model_name}.json` with field
    `{split}_{task}_{sample}`
