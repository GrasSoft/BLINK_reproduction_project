import argparse
import json
import os


def get_prediction_file(split, model_name):
    """
    Combine the task-specific prediction files for a model on split into one single final-prediction json file.
    """
    save_path = f'saved_{split}_predictions/{model_name}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    saved = {}
    for task_name in subtasks:
        output_path = f'{output_save_folder}/{model_name}/{task_name.replace("_", " ")}.json'
        outputs = json.load(open(output_path, 'r'))[split]
        for d in outputs:
            saved[d['idx']] = d['prediction']
    json.dump(saved, open(save_path, 'w'), indent=4)
    print(f'Saved predictions for split {split} to {save_path}')
    return save_path


def eval_prediction(split, model_name):
    """
    Evaluate the model on the split and return the accuracy for all tasks and also total accuracy.
    Also computes z ratio and saves val_scores.
    """
    accu_by_task = {}
    z_count_by_task = {}
    task_numbers = {}
    errors = {}
    for task_name in subtasks:
        accu_by_task[task_name] = 0
        z_count_by_task[task_name] = 0
        task_numbers[task_name] = 0
        errors[task_name] = []

    answer_file_path = f'{split}_answers.json'
    prediction_file_path = f'saved_{split}_predictions/{model_name}.json'
    answers = json.load(open(answer_file_path, 'r'))
    predictions = json.load(open(prediction_file_path, 'r'))

    # Filter answers to only include tasks in subtasks
    filtered_answers = {idx: ans for idx, ans in answers.items()
                        if '_'.join(idx.split(split)[1][1:].split('_')[:-1]) in subtasks}

    for idx, ground_answer in filtered_answers.items():
        task = '_'.join(idx.split(split)[1][1:].split('_')[:-1])

        task_numbers[task] += 1
        prediction = predictions.get(idx, "")
        if prediction == ground_answer:
            accu_by_task[task] += 1
        else:
            errors[task].append(idx)

        if prediction.strip().lower() == 'z':
            z_count_by_task[task] += 1

    average_accu = 0

    # Print number of examples used per task
    print('\nNumber of examples per task:')
    for task in subtasks:
        print(f'  {task}: {task_numbers[task]} examples')

    for task in subtasks:
        accu_by_task[task] = accu_by_task[task] / task_numbers[task] if task_numbers[task] > 0 else 0
        z_count_by_task[task] = z_count_by_task[task] / task_numbers[task] if task_numbers[task] > 0 else 0
        average_accu += accu_by_task[task]
    average_accu = average_accu / len(subtasks)
    accu_by_task["Total"] = average_accu

    print(f'Average Accuracy of model {model_name} on BLINK split {split} over all tasks is {round(100 * average_accu, 2)}%')

    # save val scores
    if split == 'val':
        val_scores_path = f'saved_val_scores/{model_name}.json'
        os.makedirs(os.path.dirname(val_scores_path), exist_ok=True)
        scores = {}
        for task in subtasks:
            scores[f'{task}_accuracy'] = accu_by_task[task]
            scores[f'{task}_z_ratio'] = z_count_by_task[task]
        scores['overall_accuracy'] = average_accu
        json.dump(scores, open(val_scores_path, 'w'), indent=4)
        print(f'Saved val scores to {val_scores_path}')

    return accu_by_task

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='LLaVa', help="select the model name")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    print(f'Evaluating model: {model_name}')

    dataset_name = 'BLINK-Benchmark/BLINK'
    output_save_folder = 'outputs'

    subtasks = [
        'Relative_Reflectance',          # low, pixel
        # 'Spatial_Relation',              # mid, image
        # 'Jigsaw',                        # mid, crop
        # 'Semantic_Correspondence',       # high, pixel
        # 'Object_Localization',           # high, crop
        # 'Counting'                       # high, image
    ]

    # 1. Save val predictions + compute accuracy + z_ratio
    split = 'val'
    get_prediction_file(split, model_name)
    eval_prediction(split, model_name)

    # 2. Save test predictions (no accuracy possible)
    split = 'test'
    get_prediction_file(split, model_name)
    print('Test predictions saved. No accuracy computed for test split.\n')


# if __name__ == '__main__':
#     dataset_name = 'BLINK-Benchmark/BLINK'
#     output_save_folder = 'outputs'
#
#     model_name = 'LLaVa'
#
#     subtasks = [
#         # 'Relative_Reflectance',          # low, pixel
#         'Spatial_Relation',              # mid, image
#         # 'Jigsaw',                        # mid, crop
#         'Semantic_Correspondence',       # high, pixel
#         # 'Object_Localization',           # high, crop
#         'Counting'                       # high, image
#     ]
#
#     # 1. Save val predictions + compute accuracy + z_ratio
#     split = 'val'
#     get_prediction_file(split, model_name)
#     eval_prediction(split, model_name)
#
#     # 2. Save test predictions (no accuracy possible)
#     split = 'test'
#     get_prediction_file(split, model_name)
#     print('Test predictions saved. No accuracy computed for test split.\n')
