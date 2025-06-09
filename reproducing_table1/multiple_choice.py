"""
NOTE:
Originally, this function used GPT-3.5-turbo to help match model outputs to the correct multiple-choice option.
However, because GPT-3.5 is not freely accessible, we replaced it with this simple rule-based string-matching approach.
This fallback looks for any mention of (A), (B), etc., directly in the model output.
"""


def match_multiple_choice(question, options, model_output):
    """
    Simple fallback for matching the most likely multiple-choice option from model_output.

    Parameters:
    - question: str, the original question (not used here but kept for consistency).
    - options: str, the options string like "(A) Option A\n(B) Option B".
    - model_output: str, the model's full response.

    Returns:
    - A string like "(A)", "(B)", etc., if found, otherwise "(Z)".
    """
    valid_choices = ['(A)', '(B)', '(C)', '(D)', '(E)']

    # Check for exact match first
    for choice in valid_choices:
        if choice in model_output:
            return choice

    # Check for partial match like 'A' instead of '(A)'
    for choice in valid_choices:
        if choice.strip('()') in model_output:
            return choice

    return '(Z)'  # Could not confidently match an option
