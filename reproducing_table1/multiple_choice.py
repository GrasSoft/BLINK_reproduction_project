"""
NOTE:
Originally, this function used GPT-3.5-turbo to help match model outputs to the correct multiple-choice option.
However, because GPT-3.5 is not freely accessible, we replaced it with this simple rule-based string-matching approach.
This fallback looks for any mention of (A), (B), etc., directly in the model output.
"""
import re


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

    # First pass: look for exact string match like "(A)" in the model output
    for choice in valid_choices:
        if choice in model_output:
            return choice

    # Second pass: check for partial match (e.g. 'A' instead of '(A)')
    for choice in valid_choices:
        letter = choice.strip('()')
        # match whole word: "A", not part of "AMAZING"
        if re.search(rf'\b{letter}\b', model_output):
            return choice

    # If no match was found, return fallback '(Z)'
    return '(Z)'  # Indicates no valid choice could be matched
