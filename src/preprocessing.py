import re


def clean_text(text):
    """
    Clean input text before feature extraction.

    Steps:
    1. Convert to lowercase.
    2. Remove all non-alphabetic characters.
    3. Remove extra whitespace.
    """

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text