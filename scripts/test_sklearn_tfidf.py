import re
import numpy as np

def sklearn_tokenize(text: str):
    # This matches exactly scikit-learn's (?u)\b\w\w+\b
    pattern = re.compile(r"(?u)\b\w\w+\b")
    words = pattern.findall(text.lower())
    
    tokens = list(words)
    # bigrams
    for i in range(len(words) - 1):
        tokens.append(words[i] + " " + words[i + 1])
    return tokens

print(sklearn_tokenize("I went for a run this morning"))
