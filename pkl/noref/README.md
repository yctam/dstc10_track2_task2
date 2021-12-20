confusion_pairs.txt consists of a pair of words by enumerating all possible pairs from each sausage in a confusion network.
This file is then used to perform letter-level alignment using nltk.edit_distance.

1) dump_alignment.py: Input: confusion_pairs.txt, Output: char_alignment.pkl
2) prepare_char_confusion.py: Input: char_alignment.pkl, Output: char_confusion.pkl, which is used in src/dataset.py to simulate errors.
