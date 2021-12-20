from nltk.metrics import edit_distance, edit_distance_align
import pickle
import re

if __name__ == '__main__':
  alignments = []
  confusion_pair = []

  with open("confusion_pairs.txt", "r") as F:
    for line in F:
      source, target = line.strip().split("\t")
      s = re.sub(" ", "", source)
      t = re.sub(" " , "", target)
      a = edit_distance_align(s, t)
      alignments.append((s, t, a.copy()))

  with open("char_alignment.pkl", "wb") as F:
    pickle.dump(alignments, F)
