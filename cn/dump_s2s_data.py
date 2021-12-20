import pickle
import re
import sys

pkl = sys.argv[1]

#with open("confusion_word_cnt.pkl", "rb") as F:
#with open("confusion_word_cnt_task12.pkl", "rb") as F:
with open(pkl, "rb") as F:
  cn_dict = pickle.load(F)

for word,confusion in cn_dict.items():
  if word == "*delete*":
    continue

  # remove trailing puncts
  word = re.sub(r"[,?!.()]", "", word)

  source = [c for c in word]

  for cand in confusion:
    if cand == "*delete*":
      continue
    target = [c for c in cand]
    print("{}\t{}".format(" ".join(source), " ".join(target)))
