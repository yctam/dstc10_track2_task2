import json
import os
import sys

class DatasetWalker(object):
    def __init__(self, dataset, dataroot, labels=False, labels_file=None):
        path = os.path.join(os.path.abspath(dataroot))
            
        logs_file = os.path.join(path, dataset, 'logs.json')
        with open(logs_file, 'r') as f:
            self.logs = json.load(f)

        self.labels = None

        if labels is True:
            if labels_file is None:
                labels_file = os.path.join(path, dataset, 'labels.json')

            with open(labels_file, 'r') as f:
                self.labels = json.load(f)

    def __iter__(self):
        if self.labels is not None:
            for log, label in zip(self.logs, self.labels):
                yield(log, label)
        else:
            for log in self.logs:
                yield(log, None)

    def __len__(self, ):
        return len(self.logs)

from transformers import AutoTokenizer

data_root = sys.argv[1]
split = sys.argv[2]
K = int(sys.argv[3])
is_tok = int(sys.argv[4])
output_dir = os.path.join(data_root, split, "{}best.{}".format(str(K), is_tok))

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create output_dir
os.makedirs(output_dir, exist_ok=True)

# load resources
dataset = DatasetWalker(split, data_root, labels=True)

# make default word-to-grapheme dictionary
grapheme_dict = {}

# open the ref file
R = open(os.path.join(data_root, split, "tk_refs.txt"), "w")

i = 0
for d in dataset:
  log, label = d
  for turn in log:
    # skip system turn
    if 'nbest' not in turn:
      continue

    # process user turn
    nbest_list = turn["nbest"]

    # get the clean version of text as reference for alignment
    if "clean_text" in turn:
      clean_text = turn["clean_text"]
      if is_tok:
        clean_text = " ".join(tokenizer.tokenize(clean_text))
      R.write("{}.txt {}\n".format(str(i), clean_text))

    # Dump N-best in SRILM format
    with open(os.path.join(output_dir, str(i) + ".txt"), "w") as F:
      F.write("NBestList1.0\n")
      for j, hyp in enumerate(nbest_list):
        if j >= K:
          break
        text = hyp["hyp"]
        score = hyp["score"]

        # tokenize text
        if is_tok:
          text = " ".join(tokenizer.tokenize(text))

        F.write("({}) {}\n".format(score, text))

        for word in text.split():
          if word not in grapheme_dict:
            grapheme_dict[word] = " ".join([c for c in word])
    # next user line
    i += 1

# close file stream
R.close()

# dump the grapheme dict
with open("gdict.txt", "w") as D:
  for k,v in grapheme_dict.items():
    D.write("{}\t1.0\t{}\n".format(k,v))
