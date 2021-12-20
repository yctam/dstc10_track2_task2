import sys
import re
import pickle

def Count_Confusion(mesh_file, confusion_cnt):
  with open(mesh_file, "r") as F:
    ref_word = ""
    for line in F:
      tokens = line.strip().lower().split()
      if re.match(r"^reference \d+", line):
        ref_word = tokens[-1]
        # remove trailing puncts
        ref_word = re.sub(r"[,?!.()]", "", ref_word)
      elif re.match(r"^hyps \d+", line):
        hyp_word = tokens[2]

        # remove trailing puncts
        hyp_word = re.sub(r"[,?!.()]", "", hyp_word)

        if hyp_word == ref_word:
          continue

        # count (ref_word,hyp_word) confusion pair
        if ref_word not in confusion_cnt:
          confusion_cnt[ref_word] = {}

        cnt_ref = confusion_cnt[ref_word]
        if hyp_word not in cnt_ref:
          cnt_ref[hyp_word] = 1
        else:
          cnt_ref[hyp_word] += 1
        #print(ref_word, hyp_word)

def Count_Confusion2(mesh_file, confusion_cnt):
  """
  This version doesn't require reference. It treats each word in a confusion set to be confusion with each other.
  We don't need to know which word is "correct" or not in a given utterance
  This way will collect more word-> wordlist confusions in a dictionary
  Count co-occurrence of word pairs in each confusion set
  """
  with open(mesh_file, "r") as F:
    ref_word = ""
    for line in F:
      # align 4 hotel 1.999997793163275 motel 2.206835809191476e-06 otel 9.157958331441725e-13
      confusion_set = []
      if re.match(r"^align \d+", line):
        tokens = line.strip().lower().split()
        for i in range(2, len(tokens)-2):
          if i%2 == 0:
            confusion_set.append(tokens[i])

        # skip empty confusion
        if len(confusion_set) <= 1:
          continue

        for wi in confusion_set:
          if wi not in confusion_cnt:
            confusion_cnt[wi] = {}
          cnt_ref = confusion_cnt[wi]
          for wj in confusion_set:
            if wi == wj:
              continue
            if wj not in cnt_ref:
              cnt_ref[wj] = 1
            else:
              cnt_ref[wj] += 1

if __name__ == '__main__':
  mesh_list = sys.argv[1]
  out_file = sys.argv[2]

  confusion_cnt = {}

  with open(mesh_list, "r") as F:
    for path in F:
      # reference is used
      #Count_Confusion(path.strip(), confusion_cnt)

      # reference is not used
      Count_Confusion2(path.strip(), confusion_cnt)

  print(confusion_cnt["the"])
  #print(confusion_cnt)

  with open(out_file, "wb") as F:
    pickle.dump(confusion_cnt, F)
