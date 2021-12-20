import sys
import pickle
import random

def Count_Alignment(file_path):
  char_confusion = {}

  # simulate insertion errors * -> c (use the previous char as key)
  char_insert_confusion = {}

  with open(file_path, "rb") as F:
    X = pickle.load(F)
  
    op_cnt = [0] * 3
    
    for x in X:
      #print(x)
      source, target, align = x
      source = [c for c in source]
      target = [c for c in target]
      #print(source)
      #print(target)
      align.pop(0)
      prv_i, prv_j = 0, 0
      for t, a in enumerate(align):
        i, j = a
        # remove offset (0,0)
        i, j = i-1, j-1
    
        c_i = source[i]
        c_j = target[j]
    
        if (i-prv_i) == 1 and (j-prv_j) == 1:
          # Sub path (both index moves 1 step ahead)
    #      print("time:", t, i, j)
    #      print(c_i, "->", c_j)
          try:
            char_confusion[str(i)+":"+c_i][c_j] += 1
          except:
            char_confusion[str(i)+":"+c_i] = {}
            char_confusion[str(i)+":"+c_i][c_j] = 1
          op_cnt[0] += 1
        elif (i-prv_i) == 0:
          # Insertion path (source index doesn't move)
    #      print('*', "->", c_j)
          try:
            char_confusion[str(i)+":"+'*'][c_j] += 1
          except:
            char_confusion[str(i)+":"+'*'] = {}
            char_confusion[str(i)+":"+'*'][c_j] = 1

          try:
            # insert after i
            if i == 0:
              char_insert_confusion['0:*'][c_j] += 1
            else:
              char_insert_confusion[str(i)+":"+c_i][c_j] += 1
          except:
            if i == 0:
              char_insert_confusion['*'] = {}
              char_insert_confusion['*'][c_j] = 1
            else:
              char_insert_confusion[str(i)+":"+c_i] = {}
              char_insert_confusion[str(i)+":"+c_i][c_j] = 1

          op_cnt[1] += 1
        else:
          # deletion path (target index doesn't move)
    #      print(c_i, "->", '*')
          try:
            char_confusion[str(i)+":"+c_i]['*'] += 1
          except:
            char_confusion[str(i)+":"+c_i] = {}
            char_confusion[str(i)+":"+c_i]['*'] = 1
          op_cnt[2] += 1
    
        # record previous coordinate
        prv_i, prv_j = i, j
  
  print(op_cnt)
  return char_confusion, char_insert_confusion

def add_noise3(char_confusion, char_insert_confusion, word, max_k=3):
  """
  generative process:
  1. pick a position
  2. choose a new character to replace (substittuion and deletion)
  """
  # choose how many positions to modify
  num_pos = random.sample(list(range(1,max_k+1)), k=1)[0]
  num_pos = min(len(word), num_pos)
  new_word = [c for c in word]

  is_subword = word.startswith("##")

  # for each postion, replace with a new char
  #for i in random.sample(range(len(word)), k=min(len(word), num_pos)):
  for j in range(num_pos):
    # sample a position from new_word
    i = random.sample(range(len(new_word)), k=1)[0]
    # don't replace special char
#    if i == "#":
#      continue

    # flip a coin to control insertion versus substitution
    error_type = random.sample([0,0,0,0,0,0,0,0,0,1], k=1)[0]
    if error_type == 0:
      key = str(i)+":"+new_word[i]
      if key in char_confusion:
        # replace new_word[i] with a new char based on char_confusion
        space = []
        for k,v in char_confusion[key].items():
          space += [k] * v
        c = random.sample(space, k=1)[0]
        new_word[i] = c 
    else:
      key = str(i)+":"+new_word[i]
      # replace new_word[i] with a new char based on char_confusion
      if key in char_insert_confusion:
        space = []
        for k,v in char_insert_confusion[key].items():
          space += [k] * v
        c = random.sample(space, k=1)[0]
        new_word.insert(i+1,c) 

  return "".join([c for c in new_word if c != "*"])

if __name__ == "__main__":
  outputs = Count_Alignment("char_alignment.pkl")

  with open("char_confusion.pkl", "wb") as F:
    pickle.dump(outputs, F)

  print(outputs)

  # test
  while True:
    word = input()
    confusion_set = {}
    for i in range(10000):
      cand = add_noise3(outputs[0], outputs[1], word, max_k=3)
      if cand in confusion_set:
        confusion_set[cand] += 1
      else:
        confusion_set[cand] = 1
    #print(sorted(confusion_set))
    print(sorted(confusion_set.items(), key=lambda x: x[1], reverse=True))
