import os
import json
import random
import logging
import sys
import random
import re
import string
import pickle
import copy

from itertools import chain

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences
)

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>", "<domain>", "<entity>", "<title>", "<body>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]


class BaseDataset(torch.utils.data.Dataset):
    """
    Adapted from Alexa DSTC9 challenge
    """
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None, text="text", title="title", body="body", disfluency_dict=None, word_confusion=None, word_confusion2=None, char_confusion=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.text = text
        self.title = title
        self.body = body

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag, self.domain_tag, self.entity_tag, self.title_tag, self.body_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        # load dict if file provided (one word per line in file)
        self.disfluency_dict = {}
        if disfluency_dict:
            with open(args.disfluency, "r") as F:
                for word in F:
                    self.disfluency_dict[word.strip().lower()] = 1

        # confusion set operates at the token-id level (no need to tokenize again)
        self.word_confusion_dict = {}
        if word_confusion:
            with open(word_confusion, "rb") as F:
                x = pickle.load(F)
                for k,v in x.items():
                    # allow *delete* of a token. In this case, [MASK] token is used to replace a target
                    #cn = [self.tokenizer.convert_tokens_to_ids(c if c != "*delete*" else "[MASK]") for c in v.keys() if c != k]
                    cn = [self.tokenizer.convert_tokens_to_ids(c) for c in v.keys() if c != k]
                    if len(cn) == 0:
                        continue
                    kid = self.tokenizer.convert_tokens_to_ids(k)
                    self.word_confusion_dict[kid] = cn
                    #logger.info("confusion: {}".format(kid))
                    #logger.info(self.word_confusion_dict[kid])

        # confusion set operates at the raw word level (need to tokenize here)
        self.word_confusion_dict2 = {}
        if word_confusion2:
            with open(word_confusion2, "rb") as F:
                x = pickle.load(F)
                for k,v in x.items():
                    # tokenize each word confusion pairs: e.g. (deliver -> delver)
                    #cn = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(c if c != "*delete*" else "[MASK]")) for c in v.keys() if c != k]
                    cn = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(c)) for c in v.keys() if c != k]
                    if len(cn) == 0:
                        continue
                    kid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(k))
                    # TODO: for m-to-n case where m > 1, it requires more complicated treatment to add noise
                    # Currently, we can only handle len(kid) = 1 case.
                    self.word_confusion_dict2[tuple(kid)] = cn
                    logger.info("confusion2: {}".format(kid))
                    logger.info(self.word_confusion_dict2[tuple(kid)])

        # load char confusion dictionary
        self.char_confusion_dict = {}
        self.char_insert_confusion_dict = {}
        if char_confusion:
            with open(char_confusion, "rb") as F:
                self.char_confusion_dict, self.char_insert_confusion_dict = pickle.load(F)

        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        self.dialogs = self._prepare_conversations()

        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        self.knowledge, self.snippets = self._prepare_knowledge()

        self._create_examples()


    def _add_noise1(self, input_ids):
        """
        Add noise using word_confusion_dict
        """
        lm_labels = [-100] * len(input_ids)

        if len(self.word_confusion_dict) == 0:
            return input_ids, lm_labels

        # scan token positions that can be replaced
        positions = []
        for i in range(len(input_ids)):
            wid = input_ids[i]
            if wid in self.word_confusion_dict:
                positions.append(i)

        if len(positions) == 0:
            return input_ids, lm_labels

        # number of positions to be replaced with noisy token
        new_input_ids = input_ids.copy()
        for i in random.sample(positions, k=min(2, len(positions))):
            wid = input_ids[i]
            new_input_ids[i] = random.sample(self.word_confusion_dict[wid], k=1)[0]
            lm_labels[i] = wid
            #logger.info("Add Noise: {} -> {}".format(wid, new_input_ids[i]))

        return new_input_ids, lm_labels

    def _add_noise2(self, input_ids):
        """
        Add noise using word_confusion_dict2
        """
        lm_labels = [-100] * len(input_ids)

        if len(self.word_confusion_dict2) == 0:
            return input_ids, lm_labels

        # scan token positions that can be replaced
        positions = []
        for i in range(len(input_ids)):
            # create a tuple as a key
            wid = (input_ids[i],)
            if wid in self.word_confusion_dict2:
                positions.append(i)

        if len(positions) == 0:
            return input_ids, lm_labels

        # number of positions to be replaced with noisy tokens (can be 1-to-n)
        new_input_ids = input_ids.copy()
        for i in random.sample(positions, k=min(2, len(positions))):
            # create a tuple as a key
            wid = input_ids[i]
            new_tuple = random.sample(self.word_confusion_dict2[(wid,)], k=1)[0]
            new_input_ids[i] = new_tuple[0]
            lm_labels[i] = wid
            if len(new_tuple) > 1:
              new_input_ids[i+1:i+1] = new_tuple[1:]
              # repeat LM targets
              #lm_labels[i+1:i+1] = [-100] * (len(new_tuple)-1)
              lm_labels[i+1:i+1] = [wid] * (len(new_tuple)-1)
            logger.info("Add Noise2: {} -> {}".format(wid, new_tuple))

        assert(len(new_input_ids) == len(lm_labels))

        return new_input_ids, lm_labels

    def _add_noise3(self, input_ids, num_pos=2):
        """
        Add noise using char_confusion_dict. Randomly pick K words to replace.
        """
        def add_char_noise(char_confusion, char_insert_confusion, word, max_k=3):
          """
          generative process:
          1. pick a position
          2. choose a new character to replace (substittuion and deletion)
          """
          # choose how many positions to modify
          num_pos = random.sample(list(range(1,max_k+1)), k=1)[0]
          new_word = [c for c in word]
        
          is_subword = word.startswith("##")
        
          # for each letter position, replace with a new letter
          for i in random.sample(range(len(word)), k=min(len(word), num_pos)):
            # don't replace special char
        #    if i == "#":
        #      continue
            key = str(i)+":"+word[i]
            if key in char_confusion:
              # replace word[i] with a new char based on char_confusion
              space = []
              for k,v in char_confusion[key].items():
                space += [k] * v
              c = random.sample(space, k=1)[0]
              new_word[i] = c 
          return "".join([c for c in new_word if c != "*"])

        def add_char_noise1(char_confusion, char_insert_confusion, word, max_k=3):
          """
          Used in DSTC10 submission.
          generative process:
          1. pick a position
          2. choose a new character to replace (substittuion and deletion)
          3. deal with insertion error
          """
          # choose how many positions to modify
          num_pos = random.sample(list(range(1,max_k+1)), k=1)[0]
          new_word = [c for c in word]
        
          is_subword = word.startswith("##")
        
          # for each letter position, replace with a new letter
          for i in random.sample(range(len(word)), k=min(len(word), num_pos)):
            # don't replace special char
        #    if i == "#":
        #      continue

            # flip a coin to control insertion versus substitution
            error_type = random.sample([0,0,0,0,0,0,0,0,0,1], k=1)[0]
            if error_type == 0:
              key = str(i)+":"+word[i]
              if key in char_confusion:
                # replace word[i] with a new char based on char_confusion
                space = []
                for k,v in char_confusion[key].items():
                  space += [k] * v
                c = random.sample(space, k=1)[0]
                new_word[i] = c
            else:
              key = str(i)+":"+word[i]
              # replace word[i] with a new char based on char_confusion
              if key in char_insert_confusion:
                space = []
                for k,v in char_insert_confusion[key].items():
                  space += [k] * v
                c = random.sample(space, k=1)[0]
                new_word.insert(i+1,c)

          return "".join([c for c in new_word if c != "*"])

        def add_char_noise2(char_confusion, char_insert_confusion, word, max_k=3):
          """
          generative process:
          1. pick a position
          2. choose a new character to replace (substittuion and deletion)
          3. deal with insertion error (with a bug fix on new_word post dstc10 eval, 10/1)
             More variety can be generated after the bug fix
          """
          # choose how many positions to modify
          num_pos = random.sample(list(range(1,max_k+1)), k=1)[0]
          num_pos = min(len(word), num_pos)
          new_word = [c for c in word]
        
          is_subword = word.startswith("##")
        
          # for each letter position, replace with a new letter
          #for i in random.sample(range(len(word)), k=min(len(word), num_pos)):
          for j in range(num_pos):
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

        lm_labels = [-100] * len(input_ids)

        if len(self.char_confusion_dict) == 0:
            return input_ids, lm_labels

        positions = list(range(len(input_ids)))
        new_input_ids = input_ids.copy()

        # remove positions that are CLS or SEP or speaker labels
        skip_tokens = [self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
        positions = [i for i in positions if input_ids[i] not in skip_tokens]

        # consider positions that are from word_confusion_dict
        # scan token positions that can be replaced
        wc_positions = []
        for i in range(len(input_ids)):
            wid = input_ids[i]
            if wid in self.word_confusion_dict:
                wc_positions.append(i)

        # weird case, this should not happen
        if len(positions) == 0:
            return input_ids, lm_labels

        for i in random.sample(positions, k=min(num_pos, len(positions))):
            wid = input_ids[i]

            if i in wc_positions:
              # higher priority to learn from subword_confusion_dict since these are refernce confusions from ASR Nbest w/ ref
              new_input_ids[i] = random.sample(self.word_confusion_dict[wid], k=1)[0]
              lm_labels[i] = wid
            else:
              # decode input_ids[i] back to word otken
              word = self.tokenizer.decode(wid)
  
              # add char noise to it
              #new_word = add_char_noise(self.char_confusion_dict, self.char_insert_confusion_dict, word)
              new_word = add_char_noise1(self.char_confusion_dict, self.char_insert_confusion_dict, word)
              # 10/1 bug fix on insertion noise
              #new_word = add_char_noise2(self.char_confusion_dict, self.char_insert_confusion_dict, word)
  
              # tokenize this word, and the convert the noisy word back to ID(s)
              new_tuple = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(new_word))
  
              # create a tuple as a key
              new_input_ids[i] = new_tuple[0]
  
              lm_labels[i] = wid
              if len(new_tuple) > 1:
                new_input_ids[i+1:i+1] = new_tuple[1:]
                # Repeat LM targets (Not tested thoroughly. Should retry it in the future)
                #lm_labels[i+1:i+1] = [-100] * (len(new_tuple)-1)
                lm_labels[i+1:i+1] = [wid] * (len(new_tuple)-1)
              #logger.info("Add Noise3: {} -> {}".format(wid, new_tuple))

        assert(len(new_input_ids) == len(lm_labels))

        return new_input_ids, lm_labels

    def _tokenize(self, text):
        """
        Note: Since DSTC10 ASR transcripts are lower casing, so we intentionally lower case text
        """
#        return self.tokenizer.tokenize(text.lower())
        return self.tokenizer.tokenize(text)

    def _prepare_conversations(self):
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])): # only show progress bar in one process
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs
    
    def _prepare_knowledge(self):
        knowledge = self.knowledge_reader.knowledge
        self.knowledge_docs = self.knowledge_reader.get_doc_list()

        tokenized_snippets = dict()
        for snippet in self.knowledge_docs:
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
            knowledge = self._knowledge_to_string(snippet["doc"], name=snippet["entity_name"] or "")
            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self._tokenize(knowledge))
            tokenized_snippets[key] = tokenized_knowledge[:self.args.knowledge_max_tokens]
        return knowledge, tokenized_snippets

    def _knowledge_to_string(self, doc, name=""):
        return doc["body"]

    def _rm_disfluency(self, raw_text):
        """
        Remove disfluent tokens if disfluency dict is provided. Not used in our DSTC10 evaluation.
        """
        if self.disfluency_dict:
            tokens = [w for w in raw_text.split() if w.lower() not in self.disfluency_dict]
            return " ".join(tokens)
        return raw_text

    def _create_examples(self):
        logger.info("Creating examples")
        self.examples = []
        for dialog_session in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0]):
            dialog_id = dialog_session["id"]
            label = dialog_session["label"]
            dialog = dialog_session["log"]
            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"]

            if not target and self.args.task not in ["clusterid_classification"] and dialog_session["label"] is not None:
                # Note: clusterID = 0 will be considered as non-seeking class in clusterid classification
                continue

            history = [
                self.tokenizer.convert_tokens_to_ids(self._tokenize(self._rm_disfluency(turn[self.text])))
                for turn in dialog
            ]

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left 
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)

            if target:
                if "knowledge" not in label:
                    # when the labels.json is from knowledge-seeking turn detection,
                    # there will be no ground truth knowledge
                    # so we just use a dummy snippet here
                    label["knowledge"] = [self.knowledge_docs[0]]

                knowledge = label["knowledge"][0]
                cluster_id = knowledge["cluster_id"] if "cluster_id" in knowledge else 0
                domain = knowledge["domain"]
            else:
                cluster_id = 0
                domain = None

            self.examples.append({
                "history": truncated_history,
                "cluster_id": cluster_id,
                "domain": domain,
                "label": label,
                "dialog_id": dialog_id
            })

    def build_input_from_segments(self, knowledge):
        raise NotImplementedError
                
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)


class KBClusterClassificationDataset(BaseDataset):
    """
    word_confusion: A pickle file containing tokenized word confusion map. This means that we first apply Bert tokenizer on the N-best lists to form confusion networks. Then collect the tokenized word confusion sets.
    word_confusion2: A pickle file containing raw word confusion map. This means that we use the raw N-best lists to form confusion networks. Then collect the raw word confusion sets.
    char_confusion: A pickle file containing position-dependent char confusions, and this what we use in the paper and DSTC10 evaluation.
    We include word_confusion and word_confusion2 for future development. We can mix different types of confusions in ASR error simulation.
    """
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None, text="text", last_turn_only=False, disfluency_dict=None, word_confusion=None, word_confusion2=None, char_confusion=None):
        super().__init__(args, tokenizer, split_type, labels, labels_file, text, disfluency_dict=disfluency_dict, word_confusion=word_confusion, word_confusion2=word_confusion2, char_confusion=char_confusion)
        self.last_turn_only = last_turn_only

    def build_input_from_segments(self, history):
        """ Build a sequence of input from history """
        instance = {}

        cls = self.tokenizer.cls_token_id
        sep = self.tokenizer.sep_token_id

        if self.last_turn_only:
            # Using the last user turn for knowledge cluster classification is better than using all dialogue turns.
            sequence = [[cls] + history[-1] + [sep]]
        else:
            # add a dummy [CLS] utterance at the beginning
            sequence = [[cls]] + history

            # add speaker labels
            sequence_with_speaker = [
                [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
                for i, s in enumerate(sequence[1:])
            ]
            sequence = [sequence[0]] + sequence_with_speaker

            # modify last turn with : [SEP] last turn [SEP]
            sequence[-1] = [sep] + sequence[-1] + [sep]

        # Determine which add_noise to use
        if len(self.word_confusion_dict) > 0:
            add_noise = self._add_noise1
        elif len(self.word_confusion_dict2) > 0: 
            add_noise = self._add_noise2
        elif len(self.char_confusion_dict) > 0: 
            add_noise = self._add_noise3
        else:
            # No noise
            add_noise = lambda x: (x, [-100]*len(x))

        lm_labels = [[]] * len(sequence)
        for i in range(len(sequence)):
            j = len(sequence)-1-i
            if i%2 == 0:
                # add noise to user utterances only.
                new_input_ids, word_labels = add_noise(sequence[j])

                # update sequence and LM labels after adding noise on user turns.
                sequence[j] = new_input_ids
                lm_labels[j] = word_labels
            else:
                lm_labels[j] = [-100]*len(sequence[j])

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [0 if i % 2 else 1 for i, s in enumerate(sequence) for _ in s]
        instance["attention_mask"] = [1.0] * len(instance["input_ids"])
        instance["lm_labels"] = list(chain(*lm_labels))

        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(example["history"])
        instance["label"] = example["cluster_id"]
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        attention_mask = [ins["attention_mask"] for ins in batch]
        labels = [ins["label"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, 0))
        attention_mask = torch.tensor(pad_ids(attention_mask, 0.0), dtype=torch.float)

        # LM labels correct ASR errors from error simulation
        #lm_labels = torch.full_like(input_ids, -100)
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        #labels = torch.tensor(labels).float()
        labels = torch.tensor(labels)

        return input_ids, token_type_ids, attention_mask, lm_labels, labels, data_info
