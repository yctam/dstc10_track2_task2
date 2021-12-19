import os
import re
import json
import random
import logging

from tqdm import tqdm


logger = logging.getLogger(__name__)

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def remove_articles(_text):
    return RE_ART.sub(' ', _text)


def white_space_fix(_text):
    return ' '.join(_text.split())


def remove_punc(_text):
    return RE_PUNC.sub(' ', _text)  # convert punctuation to spaces


def lower(_text):
    return _text.lower()


def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace. """
    return white_space_fix(remove_articles(remove_punc(lower(text))))


def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))
    
    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]

    return arrays


def truncate_sequences(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences

    while words_to_cut > len(sequences[0]):
        words_to_cut -= len(sequences[0])
        sequences = sequences[1:]
    
    sequences[0] = sequences[0][words_to_cut:]
    return sequences


def write_detection_preds(dataset_walker, output_file, data_infos, pred_ids, probs):
    # Flatten the data_infos
    data_infos = [
        {"dialog_id": info["dialog_ids"][i]}
        for info in data_infos
        for i in range(len(info["dialog_ids"]))
    ]

    labels = [{"target": False}] * len(dataset_walker)

    # Update the dialogs with detection result
    for info, pred_id, score in zip(data_infos, pred_ids, probs):
        dialog_id = info["dialog_id"]
        # Add prediction prob
        label = {"target": bool(pred_id), "score": float(score)}
        labels[dialog_id] = label

    if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as jsonfile:
        logger.info("Writing predictions to {}".format(output_file))
        json.dump(labels, jsonfile, indent=2)


def write_intent_preds(dataset_walker, output_file, data_infos, sorted_pred_ids, sorted_logits, topk=5):
    # Flatten the data_infos
    data_infos = [
        {
            "dialog_id": info["dialog_ids"][i],
        }
        for info in data_infos
        for i in range(len(info["dialog_ids"]))
    ]

    labels = [label for log, label in dataset_walker]
    logs = [log for log, label in dataset_walker]
    new_labels = [{"target": False}] * len(dataset_walker)
    # Update the dialogs with selected knowledge
    for info, sorted_pred_id, sorted_logit in zip(data_infos, sorted_pred_ids, sorted_logits):
        dialog_id = info["dialog_id"]

        results = []
        for pred_id, score in zip(sorted_pred_id[:topk], sorted_logit[:topk]): 
            snippet = {
                "clusterid": int(pred_id),
                "score": float(score),
            }
            results.append(snippet)
        new_label = {"target": bool(results[0]["clusterid"] != 0), "cluster_rank": results, "dialog_id": dialog_id}
        label = labels[dialog_id]
        log = logs[dialog_id]
        if label is None:
            label = new_label
        else:
            label = label.copy()
            if "response_tokenized" in label:
                label.pop("response_tokenized")
            label.update(new_label)

        new_labels[dialog_id] = label

    if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as jsonfile:
        logger.info("Writing predictions to {}".format(output_file))
        json.dump(new_labels, jsonfile, indent=2)
