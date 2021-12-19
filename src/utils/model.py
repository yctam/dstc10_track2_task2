import torch
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)

# Look /gpfsnyu/scratch/yt2267/anaconda3/envs/dstc10_task1/lib/python3.8/site-packages/transformers/modeling_bert.py, Line 1196
# for the format of a returned tuple
def run_batch_cls_loss(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, attention_mask, lm_labels, labels = batch
    model_outputs = model(
        input_ids=input_ids, token_type_ids=token_type_ids,
        attention_mask=attention_mask, labels=labels
    )
    cls_loss = model_outputs[0]
    cls_logits = model_outputs[1]
    return cls_loss, None, cls_logits, labels

def run_batch_cls_lm_loss(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, attention_mask, lm_labels, labels = batch
    model_outputs = model(
        input_ids=input_ids, token_type_ids=token_type_ids,
        attention_mask=attention_mask, labels=labels, lm_labels=lm_labels
    )
    lm_loss = model_outputs[0]
    cls_loss = model_outputs[1]
    cls_logits = model_outputs[2]
    if args.lm_weight == 0.0:
        # This means we disable the error correction part
        return cls_loss, None, cls_logits, labels
    return (cls_loss+args.lm_weight*lm_loss), None, cls_logits, labels
