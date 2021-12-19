import argparse
import glob
import logging
import os
import random
import shutil
import json
import string
import re

from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np
import sklearn
import sklearn.metrics
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from .dataset import (
    KBClusterClassificationDataset,
    SPECIAL_TOKENS
)
from .models import BertDoubleHeadsForSequenceClassification, RobertaDoubleHeadsForSequenceClassification
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from .utils.model import (
    run_batch_cls_loss,
    run_batch_cls_lm_loss,
)
from .utils.data import write_detection_preds, write_intent_preds


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def get_classes(task, architectures):
    """
    Knowledge cluster classification task.
    You may introduce other tasks here.
    """
    if task.lower() == "clusterid_classification":
        if architectures.startswith("Bert"):
            return KBClusterClassificationDataset, BertDoubleHeadsForSequenceClassification, run_batch_cls_lm_loss, run_batch_cls_lm_loss
        if architectures.startswith("Roberta"):
            # Not working yet
            return KBClusterClassificationDataset, RobertaDoubleHeadsForSequenceClassification, run_batch_cls_lm_loss, run_batch_cls_lm_loss
    else:
        raise ValueError("args.task not in ['clusterid_classification'], got %s" % task)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join("runs", args.exp_name) if args.exp_name else None
        tb_writer = SummaryWriter(log_dir)
        args.output_dir = log_dir

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # for reproducibility

    for _ in train_iterator:
        local_steps = 0
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            loss, _, _, _ = run_batch_fn_train(args, model, batch)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            #print(loss.item())
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss/local_steps)

        results = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=str(global_step))
        if args.local_rank in [-1, 0]:
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)
        
            checkpoint_prefix = "checkpoint"
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
        
            logger.info("Saving model checkpoint to %s", output_dir)
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
                json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))
            logger.info("Saving model checkpoint to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / local_steps


def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, desc="") -> Dict:
    result = None

    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle variable number of candidates
    if args.task == "selection":
        args.eval_batch_size = 1
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and (args.task != "selection" or eval_dataset.args.eval_all_snippets):
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            loss, lm_logits, mc_logits, mc_labels = run_batch_fn(args, model, batch)

            if args.task in ["clusterid_classification"]:
                data_infos.append(batch[-1])

            # (B, K)
            all_preds.append(mc_logits.detach().cpu().numpy())

            # (B)
            all_labels.append(mc_labels.detach().cpu().numpy())
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    if args.task == "clusterid_classification":
        def softmax(x):
            max_x = np.max(x,axis=-1,keepdims=True) #returns max of each row and keeps same dims
            e_x = np.exp(x - max_x) #subtracts each row with its max value
            z = np.sum(e_x,axis=-1,keepdims=True) #returns sum of each row and keeps same dims
            f_x = e_x / z
            return f_x

        # concat all samples along axis 0 (batch axis)
        # (N)
        all_labels = np.concatenate(all_labels)

        # (N, num_labels)
        all_logits = np.concatenate(all_preds)

        # (N, num_labels) Prob over all cluster IDs
        all_probs = softmax(all_logits)

        # taker top-1 -> (N)
        all_pred_ids = np.argmax(all_logits, axis=-1)
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)

        # calculate recall@k: sort and then reverse
        sorted_pred_ids = np.argsort(all_logits, axis=-1)[:, ::-1]
        sorted_logits = np.sort(all_logits, axis=-1)[:, ::-1]

        N = sorted_pred_ids.shape[0]
        K = sorted_pred_ids.shape[1]
        hit5 = 0
        hit10 = 0
        for i in range(N):
            if all_labels[i] in sorted_pred_ids[i,:5]:
                hit5 += 1
            if all_labels[i] in sorted_pred_ids[i,:10]:
                hit10 += 1
        recall5 = hit5 / N
        recall10 = hit10 / N

        # calculate detection metric (target=True/False)
        # all cluster
        ktd_all_labels = (all_labels != 0)
        ktd_all_pred_ids = (all_pred_ids != 0)

        ktd_accuracy = np.sum(ktd_all_pred_ids == ktd_all_labels) / len(ktd_all_labels)
        ktd_precision = sklearn.metrics.precision_score(ktd_all_labels, ktd_all_pred_ids)
        ktd_recall = sklearn.metrics.recall_score(ktd_all_labels, ktd_all_pred_ids)
        ktd_f1 = 2.0*ktd_precision*ktd_recall / (ktd_precision+ktd_recall)

        result = {"loss": eval_loss, "accuracy": accuracy, "recall@5": recall5, "recall@10": recall10, "N": N, "ktd_accuracy": ktd_accuracy, "ktd_precision": ktd_precision, "ktd_recall": ktd_recall, "ktd_f1": ktd_f1}

        # dump KTD output
        if args.output_file:
            # 1.0 - Prob(cluster0) = Pr(Target=True)
            write_detection_preds(eval_dataset.dataset_walker, args.output_file + "ktdonly.json", data_infos, ktd_all_pred_ids, np.log(1.0-all_probs[:,0]))
            write_intent_preds(eval_dataset.dataset_walker, args.output_file, data_infos, sorted_pred_ids, sorted_logits)
    else:
        raise ValueError("args.task not in ['clusterid_classification'], got %s" % args.task)

    if args.local_rank in [-1, 0] and not args.no_labels:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
# START: Added new flag
    parser.add_argument("--last_turn_only", action="store_true",
                        help="Use only the last's system and user turn for training and evaluation")
    parser.add_argument("--val_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{val_dataset}")
    parser.add_argument("--field", type=str, default="text",
                        help="The text field to read the system and user utterance")
    parser.add_argument("--check_step", type=int, default=300,
                        help="Check val set performance every check_steps.")
    parser.add_argument("--disfluency_dict", type=str, default=None,
                        help="A list of disfluent keywords to be removed from utterance.")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of labels for classifcation task. 2: binary class. >2: multi-class")
    parser.add_argument("--word_confusion", type=str, default=None,
                        help="word confusion (ref->hyp) dictionary as a pickle file (only for training)")
    parser.add_argument("--word_confusion2", type=str, default=None,
                        help="word confusion (raw confusion word pairs before tokenization) (ref->hyp) dictionary as a pickle file (only for training)")
    parser.add_argument("--char_confusion", type=str, default=None,
                        help="From char confusion matrix, randomly generate word confusion via substituion and deletion")
    parser.add_argument("--lm_weight", type=float, default=0.1,
                        help="LM weight during addnoise training")
# END
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--history_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--knowledge_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for knowledge, will override that value in config.")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                        help="knowledge file name.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--no_labels", action="store_true",
                        help="Read a dataset without labels.json. This option is useful when running "
                             "knowledge-seeking turn detection on test dataset where labels.json is not available.")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                        "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--negative_sample_method", type=str, choices=["all", "mix", "oracle"],
                        default="", help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_all_snippets", action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    verify_args(args, parser)

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)

        update_additional_params(params, args)
        args.update(params)
        args = Namespace(**args)
    
    args.params = params # used for saving checkpoints
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
    set_default_dataset_params(dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task

    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.eval_only:
        config = AutoConfig.from_pretrained(args.checkpoint)
        dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(args.task, config.architectures[0])

        args.output_dir = args.checkpoint
        model = model_class.from_pretrained(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)

        dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(args.task, config.architectures[0])

        # set output_past to False for DataParallel to work during evaluation
        config.output_past = False
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        tokenizer.add_special_tokens(SPECIAL_TOKENS)

        # Set number of labels (default in Bert config is 2. For KB cluster classification, we have many labels.)
        config.num_labels = args.num_labels

        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    if not args.eval_only:
        # Added new flags: last_turn_only, disfluency_dict, word_confusion, word_confusion2, char_confusion
        train_dataset = dataset_class(dataset_args, tokenizer, split_type="train", text=args.field, last_turn_only=args.last_turn_only, disfluency_dict=args.disfluency_dict, word_confusion=args.word_confusion, word_confusion2=args.word_confusion2, char_confusion=args.char_confusion)
        eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.val_dataset, text=args.field, last_turn_only=args.last_turn_only, disfluency_dict=args.disfluency_dict)

        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train, run_batch_fn_eval)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            with open(os.path.join(args.output_dir, "params.json"), "w") as jsonfile:
                json.dump(params, jsonfile, indent=2)

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model.to(args.device)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.eval_dataset, labels=not args.no_labels, labels_file=args.labels_file, text=args.field, last_turn_only=args.last_turn_only, disfluency_dict=args.disfluency_dict)
        result = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=args.eval_desc or "eval from the last epoch")

    return result


if __name__ == "__main__":
    main()
