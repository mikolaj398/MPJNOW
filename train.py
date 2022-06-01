import numpy as np
from seqeval.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    LayoutLMTokenizer
)
from layoutlm import FunsdDataset, LayoutlmConfig, LayoutlmForTokenClassification
from utils import get_config, save_results


torch.manual_seed(42)

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, results_file_path):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=None,
    )

    optimizer = AdamW([
            { "params": [values for name, values in model.named_parameters()], "weight_decay": args.weight_decay },
        ], lr=5e-5
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = len(train_dataloader) * args.epochs
    )
    
    loss_sum = 0.0
    model.train()

    train_iterator = trange(int(args.epochs), desc="Epoch")
    results = []
    model.zero_grad()
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for _, batch in enumerate(epoch_iterator):
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids":  batch[2].to(args.device),
                "labels": batch[3].to(args.device),
                "bbox": batch[4].to(args.device),
            }

            outputs = model(**inputs)

            loss = outputs[0]
            loss.backward()
            loss_sum += loss.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()

        epoch_result = evaluate(
            args,
            model,
            tokenizer,
            labels,
            pad_token_label_id,
            mode="test",
        )
        print(epoch_result)
        results.append(epoch_result)

        save_results(results_file_path, results)

def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode):
    eval_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode=mode)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.batch_size,
    )

    eval_loss = 0.0
    steps_counter = 0

    preds = None
    out_label_ids = None

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids": batch[2].to(args.device),
                "labels": batch[3].to(args.device),
                "bbox": batch[4].to(args.device),
            }

            outputs = model(**inputs)

            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()

        steps_counter += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / steps_counter
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        "accuracy": accuracy_score(out_label_list, preds_list)
    }

    return results


CONFIG_PATH = 'config.json'
config = get_config(CONFIG_PATH)

labels = get_labels(config.data_dir + 'labels.txt')
num_labels = len(labels)
pad_token_label_id = CrossEntropyLoss().ignore_index

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-large-uncased", do_lower_case=True)
train_dataset = FunsdDataset(config, tokenizer, labels, pad_token_label_id, mode="train")

for param, values in config.tested_params.items():
    for value in values:
        print(f"Testing {param}={value}")
        results_file_path = f"{config.save_metric_dir}{param}_{value}_results.json"

        layoutlm_config = LayoutlmConfig.from_pretrained(
            config.model_dir,
            num_labels=num_labels,
            **{param: value},
        )

        model = LayoutlmForTokenClassification.from_pretrained(
            "microsoft/layoutlm-large-uncased",
            config=layoutlm_config,
        )

        train(config, train_dataset, model, tokenizer, labels, pad_token_label_id, results_file_path)