import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import random
import time 
import argparse
import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding
import dataset, models, metric
from tqdm import tqdm
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def get_argument_parser():
    parser = argparse.ArgumentParser("cross-prompt automated essay scoring.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--prompt_id", type=int, default=1)
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--model_checkpoint", type=str, default="./bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--n_epochs", type=int, default=5)

    return parser.parse_args()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, data_loader, loss_fn, optimizer, scheduler):
    st = time.time()
    losses = []
    model.train()
    scaler = amp.GradScaler()

    for batch in data_loader:
        batch = {k: v.cuda() for k,v in batch.items()}

        optimizer.zero_grad()

        y_pred = model(batch['input_ids'], batch['attention_mask'])
        y_true = batch['y_true']
        loss = loss_fn(y_pred, y_true)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach())

    return torch.stack(losses).mean(), time.time() - st

########
@torch.no_grad()
def evaluate(model, data_loader, loss_fn):
    st = time.time()
    losses = []
    y_preds = []
    y_trues = []
    prompt_ids = []
    model.eval()

    for batch in data_loader:
        batch = {k: v.cuda() for k,v in batch.items()}

        y_pred = model(batch['input_ids'], batch['attention_mask'])
        y_true = batch['y_true']
        prompt_id = batch['prompt_id']
        loss = loss_fn(y_pred, y_true)

        prompt_ids.append(prompt_id.cpu().numpy())
        y_preds.append(y_pred.cpu().numpy())
        y_trues.append(y_true.cpu().numpy())
        losses.append(loss.detach())
    
    avg_loss = torch.stack(losses).mean()
    prompt_ids, y_preds, y_trues = np.concatenate(prompt_ids), np.concatenate(y_preds), np.concatenate(y_trues)
    y_preds = np.array([x * (dataset.SCORE[prompt_ids[i]][1] - dataset.SCORE[prompt_ids[i]][0]) + dataset.SCORE[prompt_ids[i]][0] \
                         for i, x in enumerate(y_preds)]).round().astype(int)
    y_trues = np.array([x * (dataset.SCORE[prompt_ids[i]][1] - dataset.SCORE[prompt_ids[i]][0]) + dataset.SCORE[prompt_ids[i]][0] \
                         for i, x in enumerate(y_trues)]).astype(int)
    
    qwks = []
    for pid in range(1, 9):
        y_preds_p = y_preds[prompt_ids==pid]
        y_trues_p = y_trues[prompt_ids==pid]
        if len(y_preds_p) == 0:
            continue
        qwk = metric.quadratic_weighted_kappa(y_preds_p, y_trues_p)
        qwks.append(qwk)
    
    return avg_loss, np.mean(qwks), time.time() - st    

def main():
    args = get_argument_parser()
    print(args)
    set_random_seed(args.random_seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    datasets = dataset.load_datasets(args.prompt_id, args.fold_id, tokenizer)
    collator_fn = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    train_loader = DataLoader(datasets['train'], args.batch_size, shuffle=True, collate_fn=collator_fn)
    dev_loader = DataLoader(datasets['dev'], args.batch_size, shuffle=False, collate_fn=collator_fn)
    test_loader = DataLoader(datasets['test'], args.batch_size, shuffle=False, collate_fn=collator_fn)
    model = models.BertRegressor(n_fc_layers=2, dropout=0.0).cuda()
    optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate) # weight decay default = 0.0
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
    num_warmup_steps = len(train_loader) * 1
    num_training_steps = len(train_loader) * args.n_epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
    )
    loss_fn = torch.nn.MSELoss()

    best_dev_qwk, best_test_qwk = -1., -1
    for epoch in range(args.n_epochs):
        train_loss, train_time = train(model, train_loader, loss_fn, optimizer, scheduler)
        dev_loss, dev_qwk, dev_time = evaluate(model, dev_loader, loss_fn)
        test_loss, test_qwk, test_time = evaluate(model, test_loader, loss_fn)
        if dev_qwk > best_dev_qwk:
            best_dev_qwk = dev_qwk
            best_test_qwk = test_qwk

        print(f"epoch {epoch}; train loss {train_loss:.4f}; dev_loss {dev_loss:.4f}; test_loss {test_loss:.4f}; dev_qwk {dev_qwk:.3f}; test_qwk {test_qwk:.3f}; best_dev_qwk: {best_dev_qwk:.3f}; best_test_qwk: {best_test_qwk:.3f}; ellapsed time {train_time+dev_time+test_time:.2f}")

    print() 
    return 


if __name__ == '__main__':
    main()