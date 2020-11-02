import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from src.dataset import KWSDataset
from config import params
from src.model import KWSNet
from sklearn.model_selection import train_test_split
from src.dataset import transforms, collate_fn
import wandb
import os


filenames = []
labels = []
root_dir = "speech_commands"
subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
for dir in os.scandir(root_dir):
    if not dir.is_dir() or dir.name[0] == '_':
        continue
    for file in os.listdir(os.path.join(dir)):
        filenames.append(os.path.join(dir, file))
        labels.append(dir.name)

df = pd.DataFrame(columns=['filenames', 'labels'])
df['filenames'] = filenames
df['labels'] = labels
pos = (df['labels'] == "sheila").sum()
neg = (df['labels'] != "sheila").sum()
for i in range(pos // neg - 1):
    df = pd.concat([df, df[df['labels'] == "sheila"]])

train_x, val_x, train_y, val_y = train_test_split(df['filenames'], df['labels'],
                                                  test_size=0.2, random_state=10, stratify=df['labels'])

train_dataset = KWSDataset(train_x, train_y, transform=transforms['train'])
test_dataset = KWSDataset(val_x, val_y, transform=transforms['test'])
a = (val_y == "sheila").sum()
train_dataloader = DataLoader(train_dataset,
                              batch_size=params["batch_size"],
                              num_workers=params["num_workers"],
                              shuffle=True,
                              collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset,
                             batch_size=params["batch_size"],
                             num_workers=params["num_workers"],
                             collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = KWSNet(params)
if params["from_pretrained"]:
    model.load_state_dict(torch.load(params["model_path"]))
model.to(device)
criterion = nn.NLLLoss(torch.FloatTensor([0.03, 1]).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
num_steps = len(train_dataloader) * params["num_epochs"]
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00001)

wandb.init(project=params["wandb_name"], config=params)
wandb.watch(model, log="all", log_freq=1000)

best_loss = 100.0
start_epoch = params["start_epoch"] + 1 if params["from_pretrained"] else 1
for epoch in range(start_epoch, params["num_epochs"] + 1):
    train_epoch_loss = 0.0
    train_correct = 0.0
    val_epoch_loss = 0.0
    val_correct = 0.0

    model.train()
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets).cpu()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip_grad_norm'])
        optimizer.step()
        lr_scheduler.step()
        train_epoch_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        acc = torch.eq(preds, targets).float().sum().item()
        train_correct += acc

    model.eval()
    with torch.no_grad():
        val_losses = []
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets).cpu()
            val_epoch_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            acc = torch.eq(preds, targets).float().sum().item()
            val_correct += acc


    wandb.log({"train_loss": train_epoch_loss / len(train_dataset),
               "train_accuracy": train_correct / len(train_dataset),
               "val_loss": val_epoch_loss / len(test_dataset),
               "val_accuracy": val_correct / len(test_dataset),
               })

    if (best_loss > val_epoch_loss / len(test_dataset)):
        torch.save(model.state_dict(), f"kws_model.pth")
        best_loss = val_epoch_loss / len(test_dataset)
