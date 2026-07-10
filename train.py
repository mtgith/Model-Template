import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from trainer import Trainer
from model import Model
from dataset import SampleDataset

configs = {
    "lr": 1e-4,
    "min_lr": 1e-6,
    "batch_size": 64,
    "input_size": None,
    "hidden_size": 128,
    "output_size": None,
    "num_workers": 2,
    "num_epochs": 500,
    "weight_decay": 1e-3,
    "model_dir": "model directory",
    "model_name": "model.pth",
    "data_dir": "SampleDataset",
    "es_patience": 10,
    "es_thresh": 1e-4,
    "scheduler_patience": 5,
    "shuffle": True,
    "drop_last": False
}

with open((Path(configs["model_dir"]) / "configs.json"), "w") as f:
    json.dump(configs, f, indent = 4)

train_dataset = SampleDataset(configs["data_dir"], "train")
train_loader = DataLoader(
    dataset = train_dataset,
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    shuffle = configs["shuffle"],
    drop_last = configs["drop_last"]
)

val_dataset = SampleDataset(configs["data_dir"], "val")
val_loader = DataLoader(
    dataset = val_dataset,
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    shuffle = configs["shuffle"],
    drop_last = configs["drop_last"]
)

criterion = loss_fn()
model = Model(configs["input_size"], configs["hidden_size"], configs["output_size"])
optimizer = optim.AdamW(model.parameters(), lr = configs["lr"], weight_decay = configs["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = "min",       
    factor = 0.5,        
    patience = configs["scheduler_patience"],       
    threshold = configs["es_thresh"],   
    min_lr = 1e-6,       
)
trainer = Trainer(model, configs, train_loader, val_loader, criterion, optimizer, scheduler)
trainer.train()