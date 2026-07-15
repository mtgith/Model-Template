import json
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from trainer import Trainer
from model import Model
from dataset import SampleDataset

MODE = "optuna"
configs = {
    "lr": 1e-3,
    "dropout": 0.2,
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
    "scheduler_factor": 0.5,
    "drop_last": False,
}

def saveConfigs():
    with open((Path(configs["model_dir"]) / "configs.json"), "w") as f:
        json.dump(configs, f, indent = 4)

criterion = loss_fn()

train_dataset = SampleDataset(configs["data_dir"], "train")
val_dataset = SampleDataset(configs["data_dir"], "val")

def createDataset(batch_size):
    train_loader = DataLoader(
        dataset = train_dataset,
        num_workers = configs["num_workers"],
        batch_size = batch_size,
        shuffle = True,
        drop_last = configs["drop_last"]
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        num_workers = configs["num_workers"],
        batch_size = batch_size,
        drop_last = configs["drop_last"]
    )
    return train_loader, val_loader

def train():
    train_loader, val_loader = createDataset(configs["batch_size"])
    model = Model(configs["input_size"], configs["hidden_size"], configs["output_size"], configs["dropout"])
    optimizer = optim.AdamW(model.parameters(), lr = configs["lr"], weight_decay = configs["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = "min",       
        factor = 0.5,        
        patience = configs["scheduler_patience"],       
        threshold = configs["es_thresh"],   
        min_lr = configs["min_lr"],       
    )
    trainer = Trainer(model, configs, train_loader, val_loader, criterion, optimizer, scheduler)
    trainer.train()

def objective(trial):
    trial_configs = configs.copy()
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step = 32)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log = True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log = True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.2, 0.7)
    scheduler_patience = trial.suggest_int("scheduler_patience", 5, 10)

    trial_configs.update({
        "hidden_size": hidden_size,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience
    })

    train_loader, val_loader = createDataset(trial_configs["batch_size"])
    model = Model(trial_configs["input_size"], trial_configs["hidden_size"], trial_configs["output_size"], trial_configs["dropout"])
    optimizer = optim.AdamW(model.parameters(), lr = trial_configs["lr"], weight_decay = trial_configs["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode = 'min', 
        min_lr = trial_configs["min_lr"],      
        factor = trial_configs["scheduler_factor"],        
        patience = trial_configs["scheduler_patience"],         
        threshold = trial_configs["es_thresh"]
    )
    trainer = Trainer(model, trial_configs, train_loader, val_loader, criterion, optimizer, scheduler)
    val_loss = trainer.train(trial)
    return val_loss

if MODE == "optuna":
    study = optuna.create_study(
        direction = "minimize",
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 5,
            n_warmup_steps = 20
        )
    )
    study.optimize(objective, n_trials = 50)

    print(f"Best loss: {study.best_value}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(key, value)
    configs.update(study.best_params)
    train()                           # Train with best parameters
else:
    train()

saveConfigs()