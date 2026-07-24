import json
from pathlib import Path

import optuna
from torch import optim
from torch.utils.data import DataLoader

from dataset import SampleDataset
from model import Model
from trainer import Trainer

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
        json.dump(configs, f, indent=4)


def lossFn():
    None


criterion = lossFn
train_dataset = SampleDataset(configs["data_dir"], "train")
val_dataset = SampleDataset(configs["data_dir"], "val")


def createDataset(batch_size):
    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=configs["drop_last"],
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        drop_last=configs["drop_last"],
    )
    return train_loader, val_loader


def train(cfgs, trial=None):
    train_loader, val_loader = createDataset(cfgs["batch_size"])
    model = Model(
        cfgs["input_size"],
        cfgs["hidden_size"],
        cfgs["output_size"],
        cfgs["dropout"],
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=cfgs["lr"], weight_decay=cfgs["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=cfgs["scheduler_patience"],
        threshold=cfgs["es_thresh"],
        min_lr=cfgs["min_lr"],
    )
    trainer = Trainer(
        model, cfgs, train_loader, val_loader, criterion, optimizer, scheduler
    )
    train_loss, val_loss = trainer.train(trial)
    return train_loss, val_loss


def objective(trial):
    trial_configs = configs.copy()
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.2, 0.7)
    scheduler_patience = trial.suggest_int("scheduler_patience", 5, 10)

    trial_configs.update(
        {
            "lr": lr,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "scheduler_factor": scheduler_factor,
            "scheduler_patience": scheduler_patience,
        }
    )
    train_loss, val_loss = train(trial_configs, trial)
    print(
        f"\nTrial Number: {trial.number} | Train Loss: {train_loss} | Validation Loss: {val_loss}"
    )
    return train_loss, val_loss


if MODE == "optuna":
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )
    study.optimize(objective, n_trials=50)

    print(f"Best loss: {study.best_value}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(key, value)
    configs.update(study.best_params)
    train(configs)  # Train with best parameters
else:
    train(configs)

saveConfigs()
