import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import earlystopper as es

class Trainer:
    def __init__(self, model: nn.Module, configs: dict, train_loader: DataLoader, val_loader: DataLoader, criterion: nn, optimizer: optim, scheduler: optim):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = configs["num_epochs"]
        min_delta = configs["es_thresh"]
        patience = configs["es_patience"]
        model_save_path = Path(configs["model_dir"]) / configs["model_name"]
        self.early_stopper = es.EarlyStopping(patience, min_delta, model_save_path)

    def train(self):
        """
        To freeze a model for fixed feature extraction do this:
        for p in pretrained_model.parameters():
            p.requires_grad = False
        pretrained_model.eval()

        with torch.no_grad():
            features = pretrained_model(input)
        out = model(features)
        """

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for input, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for input, target in self.val_loader:
                    output = self.model(input)
                    loss = self.criterion(output, target)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"Epoch: {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | LR: {current_lr}")
            
            self.early_stopper(val_loss, self.model)
            if self.early_stopper.stopping:
                print(f"Early Stopping at epoch {epoch} / {self.num_epochs}")
                break
