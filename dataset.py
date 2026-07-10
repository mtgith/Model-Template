import torch
from pathlib import Path
from torch.utils.data import Dataset

class SampleDataset(Dataset):
    def __init__(self, data_dir: str, mode: str):
        root = Path(data_dir)
        if mode == "train":
            dir = root / "Training"
        elif mode == "val":
            dir = root / "Validation"
        else:
            dir = root / "Testing"
        self.data = sorted(dir.glob("*.pt"))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input, target = torch.load(self.data[idx])
        return input, target