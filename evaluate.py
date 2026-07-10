import json
import torch
from pathlib import Path
from model import Model
from dataset import SampleDataset
from torch.utils.data import DataLoader

with open("configs.json", "r") as f:
    configs = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(configs["input_size"], configs["hidden_size"], configs["output_size"]).to(device)
weights = torch.load((Path(configs["model_dir"]) / configs["model_name"]), weights_only = True, map_location = device)
model.load_state_dict(weights)
model.eval()

test_dataset = SampleDataset(configs["data_dir"], "test")
test_loader = DataLoader(
    dataset = test_dataset,
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    shuffle = True
)

def evalModel():
    test_loss = 0 

    criterion = loss_fn()
    for input, target in test_loader:
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            test_loss += loss.item()

            """
            For per-sample loss do this:
            Define outside of loop under
            1) total_loss = 0.0            
            2) total_items = 0

            Compute in loop
            1) total_loss += loss.item() * inputs.size(0)
            2) total_items += inputs.size(0)

            Compute outside of loop
            1) avg_per_item_loss = total_loss / total_items
            """

    avg_loss = test_loss / len(test_loader)
    return avg_loss

evaluation = evalModel()