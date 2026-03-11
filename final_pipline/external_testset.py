import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class XRDDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Extract logits safely
            logits = None
            if isinstance(outputs, tuple):
                for out in outputs:
                    if isinstance(out, torch.Tensor) and out.shape[-1] == 7:
                        logits = out
                        break
            else:
                logits = outputs

            if logits is None:
                raise ValueError("Logits not found in model output")

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def evaluate_sg(model, loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            logits = None
            if isinstance(outputs, tuple):
                for out in outputs:
                    if isinstance(out, torch.Tensor) and out.shape[-1] == num_classes:
                        logits = out
                        break
            else:
                logits = outputs

            if logits is None:
                raise ValueError("Logits not found in model output")

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def get_model_predictions(model, loader, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            #  SAME LOGIC AS evaluate()
            logits = None
            if isinstance(outputs, tuple):
                for out in outputs:
                    if isinstance(out, torch.Tensor) and out.shape[-1] == 7:
                        logits = out
                        break
            else:
                logits = outputs

            if logits is None:
                raise ValueError("Logits not found in model output")

            predicted = torch.argmax(logits, dim=1)

            preds.extend(predicted.cpu().numpy())
            trues.extend(y.cpu().numpy())

    return np.array(preds), np.array(trues)

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    preds_all = []
    labels_all = []

    for x, y in loader:
        x = x.to(device)
        _, _, logits, _ = model(x)
        preds = logits.argmax(1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.numpy())

    return np.array(labels_all), np.array(preds_all)

