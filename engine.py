from typing import Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


def _step(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module, device, scaler: GradScaler = None):
    with torch.amp.autocast('cuda', enabled=scaler is not None):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    return outputs, loss

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, device, scaler: GradScaler = None) -> Tuple[float, float]:
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        outputs, loss = _step(model, inputs, labels, criterion, device, scaler)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device) -> Tuple[float, float]:
    model.eval()
    running_loss, running_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, loss = _step(model, inputs, labels, criterion, device)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc
