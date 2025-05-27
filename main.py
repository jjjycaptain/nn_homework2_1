import argparse
import copy
import time
import tensorboard
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data_loader import get_loaders
from model import get_model
from engine import train_one_epoch, evaluate
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Fine‑tune ResNet‑50 on Caltech‑101")
    # Data / logging ---------------------------------------------------------
    parser.add_argument("--data_dir", default="./dataset/caltech-101", type=str)
    parser.add_argument("--log_dir", default="./results", type=str)
    # Training hyper‑params ---------------------------------------------------
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr_backbone", default=1e-2, type=float)
    parser.add_argument("--lr_head", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--step_size", default=7, type=int)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--amp", action="store_true", help="mixed precision")
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--test", action="store_true", help="test the model")
    parser.add_argument("--ckp", default=None, type=str,
                        help="checkpoint path to load the model for testing")

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    if not args.train and args.test and not args.ckp:
        raise ValueError("Please provide a checkpoint path with --ckp to test the model.")

    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Set the GPU device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Model
    model, backbone_params, head_params = get_model("resnet50", num_classes=101)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # TensorBoard 
    log_dir = Path(args.log_dir) / f"bs{args.batch_size}_lrbb{args.lr_backbone}_lrhead{args.lr_head}_epochs{args.epochs}_gamma{args.gamma}_wd{args.weight_decay}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    if args.train:
        if args.epochs > 0:  # Training phase
            best_acc = 0.0
            best_wts = copy.deepcopy(model.state_dict())

            print("Start training on", device)
            for epoch in range(args.epochs):
                epoch_start = time.time()
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                scheduler.step()

                writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
                writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

                epoch_time = time.time() - epoch_start
                remaining_time = (args.epochs - (epoch + 1)) * epoch_time
                print(f"Epoch {epoch+1:02d}/{args.epochs} | "
                      f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
                      f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
                      f"time {epoch_time:.0f}s | remaining {remaining_time:.0f}s")

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_wts = copy.deepcopy(model.state_dict())

            # Save best model --------------------------------------------------------
            torch.save(best_wts, str(log_dir / "best_model.pth"))
            print(f"Best val accuracy: {best_acc:.4f}")

    if args.test:
        if args.ckp and not args.train:
            best_wts = torch.load(args.ckp, map_location=device)
            print(f"Loaded model from {args.ckp}")
        # Testing phase
        print("Start testing on", device)
        model.load_state_dict(best_wts)  # Load the best weights
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test accuracy: {test_acc:.4f}")
        writer.add_scalar("Test Accuracy", test_acc)

    writer.close()

if __name__ == "__main__":
    main()
