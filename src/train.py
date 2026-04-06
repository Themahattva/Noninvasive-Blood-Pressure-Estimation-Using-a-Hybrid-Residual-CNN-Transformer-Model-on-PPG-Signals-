import os
import torch
import torch.nn as nn
from tqdm import tqdm


def huber_loss(pred, target, delta=5.0):
    return nn.HuberLoss(delta=delta)(pred, target)


def train_epoch(model, loader, optimizer, delta, device):
    model.train()
    total_loss = 0.0
    for scal, ppg, label in loader:
        scal, ppg, label = scal.to(device), ppg.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(scal, ppg)
        loss = huber_loss(pred, label, delta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * scal.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, delta, device):
    model.eval()
    total_loss = 0.0
    for scal, ppg, label in loader:
        scal, ppg, label = scal.to(device), ppg.to(device), label.to(device)
        pred = model(scal, ppg)
        loss = huber_loss(pred, label, delta)
        total_loss += loss.item() * scal.size(0)
    return total_loss / len(loader.dataset)


def train(model, train_loader, val_loader, cfg, device):
    t_cfg = cfg['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=t_cfg['lr_reduce_factor'],
        patience=t_cfg['lr_reduce_patience']
    )
    delta = t_cfg['huber_delta']
    checkpoint_path = t_cfg['checkpoint_path']
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, t_cfg['max_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, delta, device)
        val_loss = eval_epoch(model, val_loader, delta, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= t_cfg['early_stop_patience']:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, history
