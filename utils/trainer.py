import torch
import torch.nn.functional as F

from .losses import (
    peak_preserving_loss,
    supervised_contrastive_loss,
    compute_total_loss,
)

#================================================
# TRAIN STEP
#================================================

def train_step(model, optimizer, x, y, epoch, params, device, use_augmentation=True):
    model.train()
    optimizer.zero_grad()

    x = x.to(device).float()
    y = y.to(device).long()

    # ----------------------------
    # Forward pass 
    # ----------------------------
    if use_augmentation:
        x_in = (x + params["noise_std"] * torch.randn_like(x)).clamp(0, 1)
    else:
        x_in = x

    x_hat, z_cls, logits, _ = model(x_in, return_latent=True)

    # ----------------------------
    # Core losses
    # ----------------------------
    recon_loss  = peak_preserving_loss(x_hat, x)
    clf_loss    = F.cross_entropy(logits, y)
    smooth_loss = model.gradient_sparsity_loss(x_hat)

    ntx_loss = torch.tensor(0.0, device=device)
    # ----------------------------
    # Supervised contrastive (NO augmentation)
    # ----------------------------
    supcon_loss = torch.tensor(0.0, device=device)

    if params["lambda_sup"] > 0:
        z_proj = model.project(z_cls)
        supcon_loss = supervised_contrastive_loss(z_proj, y)

    # ----------------------------
    # Total loss
    # ----------------------------
    loss = compute_total_loss(
        recon_loss, clf_loss, smooth_loss,
        ntx_loss, supcon_loss, epoch,
        params["recon_mul"], params["alpha_cls"], params["beta_smooth"],
        params["lambda_ntx"], params["lambda_sup"]
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "recon": recon_loss.item(),
        "clf": clf_loss.item(),
        "smooth": smooth_loss.item(),
        "ntx": float(ntx_loss),
        "supcon": float(supcon_loss),
    }


#================================================
# EVALUATION
#================================================

@torch.no_grad()
def evaluate_classifier(model, data_loader,device):
    model.eval()
    total = 0
    correct = 0

    for x, y in data_loader:
        x = x.to(device).float()
        y = y.to(device).long()

        # Forward pass (ignore reconstruction & latent)
        _, _, logits, _ = model(x)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)

    return correct / total



@torch.no_grad()
def val_step(model, val_loader, device):
    model.eval()
    total_correct = 0
    total = 0
    recon_total = 0

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        x_hat, _, logits, _ = model(x)
        recon_total += peak_preserving_loss(x_hat, x).item()

        preds = logits.argmax(1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)

    return {
        "val_acc": total_correct / total,
        "val_recon": recon_total / len(val_loader)
    }



def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        _, _, logits, _ = model(x)
        preds = logits.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total