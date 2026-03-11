import torch.nn.functional as F
import torch

@torch.no_grad()
def compute_reconstruction_error(model, data_loader, device):
    model.eval()
    recon_errors = []

    for x_batch, _ in data_loader:
        x_batch = x_batch.to(device).float()
        x_hat, _, _, _ = model(x_batch)

        batch_error = F.mse_loss(x_hat, x_batch, reduction="none")
        batch_error = batch_error.mean(dim=1)

        recon_errors.append(batch_error.cpu())

    recon_errors = torch.cat(recon_errors)

    return {
        "mean_mse": recon_errors.mean().item(),
        "std_mse": recon_errors.std().item(),
        "all_errors": recon_errors
    }
