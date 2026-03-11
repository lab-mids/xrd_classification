import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    B = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)
    sim = reps @ reps.t() / temperature

    mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
    sim = sim.masked_fill(mask, -1e9)

    labels = torch.arange(2 * B, device=z1.device)
    labels = (labels + B) % (2 * B)

    return F.cross_entropy(sim, labels)


def supervised_contrastive_loss(z, labels, temperature=0.1):
    z = F.normalize(z, dim=1)
    B = z.size(0)

    sim = z @ z.t() / temperature
    mask = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float().masked_fill(mask, 0.0)

    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

    pos_sum = (pos_mask * log_prob).sum(dim=1)
    pos_count = pos_mask.sum(dim=1).clamp(min=1.0)

    return -(pos_sum / pos_count).mean()


def peak_preserving_loss(x_hat, x, peak_weight=5.0):
    threshold = 0.05 * x.max(dim=1, keepdim=True).values
    peak_mask = (x > threshold).float() * peak_weight + 1.0

    mse = ((x_hat - x) ** 2 * peak_mask).mean()
    l1 = (x_hat - x).abs().mean()

    return mse + 0.1 * l1


def compute_total_loss(
    recon_loss,
    clf_loss,
    smooth_loss,
    ntx_loss,
    supcon_loss,
    epoch,
    recon_mul,
    alpha_cls,
    beta_smooth,
    lambda_ntx,
    lambda_sup,
):
    contrastive_scale = 0.0
    if epoch > 10:
        contrastive_scale = min(1.0, (epoch - 10) / 30.0)

    return (
        recon_mul * recon_loss
        + alpha_cls * clf_loss
        + beta_smooth * smooth_loss
        + contrastive_scale * (lambda_ntx * ntx_loss + lambda_sup * supcon_loss)
    )