import torch
import torch.nn.functional as F


def xrd_augment(x):
    B, L = x.shape
    x_aug = x.clone()

    max_shift = int(0.01 * L)
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (B, 1), device=x.device)
        x_aug = torch.stack(
            [torch.roll(x_aug[i], int(shifts[i].item()), dims=0) for i in range(B)]
        )

    if torch.rand(1, device=x.device) < 0.3:
        m = int(0.02 * L)
        if m > 0:
            start = torch.randint(0, L - m, (B,), device=x.device)
            for i in range(B):
                ramp = torch.linspace(1.0, 0.0, m, device=x.device)
                x_aug[i, start[i] : start[i] + m] *= ramp

    scale = 0.9 + 0.2 * torch.rand(B, 1, device=x.device)
    x_aug = x_aug * scale

    k = 201
    if k < L:
        pad = k // 2
        kernel = torch.ones(1, 1, k, device=x.device) / k
        bg = F.conv1d(x_aug.unsqueeze(1), kernel, padding=pad).squeeze(1)
        x_aug = x_aug - 0.5 * bg

    noise = 0.01 * torch.randn_like(x_aug).unsqueeze(1)
    noise_kernel = torch.tensor([[[0.2, 0.6, 0.2]]], device=x.device)
    noise = F.conv1d(noise, noise_kernel, padding=1).squeeze(1)

    return (x_aug + noise).clamp(0.0, 1.0)