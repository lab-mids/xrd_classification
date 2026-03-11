import torch
def mix_xrd_patterns(x1, x2, alpha=None):
    if alpha is None:
        alpha = torch.rand(1).item() * 0.5 + 0.25  # mix between 0.25–0.75

    x_mix = alpha * x1 + (1 - alpha) * x2
    x_mix = x_mix / x_mix.max()  # normalize

    return x_mix
def create_mixed_batch(loader, device):
    x_batch, y_batch = next(iter(loader))
    x_batch = x_batch.to(device).float()
    y_batch = y_batch.to(device)

    B = x_batch.size(0)

    perm = torch.randperm(B)

    x_mixed = []
    label_pairs = []

    for i in range(B):
        x_mix = mix_xrd_patterns(x_batch[i], x_batch[perm[i]])
        x_mixed.append(x_mix.unsqueeze(0))
        label_pairs.append((y_batch[i].item(), y_batch[perm[i]].item()))

    return torch.cat(x_mixed), label_pairs
import torch.nn.functional as F

@torch.no_grad()
def predict_topk(model, x, k=2):
    model.eval()
    _, _, logits, _ = model(x)
    probs = F.softmax(logits, dim=1)

    top_probs, top_classes = torch.topk(probs, k=k, dim=1)
    return top_classes, top_probs
