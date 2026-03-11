
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
import plotly.express as px
from captum.attr import GradientShap
from collections import defaultdict
import matplotlib.pyplot as plt

def compute_mean_xrd_patterns_by_class(
    dataloader,
    class_names,
    max_patterns_per_class=200,
    device="cpu"
):
    num_classes = len(class_names)
    collected = {i: [] for i in range(num_classes)}

    for x, y in dataloader:
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        for xi, yi in zip(x, y):
            if len(collected[yi]) < max_patterns_per_class:
                collected[yi].append(xi)

        if all(len(collected[i]) >= max_patterns_per_class for i in range(num_classes)):
            break

    mean_patterns = {}

    for cls in range(num_classes):
        patterns = np.array(collected[cls])  # (N, input_len)
        mean_patterns[cls] = patterns.mean(axis=0)

    return mean_patterns


def get_test_samples(test_loader, num_samples=32):
    xs, ys = [], []

    for x, y in test_loader:
        # x must be RAW XRD
        # expected shape: (B, 1, 2048)
        xs.append(x)
        ys.append(y)

        if torch.cat(xs).shape[0] >= num_samples:
            break

    x_test = torch.cat(xs)[:num_samples]
    y_test = torch.cat(ys)[:num_samples]

    return x_test, y_test