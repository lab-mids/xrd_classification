import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def plot_external_example(model, data_loader, device, sample_index=0):
    model.eval()

    # get first batch
    x, y = next(iter(data_loader))
    x = x.to(device).float()
    y = y.to(device)

    # forward
    x_hat, _, logits, _ = model(x)

    # pick sample
    x_true = x[sample_index]
    x_recon = x_hat[sample_index]
    y_true = y[sample_index].item()

    # classification info
    probs = torch.softmax(logits[sample_index], dim=0)
    pred_class = probs.argmax().item()
    confidence = probs.max().item()

    # reconstruction error (per-sample MSE)
    recon_error = F.mse_loss(x_recon, x_true).item()

    # convert to numpy
    x_true_np = x_true.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()

    # plot
    plt.figure(figsize=(14, 4))
    plt.plot(x_true_np, label="Original XRD", linewidth=1.5)
    plt.plot(x_recon_np, label="Reconstructed XRD", linewidth=1.2)

    plt.title(
        f"External Test Example\n"
        f"True = {y_true} | Pred = {pred_class} "
        f"(conf = {confidence:.2f}) | "
        f"MSE = {recon_error:.6f}"
    )

    plt.legend()
    plt.xlabel("2θ index")
    plt.ylabel("Intensity")
    plt.tight_layout()
    plt.show()

    return {
        "true_class": y_true,
        "predicted_class": pred_class,
        "confidence": confidence,
        "reconstruction_error": recon_error
    }



@torch.no_grad()
def plot_cs_example(model, data_loader, device, class_names, sample_index=0):
    model.eval()

    # --- get first batch ---
    x, y = next(iter(data_loader))
    x = x.to(device).float()
    y = y.to(device)

    # --- forward pass ---
    x_hat, _, logits, _ = model(x)

    # --- safety check ---
    if sample_index >= x.shape[0]:
        raise ValueError(f"sample_index {sample_index} exceeds batch size {x.shape[0]}")

    # --- select sample ---
    x_true = x[sample_index]
    x_recon = x_hat[sample_index]
    y_true_idx = int(y[sample_index].item())

    # --- prediction ---
    probs = torch.softmax(logits[sample_index], dim=0)
    pred_idx = int(probs.argmax().item())
    confidence = float(probs.max().item())

    # --- class name safety ---
    if y_true_idx < len(class_names):
        true_label = class_names[y_true_idx]
    else:
        true_label = f"Unknown({y_true_idx})"

    if pred_idx < len(class_names):
        pred_label = class_names[pred_idx]
    else:
        pred_label = f"Unknown({pred_idx})"

    # --- reconstruction error ---
    recon_error = F.mse_loss(x_recon, x_true).item()

    # --- convert to numpy ---
    x_true_np = x_true.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()

    # --- plot ---
    plt.figure(figsize=(14, 4))
    plt.plot(x_true_np, label="Original XRD", linewidth=1.5)
    plt.plot(x_recon_np, label="Reconstructed XRD", linewidth=1.2)

    plt.title(
        f"Crystal System Example\n"
        f"True = {true_label} | Pred = {pred_label} "
        f"(conf = {confidence:.2f}) | "
        f"MSE = {recon_error:.6f}"
    )

    plt.legend()
    plt.xlabel("2θ index")
    plt.ylabel("Intensity")
    plt.tight_layout()
    plt.show()

    return {
        "true_label": true_label,
        "predicted_label": pred_label,
        "confidence": confidence,
        "reconstruction_error": recon_error
    }


@torch.no_grad()
def plot_example_prediction_and_reconstruction(model, val_loader, device, sample_index=0):
    model.eval()

    # --- get 1 batch ---
    x, y = next(iter(val_loader))
    x = x.to(device).float()
    y = y.to(device)

    # --- forward pass ---
    x_hat, _, logits, _ = model(x)

    # --- find sample ---
    x_true = x[sample_index].cpu().numpy()
    x_recon = x_hat[sample_index].cpu().numpy()
    y_true = y[sample_index].item()

    # predicted class
    probs = torch.softmax(logits[sample_index], dim=0).cpu().numpy()
    pred_class = probs.argmax()
    confidence = probs.max()

    # --- plot ---
    plt.figure(figsize=(14, 4))
    plt.plot(x_true, label="Original XRD", linewidth=1.5)
    plt.plot(x_recon, label="Reconstructed XRD", linewidth=1.2)
    plt.title(
        f"XRD Example\nTrue class = {y_true} | Predicted = {pred_class} "
        f"(confidence = {confidence:.2f})"
    )
    plt.legend()
    plt.xlabel("2θ index")
    plt.ylabel("Intensity")
    plt.tight_layout()
    plt.show()

    return {
        "true_class": y_true,
        "predicted_class": pred_class,
        "confidence": confidence
    }

@torch.no_grad()
def plot_many_reconstructions(model, val_loader, device,num_samples=5):
    model.eval()
    x, y = next(iter(val_loader))
    x = x.to(device).float()
    
    x_hat, _, _, _ = model(x)

    plt.figure(figsize=(14, 3 * num_samples))

    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(x[i].cpu().numpy(), label="Original", linewidth=1.5)
        plt.plot(x_hat[i].cpu().numpy(), label="Reconstructed", linewidth=1.2)
        plt.title(f"Reconstruction #{i}")
        plt.legend()

    plt.tight_layout()
    plt.show()