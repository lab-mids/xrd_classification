import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def encode_xrd_csv(csv_path, model, cs, device, plot=False):
    """
    Universal XRD encoding function that handles:
      - models returning a tensor
      - models returning a tuple/list (e.g., (x_hat, mu, logvar, z_cls))
    """
    # -------------------------------
    # 1. Load CSV
    # -------------------------------
    df = pd.read_csv(csv_path)

    if "2theta" not in df.columns or "intensity" not in df.columns:
        raise ValueError("CSV MUST contain columns: 2theta, intensity")

    theta = df["2theta"].values
    inten = df["intensity"].values

    mask = (theta >= 10) & (theta <= 90)
    theta = theta[mask]
    inten = inten[mask]

    if theta.size == 0:
        raise ValueError("No valid XRD points inside 10–90° range.")

    # -------------------------------
    # 2. Interpolate
    # -------------------------------
    target_len = cs["input_len"]
    theta_grid = np.linspace(10, 90, target_len)
    inten_interp = np.interp(theta_grid, theta, inten)

    # Normalize 0–1
    inten_interp -= inten_interp.min()
    if inten_interp.max() > 0:
        inten_interp /= inten_interp.max()

    # -------------------------------
    # 3. Convert to model input shape
    # -------------------------------
    x = torch.tensor(inten_interp, dtype=torch.float32)

    # ALWAYS ensure shape (1, 1, L)
    if x.ndim == 1:
        x = x.unsqueeze(0)   # (1, L)
    if x.ndim == 2:
        x = x.unsqueeze(1)   # (1, 1, L)

    x = x.to(device)

    # -------------------------------
    # 4. Forward pass (supports all model types)
    # -------------------------------
    model.eval()
    with torch.no_grad():
        out = model(x)

        # Case 1: output is a tensor
        if isinstance(out, torch.Tensor):
            x_hat = out

        # Case 2: output is a tuple/list → FIRST item is always reconstruction
        elif isinstance(out, (tuple, list)):
            x_hat = out[0]

        else:
            raise RuntimeError(f"Unexpected model output type: {type(out)}")

        # Extract latent encodings
        z_rec, z_cls = model.encode(x)

    # -------------------------------
    # 5. Convert to numpy
    # -------------------------------
    x_np     = x.squeeze().cpu().numpy()
    x_hat_np = x_hat.squeeze().cpu().numpy()
    z_rec_np = z_rec.squeeze().cpu().numpy()
    z_cls_np = z_cls.squeeze().cpu().numpy()

    # -------------------------------
    # 6. Reconstruction error
    # -------------------------------
    mse  = float(np.mean((x_np - x_hat_np)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(x_np - x_hat_np)))

    # -------------------------------
    # 7. Plot
    # -------------------------------
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(theta_grid, x_np, label="Input")
        plt.plot(theta_grid, x_hat_np, label="Reconstruction")
        plt.title(f"Reconstruction (RMSE={rmse:.4f})")
        plt.xlabel("2θ (degrees)")
        plt.ylabel("Normalized Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------
    # 8. Return results
    # -------------------------------
    return {
        "theta_axis": theta_grid,
        "x_input": x_np,
        "x_recon": x_hat_np,
        "z_rec": z_rec_np,
        "z_cls": z_cls_np,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "tensor_input": x,
    }

def classify_xrd_csv(
    csv_path,
    model,
    cs,
    device,
    return_latent=True
):
    # -------------------------------
    # 1. Load CSV
    # -------------------------------
    df = pd.read_csv(csv_path)

    if "2theta" not in df.columns or "intensity" not in df.columns:
        raise ValueError("CSV MUST contain columns: 2theta, intensity")

    theta = df["2theta"].values
    inten = df["intensity"].values

    mask = (theta >= 10) & (theta <= 90)
    theta = theta[mask]
    inten = inten[mask]

    if theta.size == 0:
        raise ValueError("No valid XRD points inside 10–90° range.")

    # -------------------------------
    # 2. Interpolate
    # -------------------------------
    target_len = cs["input_len"]
    theta_grid = np.linspace(10, 90, target_len)
    inten_interp = np.interp(theta_grid, theta, inten)

    inten_interp -= inten_interp.min()
    if inten_interp.max() > 0:
        inten_interp /= inten_interp.max()

    # -------------------------------
    # 3. Tensor
    # -------------------------------
    x = torch.tensor(inten_interp, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    # -------------------------------
    # 4. Forward
    # -------------------------------
    model.eval()
    with torch.no_grad():
        x_hat, _, logits, _ = model(x)
        z_rec, z_cls = model.encode(x)

        probs = None
        pred_class_idx = None
        pred_class_name = None
        prob_by_class = None

        if logits is not None:
            probs = F.softmax(logits, dim=1)
            pred_class_idx = probs.argmax(dim=1).item()
            class_names = cs["class_names"]
            pred_class_name = class_names[pred_class_idx]
            prob_by_class = {
                class_names[i]: float(probs[0, i])
                for i in range(len(class_names))
            }

    # -------------------------------
    # 5. Return
    # -------------------------------
    result = {
        "theta_axis": theta_grid,
        "z_cls": z_cls.squeeze().cpu().numpy(),
        "z_rec": z_rec.squeeze().cpu().numpy(),
    }

    if logits is not None:
        result.update({
            "predicted_class_index": pred_class_idx,
            "predicted_class_name": pred_class_name,
            "probabilities": probs.squeeze().cpu().numpy(),
            "probability_by_class": prob_by_class,
            "logits": logits.squeeze().cpu().numpy(),
        })

    return result