from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_style("whitegrid")


def plot_scatter(y_true, y_pred, title, save_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    ax.set_title(title)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_loss_curve(history, save_path):
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(train_loss) > 0:
        ax.plot(epochs, train_loss, label="train")
    if len(val_loss) > 0:
        ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curve")
    ax.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_latent_umap(latents, labels, save_path):
    latents = np.asarray(latents, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if latents.ndim != 2 or latents.shape[0] == 0:
        raise ValueError("latents must be a non-empty 2D array")

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(latents)
        title = "Latent UMAP"
    except Exception:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        emb = reducer.fit_transform(latents)
        title = "Latent PCA (UMAP fallback)"

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="viridis", s=12, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Label")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
