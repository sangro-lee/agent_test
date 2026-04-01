from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_style("whitegrid")


def plot_scatter(y_true, y_pred, title, save_path):
    from scipy.stats import pearsonr

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r, _ = pearsonr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("True pIC50")
    ax.set_ylabel("Pred pIC50")
    ax.set_title(f"{title}  (Pearson r = {r:.3f})")

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


def plot_latent_umap(latents, labels, save_path, split_name="", n_neighbors: int = 15):
    try:
        import umap as umap_lib
    except ImportError:
        return  # silently skip if umap-learn not installed

    latents = np.asarray(latents, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if latents.ndim != 2 or latents.shape[0] == 0:
        raise ValueError("latents must be a non-empty 2D array")

    n_neighbors = min(n_neighbors, latents.shape[0] - 1)
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(latents)

    title = f"Latent UMAP{' — ' + split_name if split_name else ''}"
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="viridis", s=12, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("pIC50")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_latent_tsne(latents, labels, save_path, split_name=""):
    from sklearn.manifold import TSNE

    latents = np.asarray(latents, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if latents.ndim != 2 or latents.shape[0] == 0:
        raise ValueError("latents must be a non-empty 2D array")

    perplexity = min(30, latents.shape[0] - 1)
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(latents)

    title = f"Latent t-SNE{' — ' + split_name if split_name else ''}"
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="viridis", s=12, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("pIC50")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
