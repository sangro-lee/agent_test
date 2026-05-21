import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

z_train   = np.load("../../outputs/runs/rgcn_vqc_z4_sub_1/latents_train.npy")
z_samples = np.load("../../outputs/runs/rgcn_vqc_z4_sub_1/diffusion/2026-04-24/T1000_ep3000/cfg_w7.0_ddim/z_samples.npy")

kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(z_samples)
density = kde.score_samples(z_samples)

dims = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (i, j) in zip(axes.flat, dims):
    ax.scatter(z_train[:,i],   z_train[:,j],   c="lightgray", s=3, alpha=0.4, label="train")
    sc = ax.scatter(z_samples[:,i], z_samples[:,j], c=density, cmap="viridis", s=8, label="diffusion")
    ax.set_xlabel(f"z{i}"); ax.set_ylabel(f"z{j}")
    plt.colorbar(sc, ax=ax)

axes.flat[0].legend()
plt.tight_layout()
plt.savefig("kde_pairplot_vqc_7.png", dpi=300)

