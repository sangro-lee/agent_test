import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# ===== config =====
guidance_ws = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0]
# ==================

z_train = np.load("../../outputs/runs/rgcn_mlp_z4_sub_1/latents_train.npy")
y_train = np.load("../../outputs/runs/rgcn_mlp_z4_sub_1/y_train.npy").squeeze()

dims = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

for guidance_w in guidance_ws:

    run_tag = f"{guidance_w:g}"

    print(f"Processing guidance w={run_tag}")

    z_samples = np.load(
        f"../../outputs/runs/mlp_sub_1_0519_nf/diffusion/2026-05-19/"
        f"T1000_ep3000/cfg_w{guidance_w}_ddim/z_samples.npy"
    )

    kde = KernelDensity(
        kernel="gaussian",
        bandwidth="scott"
    ).fit(z_samples)

    density = kde.score_samples(z_samples)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    sc_tr_last = sc_diff_last = None

    for ax, (i, j) in zip(axes.flat, dims):

        sc_tr = ax.scatter(
            z_train[:, i],
            z_train[:, j],
            c=y_train,
            cmap="RdYlGn",
            vmin=y_train.min(),
            vmax=y_train.max(),
            s=3,
            alpha=0.6
        )

        sc_diff = ax.scatter(
            z_samples[:, i],
            z_samples[:, j],
            c=density,
            cmap="viridis",
            s=10,
            alpha=0.8
        )

        ax.set_xlabel(f"z{i}")
        ax.set_ylabel(f"z{j}")

        sc_tr_last = sc_tr
        sc_diff_last = sc_diff

    fig.colorbar(
        sc_tr_last,
        ax=axes[:, 2],
        label="pIC50 (train)",
        shrink=0.8
    )

    fig.colorbar(
        sc_diff_last,
        ax=axes[:, 0],
        label="KDE log-density",
        shrink=0.8
    )

    plt.suptitle(f"vqc z4 — guidance w={run_tag}", fontsize=12)

    save_path = f"./fix_mlp/kde_pairplot_mlp_0519_nf_{run_tag}_activity.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
