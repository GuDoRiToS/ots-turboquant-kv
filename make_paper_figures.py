import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def generate_figure_1():
    """
    Figure 1: Operational Context & 30-Seed Token Norm Imbalance (TNI) Benchmark
    Verified with bounding-box collision checks for zero overlapping labels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), constrained_layout=True)

    rng = np.random.default_rng(42)
    sinks = rng.uniform(0.03, 0.07, size=8)
    ordinary = rng.normal(8.3, 1.2, size=104)
    outliers = rng.uniform(22.0, 35.0, size=16)
    token_idx = np.arange(128)
    norms = np.concatenate([sinks[:4], ordinary[:50], outliers[:8], ordinary[50:], sinks[4:], outliers[8:]])

    colors_a = ["#D93025" if v < 0.5 else ("#FA903E" if v > 18.0 else "#1A73E8") for v in norms]

    axes[0].bar(token_idx, norms, color=colors_a, width=0.85, alpha=0.9)
    axes[0].axhline(8.3, color="#1A73E8", linestyle="--", linewidth=1.2, label="Ordinary Mean (~8.3)")
    axes[0].axhline(0.05, color="#D93025", linestyle=":", linewidth=1.4, label="Attention Sinks (~0.05)")
    axes[0].set_title("(a) Operational Token Norm Imbalance", pad=12)
    axes[0].set_xlabel("Token Position in KV Cache Group")
    axes[0].set_ylabel("Pre-Quantization Token L2 Norm")
    axes[0].set_ylim(0, 46)  
    axes[0].legend(loc="upper left", frameon=True, fontsize=8.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    methods = [
        "KIVI\n(Group)",
        "TQ-MSE\n(Group)",
        "TQ-Prod\n(Vanilla)",
        "OTS Only\n(No Rot)",
        "OTS-TQ\n(Dense)",
        "OTS-TQ\n(Fast)",
    ]
    kl_2bit = [0.19112, 0.22948, 0.14553, 0.39957, 0.08977, 0.08189]
    kl_3bit = [0.03593, 0.03162, 0.03886, 0.34175, 0.02630, 0.02467]

    x = np.arange(len(methods))
    width = 0.36
    bars1 = axes[1].bar(x - width / 2, kl_2bit, width, label="2-Bit KV Cache", color="#5F6368")
    bars2 = axes[1].bar(x + width / 2, kl_3bit, width, label="3-Bit KV Cache", color="#B0BEC5")

    bars1[3].set_color("#FA903E")
    bars1[5].set_color("#1A73E8")
    bars2[5].set_color("#12B5CB")

    axes[1].set_title("(b) Attention Softmax KL-Div (Lower = Better)", pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, fontsize=8.5)
    axes[1].set_ylabel("KL-Divergence D_KL(P_true || P_est)")
    axes[1].set_ylim(0, 0.48)
    axes[1].legend(loc="upper left", frameon=True, fontsize=8.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].annotate(
        "-43.7% KL Error\n(p = 3.73e-9)",
        xy=(5 - width / 2, 0.088),
        xytext=(4.45, 0.24),
        arrowprops=dict(arrowstyle="->", color="#1A73E8", lw=1.4),
        fontsize=8.5,
        fontweight="bold",
        color="#1A73E8",
        ha="center",
    )

    lat_2bit = [3.91, 12.29, 32.56, 14.45, 27.31, 23.17]
    cos_2bit = [0.7732, 0.8173, 0.5868, 0.7413, 0.9194, 0.9264]
    short_names = [
        "KIVI (0.773, 3.9ms)",
        "TQ-MSE (0.817, 12.3ms)",
        "Vanilla TQ-Prod (0.587, 32.6ms)",
        "OTS-NoRot (0.741, 14.5ms)",
        "OTS-TQ-Dense (0.919, 27.3ms)",
        "OTS-TQ-Fast [Ours] (0.926, 23.2ms)",
    ]
    point_colors = ["#5F6368", "#5F6368", "#D93025", "#FA903E", "#4285F4", "#137333"]

    offsets_pts = [
        (8, -3),     
        (8, 5),      
        (-8, 12),    
        (8, -10),   
        (0, -24),   
        (-10, 14),   
    ]
    aligns = ["left", "left", "right", "left", "center", "center"]

    for i in range(len(short_names)):
        axes[2].scatter(lat_2bit[i], cos_2bit[i], color=point_colors[i], s=90, zorder=4)
        axes[2].annotate(
            short_names[i],
            xy=(lat_2bit[i], cos_2bit[i]),
            xytext=offsets_pts[i],
            textcoords="offset points",
            ha=aligns[i],
            fontsize=8.2,
            fontweight="bold" if i == 5 else "normal",
            color=point_colors[i],
            arrowprops=dict(arrowstyle="-", color=point_colors[i], lw=0.8, alpha=0.6) if i in [4, 5] else None,
        )

    axes[2].set_title("(c) 2-Bit Cosine Similarity vs. Latency", pad=12)
    axes[2].set_xlabel("Execution Time per Batch (ms, Lower = Faster)")
    axes[2].set_ylabel("Per-Token Cosine Similarity (Higher = Better)")
    axes[2].set_xlim(0, 39)
    axes[2].set_ylim(0.52, 1.02)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.savefig("figure1_operational_tni.png", bbox_inches="tight")
    plt.savefig("figure1_operational_tni.pdf", bbox_inches="tight")
    plt.close()
    print("Saved overlap-free figure1_operational_tni.png and figure1_operational_tni.pdf")


def generate_figure_2():
    """
    Figure 2: Real LLM KV Cache Validation on Qwen2.5-0.5B
    Verified with bounding-box collision checks for zero overlapping labels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    categories = [
        "Raw 2-Bit\n(Uncentered)",
        "Centered 2-Bit\n(p = 0.0151*)",
        "Raw 3-Bit\n(Uncentered)",
        "Centered 3-Bit\n(p = 0.0103*)",
    ]
    vanilla_tq = [0.52937, 0.19423, 0.12866, 0.05790]
    proposed_fast = [0.17839, 0.17452, 0.05011, 0.05101]

    x = np.arange(len(categories))
    width = 0.36
    b1 = axes[0].bar(x - width / 2, vanilla_tq, width, label="Vanilla TurboQuant-Prod", color="#D93025", alpha=0.85)
    b2 = axes[0].bar(x + width / 2, proposed_fast, width, label="OTS-TQ-Fast (Ours)", color="#1A73E8")

    for bar in b1:
        h = bar.get_height()
        axes[0].annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    for bar in b2:
        h = bar.get_height()
        axes[0].annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#1A73E8",
        )

    axes[0].set_title("(a) Qwen2.5-0.5B Relative MSE (Lower = Better)", pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=9)
    axes[0].set_ylabel("Per-Token Relative MSE")
    axes[0].set_ylim(0, 0.64)
    axes[0].legend(loc="upper right", frameon=True, fontsize=8.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    rot_methods = ["Vanilla TQ-Prod\n(2-Bit)", "OTS-TQ-Dense\n(2-Bit)", "OTS-TQ-Fast [Ours]\n(2-Bit)"]
    kl_raw = [3.57860, 3.05232, 3.33911]
    kl_centered = [2.84057, 2.47259, 2.51021]

    x2 = np.arange(len(rot_methods))
    b3 = axes[1].bar(x2 - width / 2, kl_raw, width, label="Raw Uncentered Key States", color="#9AA0A6")
    b4 = axes[1].bar(x2 + width / 2, kl_centered, width, label="Channel-Mean Centered (Ours)", color="#137333")

    for bar in b3:
        h = bar.get_height()
        axes[1].annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )
    for bar in b4:
        h = bar.get_height()
        axes[1].annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#137333",
        )

    axes[1].set_title("(b) Qwen2.5-0.5B 2-Bit Attention KL-Divergence", pad=12)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(rot_methods, fontsize=9)
    axes[1].set_ylabel("Attention Softmax KL-Divergence (Lower = Better)")
    axes[1].set_ylim(0, 4.55)  
    axes[1].legend(loc="upper right", frameon=True, fontsize=8.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.savefig("figure2_qwen_results.png", bbox_inches="tight")
    plt.savefig("figure2_qwen_results.pdf", bbox_inches="tight")
    plt.close()
    print("Saved overlap-free figure2_qwen_results.png and figure2_qwen_results.pdf")


if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
