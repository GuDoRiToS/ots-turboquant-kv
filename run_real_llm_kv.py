import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import your functions directly from Step 1
from run_tni_ablation import (
    compute_lloyd_max_beta,
    evaluate_metrics,
    paired_cliffs_delta,
    run_method,
)

# 15 long-context technical passages to generate real LLM KV cache states
PASSAGES = [
    "Large language models require substantial high-bandwidth memory during long-context autoregressive generation because the key-value cache grows linearly with sequence length. Compressing these tensors to 2-bit representations enables edge deployment.",
    "In transformer attention layers, attention sink tokens at the beginning of a sequence exhibit very small L2 norms near 0.05 while still receiving massive softmax attention weights, creating severe token norm imbalance across groups.",
    "Vector quantization methods apply random orthogonal rotations so that high-dimensional coordinates converge toward a known Beta distribution, allowing optimal Lloyd-Max scalar codebooks without needing offline calibration datasets.",
    "Quantized Johnson-Lindenstrauss projections preserve unbiased inner products by projecting quantization residuals onto random Rademacher sign vectors and storing only a single sign bit per coordinate dimension.",
    "Fast Walsh-Hadamard transforms reduce orthogonal rotation complexity from quadratic matrix multiplication to log-linear time while spreading sparse channel-wise outliers uniformly across all feature dimensions.",
    "Neurosymbolic vulnerability detection combines static data-flow analysis queries with large language model reasoning to trace taint propagation from untrusted entry points to security-critical sink operations.",
    "Autonomous coding agents operating in stateful workspaces require execution-time authorization gates at the action boundary to prevent intermediate irreversible system modifications during multi-step debugging.",
    "Edge computing devices in industrial Internet of Things environments rely on model pruning and low-bit post-training quantization to execute real-time fault diagnostics within strict memory and thermal envelopes.",
    "Empirical software engineering benchmarks require non-parametric statistical validation using two-sided Wilcoxon signed-rank tests and paired Cliff delta effect sizes to confirm performance gains across random seeds.",
    "System-aware key-value cache optimization coordinates temporal scheduling, spatial memory hierarchy offloading between GPU HBM and host CPU DRAM, and structural low-bit compression without retraining model weights.",
    "Multi-modal and omni-modal foundation models process interleaved visual, audio, and text tokens that exhibit wide inter-modality norm disparities, causing standard per-channel group quantizers to lose precision.",
    "Interactive explanation interfaces that represent language model reasoning steps as navigable dependency graphs enable human reviewers to localize logical errors faster and avoid overreliance on flawed outputs.",
    "Disaggregated inference architectures decouple compute-bound prefill phases from memory-bound decoding phases across specialized accelerator nodes to maximize serving throughput under service-level objectives.",
    "Adversarial robustness evaluations of automated language model judges reveal that minor prompt template variations and high-perplexity token perturbations can shift measured attack success rates significantly.",
    "Selective benchmark construction filters evaluation datasets using capability alignment deviation and discriminability scores to retain a compact subset of test items that preserve full leaderboard ranking consistency."
]

def extract_real_kv_samples(model_name: str = "Qwen/Qwen2.5-0.5B", seq_len: int = 32):
    print(f"Loading real AI model: {model_name} (runs on CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    samples = []
    # Hook to capture real Query (Q) and Key (K) states from the middle transformer layer
    target_layer = model.model.layers[len(model.model.layers) // 2].self_attn
    captured = {}

    def q_hook(module, inp, out):
        captured["Q"] = out.detach().cpu().numpy()

    def k_hook(module, inp, out):
        captured["K"] = out.detach().cpu().numpy()

    h_q = target_layer.q_proj.register_forward_hook(q_hook)
    h_k = target_layer.k_proj.register_forward_hook(k_hook)

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    with torch.no_grad():
        for text in PASSAGES:
            # Repeat text slightly to guarantee >= seq_len tokens
            inputs = tokenizer(text * 2, return_tensors="pt", truncation=True, max_length=seq_len)
            if inputs["input_ids"].shape[1] < seq_len:
                continue
            model(**inputs)

            # Reshape K to (num_kv_heads * seq_len, head_dim) so we have a batch of real KV vectors
            K_raw = captured["K"][0, :seq_len, :].reshape(seq_len, num_kv_heads, head_dim)
            K_mat = np.transpose(K_raw, (1, 0, 2)).reshape(-1, head_dim).astype(np.float32)

            K_mat = np.transpose(K_raw, (1, 0, 2)).reshape(-1, head_dim).astype(np.float32)
            K_mat = K_mat - np.mean(K_mat, axis=0, keepdims=True)  # <-- ADD THIS 1 LINE (Channel-Mean Centering)

            # Take matching Query vectors (32, head_dim)
            Q_raw = captured["Q"][0, :seq_len, :head_dim].reshape(seq_len, head_dim).astype(np.float32)
            samples.append((K_mat, Q_raw))

    h_q.remove()
    h_k.remove()
    return samples, head_dim

if __name__ == "__main__":
    samples, d = extract_real_kv_samples()
    n_samples = len(samples)
    n_tokens = samples[0][0].shape[0]

    # Measure real Token Norm Imbalance in Qwen2.5-0.5B
    all_norms = np.concatenate([np.linalg.norm(K, axis=-1) for K, _ in samples])
    print(f"Extracted {n_samples} real KV cache batches (N={n_tokens} vectors/batch, head_dim d={d})")
    print(f"Real LLM Token L2 Norms -> Min: {all_norms.min():.3f}, Median: {np.median(all_norms):.3f}, Max: {all_norms.max():.3f}")

    methods = [
        "1_KIVI_Group",
        "2_TurboQuant_MSE_Group",
        "3_TurboQuant_Prod_Group",
        "4_Ablation_OTS_NoRot",
        "5_Proposed_OTS_TQ_Dense",
        "6_Proposed_OTS_TQ_Fast",
    ]

    for bits in [2, 3]:
        thresholds, centroids = compute_lloyd_max_beta(d, bits)
        results = {m: {"mse": [], "cos": [], "ip": [], "kl": [], "ms": []} for m in methods}

        for idx, (K, Q) in enumerate(samples):
            rng = np.random.default_rng(2000 + idx)
            G_mat = rng.standard_normal((d, d)).astype(np.float32)
            Pi_dense, R_qr = np.linalg.qr(G_mat)
            Pi_dense *= np.sign(np.diag(R_qr))

            hadamard_signs = rng.choice([-1.0, 1.0], size=(1, d)).astype(np.float32)
            qjl_R = rng.choice([-1.0, 1.0], size=(d, d)).astype(np.float32)

            for m in methods:
                K_hat, IP_est, ms = run_method(
                    K, Q, m, bits, thresholds, centroids, Pi_dense, hadamard_signs, qjl_R, group_size=32
                )
                rel_mse, cos_sim, ip_corr, kl_div = evaluate_metrics(K, Q, K_hat, IP_est)
                results[m]["mse"].append(rel_mse)
                results[m]["cos"].append(cos_sim)
                results[m]["ip"].append(ip_corr)
                results[m]["kl"].append(kl_div)
                results[m]["ms"].append(ms)

        print(f"\n======== REAL LLM ({bits}-BIT) KV CACHE: Qwen2.5-0.5B ({n_samples} Passages, d={d}) ========")
        print(f"{'Method':<26} | {'Rel MSE':<10} | {'Cos Sim':<10} | {'IP Corr':<10} | {'Attn KL':<10} | {'Time(ms)':<8}")
        print("-" * 88)
        for m in methods:
            print(
                f"{m:<26} | {np.mean(results[m]['mse']):<10.5f} | "
                f"{np.mean(results[m]['cos']):<10.4f} | {np.mean(results[m]['ip']):<10.4f} | "
                f"{np.mean(results[m]['kl']):<10.5f} | {np.mean(results[m]['ms']):<8.2f}"
            )

        fast_mse = np.array(results["6_Proposed_OTS_TQ_Fast"]["mse"])
        base_mse = np.array(results["3_TurboQuant_Prod_Group"]["mse"])
        _, p_mse = stats.wilcoxon(fast_mse, base_mse)
        delta_mse = paired_cliffs_delta(base_mse, fast_mse)

        fast_kl = np.array(results["6_Proposed_OTS_TQ_Fast"]["kl"])
        base_kl = np.array(results["3_TurboQuant_Prod_Group"]["kl"])
        _, p_kl = stats.wilcoxon(fast_kl, base_kl)
        delta_kl = paired_cliffs_delta(base_kl, fast_kl)
        print(f"Wilcoxon (OTS-TQ-Fast vs Vanilla TQ-Prod): MSE p={p_mse:.2e} (delta={delta_mse:.3f}), KL p={p_kl:.2e} (delta={delta_kl:.3f})")