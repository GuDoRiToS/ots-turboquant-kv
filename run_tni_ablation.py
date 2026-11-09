import time
import numpy as np
from scipy import stats

def compute_lloyd_max_beta(d: int, bits: int, max_iter: int = 100, tol: float = 1e-7):
    """
    Computes exact Lloyd-Max scalar quantizer centroids and thresholds for the 
    marginal distribution of a single coordinate of a random unit vector in R^d.
    """
    n_levels = 2 ** bits
    a = (d - 1) / 2.0
    dist = stats.beta(a, a, loc=-1.0, scale=2.0)
    
    probs = np.linspace(0, 1, n_levels + 1)
    thresholds = dist.ppf(probs)
    thresholds[0] = -1.0
    thresholds[-1] = 1.0
    centroids = 0.5 * (thresholds[:-1] + thresholds[1:])
    
    grid = np.linspace(-0.9999, 0.9999, 20001)
    pdf = dist.pdf(grid)
    
    for _ in range(max_iter):
        old_centroids = centroids.copy()
        thresholds[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        for k in range(n_levels):
            mask = (grid >= thresholds[k]) & (grid < thresholds[k+1])
            if np.any(mask):
                w = pdf[mask]
                w_sum = np.sum(w)
                if w_sum > 0:
                    centroids[k] = np.sum(grid[mask] * w) / w_sum
        if np.max(np.abs(centroids - old_centroids)) < tol:
            break
    return thresholds.astype(np.float32), centroids.astype(np.float32)

def fwht(X: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform along the last axis in O(d log d)."""
    Y = X.copy()
    d = Y.shape[-1]
    h = 1
    while h < d:
        Y = Y.reshape(-1, d // (2 * h), 2, h)
        a = Y[:, :, 0, :].copy()
        b = Y[:, :, 1, :].copy()
        Y[:, :, 0, :] = a + b
        Y[:, :, 1, :] = a - b
        Y = Y.reshape(-1, d)
        h *= 2
    return Y / np.sqrt(d)

def generate_tni_kv_cache(n_tokens: int = 128, d: int = 256, group_size: int = 32, seed: int = 42):
    """
    Generates a realistic KV Cache Key matrix K (n_tokens, d) and Query matrix Q (32, d)
    exhibiting:
      1. Token Norm Imbalance (TNI):
         - Attention Sink tokens: L2 norm ~ U(0.03, 0.07) (mean ~0.05)
         - Ordinary tokens:       L2 norm ~ N(8.3, 1.2)
         - Modality/Spike tokens: L2 norm ~ U(22.0, 35.0)
      2. Channel-wise Outliers (sparse outlier channels typical of LLM Key states)
    """
    rng = np.random.default_rng(seed)
    
    raw_K = rng.standard_normal((n_tokens, d))
    outlier_channels = rng.choice(d, size=max(1, d // 25), replace=False)
    raw_K[:, outlier_channels] *= 12.0  
    
    unit_K = raw_K / (np.linalg.norm(raw_K, axis=-1, keepdims=True) + 1e-12)
    
    norms = np.zeros((n_tokens, 1), dtype=np.float32)
    for i in range(n_tokens):
        r = rng.random()
        if i < 4 or r < 0.05:  
            norms[i, 0] = rng.uniform(0.03, 0.07)
        elif r < 0.90:         
            norms[i, 0] = max(1.0, rng.normal(8.3, 1.2))
        else:                 
            norms[i, 0] = rng.uniform(22.0, 35.0)
            
    K = unit_K * norms
    Q = rng.standard_normal((32, d))
    Q[:, outlier_channels] *= 4.0
    Q = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-12) * np.sqrt(d)
    return K.astype(np.float32), Q.astype(np.float32), norms

def quantize_with_codebook(Y: np.ndarray, thresholds: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(thresholds[1:-1], Y)
    return centroids[idx]

def run_method(
    K: np.ndarray,
    Q: np.ndarray,
    method_name: str,
    bits: int,
    thresholds: np.ndarray,
    centroids: np.ndarray,
    Pi_dense: np.ndarray,
    hadamard_signs: np.ndarray,
    qjl_R: np.ndarray,
    group_size: int = 32,
):
    t0 = time.perf_counter()
    N, d = K.shape
    num_groups = N // group_size
    
    if method_name == "1_KIVI_Group":
        K_reshaped = K.reshape(num_groups, group_size, d)
        c_min = np.min(K_reshaped, axis=1, keepdims=True)
        c_max = np.max(K_reshaped, axis=1, keepdims=True)
        levels = (2 ** bits) - 1
        scale = (c_max - c_min) / (levels + 1e-12)
        q_idx = np.round((K_reshaped - c_min) / (scale + 1e-12)).clip(0, levels)
        K_hat = (q_idx * scale + c_min).reshape(N, d)
        IP_est = Q @ K_hat.T

    elif method_name in ["2_TurboQuant_MSE_Group", "3_TurboQuant_Prod_Group"]:
        K_groups = K.reshape(num_groups, group_size, d)
        token_l2 = np.linalg.norm(K_groups, axis=-1, keepdims=True)
        group_max_norm = np.max(token_l2, axis=1, keepdims=True) + 1e-12
        K_scaled = (K_groups / group_max_norm).reshape(N, d)
        
        Y = K_scaled @ Pi_dense.T
        Y_hat = quantize_with_codebook(Y, thresholds, centroids)
        
        if method_name == "2_TurboQuant_MSE_Group":
            K_hat = (Y_hat @ Pi_dense).reshape(num_groups, group_size, d) * group_max_norm
            K_hat = K_hat.reshape(N, d)
            IP_est = Q @ K_hat.T
        else:
            res = Y - Y_hat
            res_norms = np.linalg.norm(res, axis=-1, keepdims=True)
            signs = np.where((res @ qjl_R.T) >= 0, 1.0, -1.0)
            
            Q_rot = Q @ Pi_dense.T
            Q_proj = Q_rot @ qjl_R.T
            qjl_corr = np.sqrt(np.pi / 2.0) / d * (Q_proj @ (signs * res_norms).T)
            base_ip = Q_rot @ Y_hat.T
            
            group_scales_flat = np.repeat(group_max_norm.squeeze(), group_size)
            IP_est = (base_ip + qjl_corr) * group_scales_flat[None, :]
            
            res_recon = np.sqrt(np.pi / 2.0) / d * ((signs * res_norms) @ qjl_R)
            K_hat = ((Y_hat + res_recon) @ Pi_dense).reshape(num_groups, group_size, d) * group_max_norm
            K_hat = K_hat.reshape(N, d)

    elif method_name == "4_Ablation_OTS_NoRot":
        token_norms = np.linalg.norm(K, axis=-1, keepdims=True).astype(np.float16)
        K_unit = K / (token_norms.astype(np.float32) + 1e-12)
        
        Y_hat = quantize_with_codebook(K_unit, thresholds, centroids)
        res = K_unit - Y_hat
        res_norms = np.linalg.norm(res, axis=-1, keepdims=True)
        signs = np.where((res @ qjl_R.T) >= 0, 1.0, -1.0)
        
        Q_proj = Q @ qjl_R.T
        qjl_corr = np.sqrt(np.pi / 2.0) / d * (Q_proj @ (signs * res_norms).T)
        base_ip = Q @ Y_hat.T
        IP_est = (base_ip + qjl_corr) * token_norms.astype(np.float32).T
        
        res_recon = np.sqrt(np.pi / 2.0) / d * ((signs * res_norms) @ qjl_R)
        K_hat = (Y_hat + res_recon) * token_norms.astype(np.float32)

    elif method_name == "5_Proposed_OTS_TQ_Dense":
        token_norms = np.linalg.norm(K, axis=-1, keepdims=True).astype(np.float16)
        K_unit = K / (token_norms.astype(np.float32) + 1e-12)
        
        Y = K_unit @ Pi_dense.T
        Y_hat = quantize_with_codebook(Y, thresholds, centroids)
        
        res = Y - Y_hat
        res_norms = np.linalg.norm(res, axis=-1, keepdims=True)
        signs = np.where((res @ qjl_R.T) >= 0, 1.0, -1.0)
        
        Q_rot = Q @ Pi_dense.T
        Q_proj = Q_rot @ qjl_R.T
        qjl_corr = np.sqrt(np.pi / 2.0) / d * (Q_proj @ (signs * res_norms).T)
        base_ip = Q_rot @ Y_hat.T
        IP_est = (base_ip + qjl_corr) * token_norms.astype(np.float32).T
        
        res_recon = np.sqrt(np.pi / 2.0) / d * ((signs * res_norms) @ qjl_R)
        K_hat = ((Y_hat + res_recon) @ Pi_dense) * token_norms.astype(np.float32)

    elif method_name == "6_Proposed_OTS_TQ_Fast":
        token_norms = np.linalg.norm(K, axis=-1, keepdims=True).astype(np.float16)
        K_unit = K / (token_norms.astype(np.float32) + 1e-12)
        
        Y = fwht(K_unit * hadamard_signs)
        Y_hat = quantize_with_codebook(Y, thresholds, centroids)
        
        res = Y - Y_hat
        res_norms = np.linalg.norm(res, axis=-1, keepdims=True)
        signs = np.where((res @ qjl_R.T) >= 0, 1.0, -1.0)
        
        Q_rot = fwht(Q * hadamard_signs)
        Q_proj = Q_rot @ qjl_R.T
        qjl_corr = np.sqrt(np.pi / 2.0) / d * (Q_proj @ (signs * res_norms).T)
        base_ip = Q_rot @ Y_hat.T
        IP_est = (base_ip + qjl_corr) * token_norms.astype(np.float32).T
        
        res_recon = np.sqrt(np.pi / 2.0) / d * ((signs * res_norms) @ qjl_R)
        K_hat = (fwht(Y_hat + res_recon) * hadamard_signs) * token_norms.astype(np.float32)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return K_hat, IP_est, elapsed_ms

def evaluate_metrics(K: np.ndarray, Q: np.ndarray, K_hat: np.ndarray, IP_est: np.ndarray):
    rel_mse = np.mean(np.sum((K - K_hat) ** 2, axis=-1) / (np.sum(K ** 2, axis=-1) + 1e-12))
    
    cos_sim = np.mean(
        np.sum(K * K_hat, axis=-1) / (np.linalg.norm(K, axis=-1) * np.linalg.norm(K_hat, axis=-1) + 1e-12)
    )
    
    true_ip = Q @ K.T
    ip_corr = np.corrcoef(true_ip.ravel(), IP_est.ravel())[0, 1]
    
    scale = 1.0 / np.sqrt(K.shape[-1])
    logits_true = true_ip * scale
    logits_est = IP_est * scale
    
    logits_true -= np.max(logits_true, axis=-1, keepdims=True)
    logits_est -= np.max(logits_est, axis=-1, keepdims=True)
    
    P_true = np.exp(logits_true) / np.sum(np.exp(logits_true), axis=-1, keepdims=True)
    P_est = np.exp(logits_est) / np.sum(np.exp(logits_est), axis=-1, keepdims=True)
    
    kl_div = np.mean(np.sum(P_true * np.log((P_true + 1e-12) / (P_est + 1e-12)), axis=-1))
    return rel_mse, cos_sim, ip_corr, kl_div

def paired_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    diff = x[:, None] - y[None, :]
    n = len(x) * len(y)
    return float(np.sum(diff > 0) - np.sum(diff < 0)) / n

if __name__ == "__main__":
    d = 256
    n_tokens = 128
    n_seeds = 30
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
        
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            G_mat = rng.standard_normal((d, d)).astype(np.float32)
            Pi_dense, R_qr = np.linalg.qr(G_mat)
            Pi_dense *= np.sign(np.diag(R_qr))
            
            hadamard_signs = rng.choice([-1.0, 1.0], size=(1, d)).astype(np.float32)
            qjl_R = rng.choice([-1.0, 1.0], size=(d, d)).astype(np.float32)
            
            K, Q, _ = generate_tni_kv_cache(n_tokens=n_tokens, d=d, group_size=32, seed=seed)
            
            for m in methods:
                K_hat, IP_est, ms = run_method(
                    K, Q, m, bits, thresholds, centroids, Pi_dense, hadamard_signs, qjl_R
                )
                rel_mse, cos_sim, ip_corr, kl_div = evaluate_metrics(K, Q, K_hat, IP_est)
                results[m]["mse"].append(rel_mse)
                results[m]["cos"].append(cos_sim)
                results[m]["ip"].append(ip_corr)
                results[m]["kl"].append(kl_div)
                results[m]["ms"].append(ms)
                
        print(f"\n================ {bits}-BIT KV CACHE QUANTIZATION (30 SEEDS, d={d}, N={n_tokens}) ================")
        print(f"{'Method':<26} | {'Rel MSE':<10} | {'Cos Sim':<10} | {'IP Corr':<10} | {'Attn KL':<10} | {'Time(ms)':<8}")
        print("-" * 88)
        for m in methods:
            mse_m = np.mean(results[m]["mse"])
            cos_m = np.mean(results[m]["cos"])
            ip_m = np.mean(results[m]["ip"])
            kl_m = np.mean(results[m]["kl"])
            ms_m = np.mean(results[m]["ms"])
            print(f"{m:<26} | {mse_m:<10.5f} | {cos_m:<10.4f} | {ip_m:<10.4f} | {kl_m:<10.5f} | {ms_m:<8.2f}")
            
        fast_kl = np.array(results["6_Proposed_OTS_TQ_Fast"]["kl"])
        base_kl = np.array(results["3_TurboQuant_Prod_Group"]["kl"])
        _, p_kl = stats.wilcoxon(fast_kl, base_kl)
        delta_kl = paired_cliffs_delta(base_kl, fast_kl)
        
        fast_cos = np.array(results["6_Proposed_OTS_TQ_Fast"]["cos"])
        base_cos = np.array(results["3_TurboQuant_Prod_Group"]["cos"])
        _, p_cos = stats.wilcoxon(fast_cos, base_cos)
        delta_cos = paired_cliffs_delta(fast_cos, base_cos)
        print(f"Wilcoxon (OTS-TQ-Fast vs Vanilla TQ-Prod): KL p={p_kl:.2e} (Cliff's delta={delta_kl:.3f}), CosSim p={p_cos:.2e} (Cliff's delta={delta_cos:.3f})")
