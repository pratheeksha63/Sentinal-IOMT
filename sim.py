import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import make_classification
import shap

def parse_args():
    parser = argparse.ArgumentParser(description="Sentinel-IoMT: Federated IDS")
    parser.add_argument("--dataset", type=str, default="ciciomt2024", choices=["ciciomt2024", "ciciot2023", "synthetic"])
    parser.add_argument("--max_rows", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--n_clients", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--aggregator", type=str, default="fedavg", choices=["fedavg", "median", "trimmed_mean", "krum", "xai_guided"])
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--xai_mode", type=str, default="shap", choices=["shap", "weight_dist", "both"])
    parser.add_argument("--attack", type=str, default="none", choices=["none", "label_flip", "scaled_update", "stealth"])
    parser.add_argument("--n_malicious", type=int, default=1)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--grid", action="store_true")
    return parser.parse_args()

def map_coarse_label(val):
    s = str(val).lower().strip()
    if "benign" in s:
        return "Benign"
    if "ddos" in s:
        return "DDoS"
    if "dos" in s:
        return "DoS"
    if any(k in s for k in ["recon", "scan", "sweep", "prob"]):
        return "Recon"
    if "mqtt" in s:
        return "MQTT"
    if "arp" in s or "spoof" in s:
        return "ARP_Spoofing"
    return None

def load_data(dataset_name, max_rows, seed=0):
    np.random.seed(seed)
    data_dir = os.path.join(".", "data", dataset_name)
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True) if os.path.exists(data_dir) else []
    
    if dataset_name != "synthetic" and len(csv_files) > 0:
        dfs = []
        rows_so_far = 0
        for f in csv_files:
            try:
                chunk = pd.read_csv(f, nrows=(max_rows - rows_so_far) if max_rows > 0 else None)
                dfs.append(chunk)
                rows_so_far += len(chunk)
                if max_rows > 0 and rows_so_far >= max_rows:
                    break
            except Exception:
                continue
        if len(dfs) > 0:
            df = pd.concat(dfs, ignore_index=True)
            label_col = None
            for col in df.columns:
                if col.lower() in ["label", "class", "attack", "attack_type"]:
                    label_col = col
                    break
            if label_col is None:
                label_col = df.columns[-1]
            
            df["mapped_label"] = df[label_col].apply(map_coarse_label)
            df = df[df["mapped_label"].notna()].copy()
            
            feat_cols = [c for c in df.columns if c not in [label_col, "mapped_label"]]
            for c in feat_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)
            
            if len(df) > (max_rows if max_rows > 0 else 0) and max_rows > 0:
                df = df.sample(n=max_rows, random_state=seed)
            
            class_names = sorted(df["mapped_label"].unique().tolist())
            class_to_id = {cls: i for i, cls in enumerate(class_names)}
            y = df["mapped_label"].map(class_to_id).values.astype(np.int64)
            X = df[feat_cols].values.astype(np.float32)
            feature_names = feat_cols
            input_dim = X.shape[1]
            n_classes = len(class_names)
            return X, y, feature_names, class_names, input_dim, n_classes

    # Synthetic fallback
    n_samples = max_rows if (0 < max_rows <= 10000) else 10000
    feature_names = [
        "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts",
        "fwd_pkt_len_max", "fwd_pkt_len_mean", "bwd_pkt_len_max", "bwd_pkt_len_mean", "flow_byts_s",
        "flow_pkts_s", "flow_iat_mean", "flow_iat_std", "fwd_iat_tot", "bwd_iat_tot",
        "fwd_pkts_s", "bwd_pkts_s", "pkt_len_mean", "pkt_len_std", "syn_flag_cnt"
    ]
    class_names = ["Benign", "DDoS", "DoS", "Recon", "MQTT", "ARP_Spoofing"]
    X, y = make_classification(
        n_samples=n_samples, n_features=len(feature_names), n_informative=16,
        n_redundant=2, n_classes=len(class_names), n_clusters_per_class=1, random_state=seed
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    return X, y, feature_names, class_names, len(feature_names), len(class_names)

def dirichlet_partition(y_train, n_clients, alpha, seed=0):
    np.random.seed(seed)
    n_classes = len(np.unique(y_train))
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(y_train == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        counts = (proportions * len(idx_c)).astype(int)
        counts[-1] = len(idx_c) - counts[:-1].sum()
        start = 0
        for i in range(n_clients):
            cnt = counts[i]
            client_indices[i].extend(idx_c[start : start + cnt])
            start += cnt
    for i in range(n_clients):
        np.random.shuffle(client_indices[i])
    return [np.array(idx, dtype=int) for idx in client_indices]

class IoMTMLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

def get_weights(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def set_weights(model, weights):
    model.load_state_dict(weights)

def flatten_weights(weights):
    return torch.cat([v.flatten() for v in weights.values()])

def compute_ece(confidences, predictions, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    per_class = f1_score(y_test, preds, average=None, zero_division=0).tolist()
    ece = compute_ece(confs, preds, y_test)
    
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    review_fractions = [0.0, 0.02, 0.05, 0.10, 0.20]
    review_f1s = {}
    sorted_idx = np.argsort(-entropy)
    n = len(y_test)
    for frac in review_fractions:
        k = int(frac * n)
        adj_preds = preds.copy()
        if k > 0:
            adj_preds[sorted_idx[:k]] = y_test[sorted_idx[:k]]
        review_f1s[f"f1_rev_{int(frac*100)}"] = float(f1_score(y_test, adj_preds, average="macro", zero_division=0))
    
    return acc, f1, per_class, ece, entropy, probs, preds, review_f1s

def compute_client_shap(model, X_local, feature_names, n_samples=100, bg_samples=50):
    model.eval()
    n = len(X_local)
    eval_idx = np.random.choice(n, min(n, n_samples), replace=False)
    bg_idx = np.random.choice(n, min(n, bg_samples), replace=False)
    
    eval_t = torch.tensor(X_local[eval_idx], dtype=torch.float32, requires_grad=True)
    bg_mean = torch.tensor(X_local[bg_idx], dtype=torch.float32).mean(dim=0, keepdim=True)
    
    out = model(eval_t)
    n_classes = out.shape[1]
    diff = (eval_t - bg_mean).detach()
    
    grads = []
    for c in range(n_classes):
        model.zero_grad()
        if eval_t.grad is not None:
            eval_t.grad.zero_()
        g = torch.autograd.grad(out[:, c].sum(), eval_t, retain_graph=True)[0]
        grads.append(g.detach())
        
    grads = torch.stack(grads, dim=-1)
    attrib = (diff.unsqueeze(-1) * grads).abs()
    mean_abs = attrib.mean(dim=(0, 2)).numpy()
    mean_abs_norm = mean_abs / (np.sum(mean_abs) + 1e-9)
    top3_idx = np.argsort(mean_abs_norm)[-3:][::-1]
    top3_feats = [feature_names[i] for i in top3_idx]
    return mean_abs_norm, top3_feats

def train_local(client_id, model, X_client, y_client, global_weights, epochs, lr, batch_size, mu=0.0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X_client, dtype=torch.float32), torch.tensor(y_client, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    g_weights = {k: v.to(next(model.parameters()).device) for k, v in global_weights.items()}
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            if mu > 0:
                prox = sum((p - g_weights[k]).pow(2).sum() for k, p in model.named_parameters())
                loss = loss + 0.5 * mu * prox
            loss.backward()
            optimizer.step()
    return get_weights(model)

def run_federated_experiment(args, X_train, y_train, X_test, y_test, feature_names, class_names, input_dim, n_classes, visual_mode=False):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    n_clients = args.n_clients
    client_indices = dirichlet_partition(y_train, n_clients, args.alpha, seed=args.seed)
    client_train_data = [(X_train[idx], y_train[idx]) for idx in client_indices]
    
    global_model = IoMTMLP(input_dim, n_classes)
    global_weights = get_weights(global_model)
    
    total_params = sum(p.numel() for p in global_model.parameters())
    bytes_per_round = 2 * total_params * 4 * n_clients
    total_bytes = bytes_per_round * args.rounds
    
    malicious_clients = list(range(n_clients - args.n_malicious, n_clients)) if args.attack != "none" else []
    
    dos_class_idx = class_names.index("DoS") if "DoS" in class_names else (1 if n_classes > 1 else 0)
    benign_class_idx = class_names.index("Benign") if "Benign" in class_names else 0
    
    round_accs = []
    rounds_to_90 = -1
    events = []
    
    active_attack_rounds = 0
    tpr_hits = 0
    fpr_hits = 0
    total_honest_evals = 0
    shap_dists_honest_all = []
    shap_dists_mal_all = []
    
    needs_shap = visual_mode or (args.aggregator == "xai_guided" and args.xai_mode in ["shap", "both"])
    
    for r in range(args.rounds):
        attack_active = (r >= 3) and (len(malicious_clients) > 0)
        if attack_active:
            active_attack_rounds += 1
        
        client_local_weights = []
        client_accs = []
        client_shap_profiles = []
        client_shap_top3 = []
        client_review_alerts = []
        
        for c in range(n_clients):
            Xc, yc = client_train_data[c]
            local_model = IoMTMLP(input_dim, n_classes)
            set_weights(local_model, global_weights)
            
            is_malicious = (c in malicious_clients) and attack_active
            
            if is_malicious and args.attack == "label_flip":
                yc_train = (yc + np.random.randint(1, n_classes)) % n_classes
            elif is_malicious and args.attack == "stealth":
                yc_train = yc.copy()
                yc_train[yc_train == dos_class_idx] = benign_class_idx
            else:
                yc_train = yc
                
            w_trained = train_local(c, local_model, Xc, yc_train, global_weights, args.local_epochs, args.lr, args.batch_size, args.mu)
            
            w_final = {}
            if is_malicious and args.attack == "scaled_update":
                for k in w_trained:
                    diff = w_trained[k] - global_weights[k]
                    w_final[k] = global_weights[k] + 10.0 * diff
            elif is_malicious and args.attack == "stealth":
                diff_norm = torch.norm(flatten_weights({k: w_trained[k] - global_weights[k] for k in w_trained}))
                bound = 1.0
                scale = min(1.0, bound / (float(diff_norm) + 1e-9))
                for k in w_trained:
                    diff = w_trained[k] - global_weights[k]
                    w_final[k] = global_weights[k] + scale * diff
            else:
                w_final = w_trained
                
            client_local_weights.append(w_final)
            
            if visual_mode:
                set_weights(local_model, w_final)
                c_acc, _, _, _, c_ent, _, _, _ = evaluate_model(local_model, Xc[:min(100, len(Xc))], yc[:min(100, len(yc))])
                client_accs.append(c_acc)
                max_ent_idx = int(np.argmax(c_ent))
                client_review_alerts.append({
                    "client": c,
                    "class": class_names[int(yc[max_ent_idx])],
                    "entropy": float(round(c_ent[max_ent_idx], 3))
                })
            
            if needs_shap:
                set_weights(local_model, w_final)
                s_prof, s_top3 = compute_client_shap(local_model, Xc, feature_names, n_samples=100, bg_samples=50)
                client_shap_profiles.append(s_prof)
                client_shap_top3.append(s_top3)
                
        # Anomaly scoring & Distances
        flagged_clients = []
        shap_dists = np.zeros(n_clients)
        scores = np.zeros(n_clients)
        
        if needs_shap:
            shap_profiles = np.array(client_shap_profiles)
            median_shap = np.median(shap_profiles, axis=0)
            for c in range(n_clients):
                p = shap_profiles[c]
                cos_sim = np.dot(p, median_shap) / ((np.linalg.norm(p) * np.linalg.norm(median_shap)) + 1e-9)
                shap_dists[c] = 1.0 - cos_sim
                
        if args.aggregator == "xai_guided":
            if args.xai_mode in ["weight_dist", "both"]:
                flat_diffs = np.array([flatten_weights({k: client_local_weights[c][k] - global_weights[k] for k in global_weights}).numpy() for c in range(n_clients)])
                med_diff = np.median(flat_diffs, axis=0)
                w_dists = np.array([1.0 - (np.dot(flat_diffs[c], med_diff) / ((np.linalg.norm(flat_diffs[c]) * np.linalg.norm(med_diff)) + 1e-9)) for c in range(n_clients)])
            
            if args.xai_mode == "shap":
                scores = shap_dists
            elif args.xai_mode == "weight_dist":
                scores = w_dists
            else:
                s1 = shap_dists / (np.max(shap_dists) + 1e-9)
                s2 = w_dists / (np.max(w_dists) + 1e-9)
                scores = np.maximum(s1, s2)
                
            med_score = np.median(scores)
            mad = np.median(np.abs(scores - med_score))
            robust_z = 0.6745 * (scores - med_score) / (mad + 1e-6)
            flagged_clients = np.where(robust_z > args.tau)[0].tolist()
            
            if attack_active:
                for m in malicious_clients:
                    if m in flagged_clients:
                        tpr_hits += 1
                for h in range(n_clients):
                    if h not in malicious_clients:
                        total_honest_evals += 1
                        if h in flagged_clients:
                            fpr_hits += 1

        for c in range(n_clients):
            if c in malicious_clients and attack_active:
                shap_dists_mal_all.append(shap_dists[c])
            else:
                shap_dists_honest_all.append(shap_dists[c])

        # Aggregation
        client_sizes = np.array([len(client_train_data[c][0]) for c in range(n_clients)])
        weights = client_sizes.astype(float)
        
        if args.aggregator == "fedavg":
            weights = weights / weights.sum()
            new_global_weights = {k: sum(weights[c] * client_local_weights[c][k] for c in range(n_clients)) for k in global_weights}
        elif args.aggregator == "median":
            new_global_weights = {}
            for k in global_weights:
                stacked = torch.stack([client_local_weights[c][k] for c in range(n_clients)], dim=0)
                new_global_weights[k] = torch.median(stacked, dim=0).values
        elif args.aggregator == "trimmed_mean":
            new_global_weights = {}
            for k in global_weights:
                stacked = torch.stack([client_local_weights[c][k] for c in range(n_clients)], dim=0)
                if n_clients >= 3:
                    sorted_t, _ = torch.sort(stacked, dim=0)
                    new_global_weights[k] = sorted_t[1:-1].mean(dim=0)
                else:
                    new_global_weights[k] = stacked.mean(dim=0)
        elif args.aggregator == "krum":
            if n_clients >= 5:
                f_count = 1
                krum_scores = []
                for i in range(n_clients):
                    dists = [torch.norm(flatten_weights(client_local_weights[i]) - flatten_weights(client_local_weights[j]))**2 for j in range(n_clients) if i != j]
                    dists.sort()
                    krum_scores.append(sum(dists[:n_clients - f_count - 2]))
                best_idx = int(np.argmin(krum_scores))
                new_global_weights = {k: client_local_weights[best_idx][k].clone() for k in global_weights}
            else:
                weights = weights / weights.sum()
                new_global_weights = {k: sum(weights[c] * client_local_weights[c][k] for c in range(n_clients)) for k in global_weights}
        elif args.aggregator == "xai_guided":
            for fc in flagged_clients:
                weights[fc] = 0.0
            weights = (weights / weights.sum()) if weights.sum() > 0 else (np.ones(n_clients) / n_clients)
            new_global_weights = {k: sum(weights[c] * client_local_weights[c][k] for c in range(n_clients)) for k in global_weights}
        else:
            weights = weights / weights.sum()
            new_global_weights = {k: sum(weights[c] * client_local_weights[c][k] for c in range(n_clients)) for k in global_weights}
            
        global_weights = new_global_weights
        set_weights(global_model, global_weights)
        
        g_acc, g_f1, _, _, _, _, _, _ = evaluate_model(global_model, X_test, y_test)
        round_accs.append(g_acc)
        if g_f1 >= 0.90 and rounds_to_90 == -1:
            rounds_to_90 = r + 1
            
        if visual_mode:
            client_items = []
            for c in range(n_clients):
                is_mal = (c in malicious_clients) and attack_active
                status = "flagged" if (c in flagged_clients) else ("malicious" if is_mal else "normal")
                client_items.append({
                    "id": c,
                    "acc": float(round(client_accs[c], 3)),
                    "status": status,
                    "shap_top3": client_shap_top3[c] if len(client_shap_top3) > c else [],
                    "anomaly_score": float(round(scores[c], 4)),
                    "weight": float(round(weights[c], 3))
                })
            events.append({
                "round": r,
                "clients": client_items,
                "global_acc": float(round(g_acc, 3)),
                "attack_active": bool(attack_active),
                "flagged": flagged_clients,
                "review_alerts": client_review_alerts
            })
            
    final_acc, final_f1, final_per_class, final_ece, _, _, _, final_rev_f1s = evaluate_model(global_model, X_test, y_test)
    tpr = (tpr_hits / max(1, active_attack_rounds * len(malicious_clients))) if active_attack_rounds > 0 else 0.0
    fpr = (fpr_hits / max(1, total_honest_evals)) if total_honest_evals > 0 else 0.0
    
    res = {
        "dataset": args.dataset,
        "aggregator": args.aggregator,
        "attack": args.attack,
        "alpha": args.alpha,
        "seed": args.seed,
        "xai_mode": args.xai_mode,
        "final_acc": final_acc,
        "macro_f1": final_f1,
        "per_class_f1": json.dumps({class_names[i]: round(final_per_class[i], 3) for i in range(len(class_names))}),
        "ece": final_ece,
        "rounds_to_0_90_f1": rounds_to_90,
        "bytes_communicated": total_bytes,
        "shap_dist_honest_mean": float(np.mean(shap_dists_honest_all)) if len(shap_dists_honest_all) > 0 else 0.0,
        "shap_dist_mal_mean": float(np.mean(shap_dists_mal_all)) if len(shap_dists_mal_all) > 0 else 0.0,
        "tpr": tpr,
        "fpr": fpr,
        "acc_history": ";".join([str(round(a, 3)) for a in round_accs]),
        **final_rev_f1s
    }
    return res, events

def train_centralized(X_train, y_train, X_test, y_test, input_dim, n_classes, epochs=40, batch_size=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = IoMTMLP(input_dim, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    acc, f1, per_class, ece, _, _, _, rev_f1s = evaluate_model(model, X_test, y_test)
    return {
        "final_acc": acc,
        "macro_f1": f1,
        "per_class_f1": json.dumps(per_class),
        "ece": ece,
        **rev_f1s
    }

def main():
    args = parse_args()
    
    if args.visual:
        X_raw, y_raw, feature_names, class_names, input_dim, n_classes = load_data(args.dataset, args.max_rows, seed=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=0)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        vis_args_xai = argparse.Namespace(**vars(args))
        vis_args_xai.seed = 0
        vis_args_xai.n_clients = 4
        vis_args_xai.alpha = 0.5
        vis_args_xai.attack = "scaled_update"
        vis_args_xai.aggregator = "xai_guided"
        vis_args_xai.rounds = 20
        _, events_xai = run_federated_experiment(vis_args_xai, X_tr, y_tr, X_te, y_te, feature_names, class_names, input_dim, n_classes, visual_mode=True)
        with open("events_xai.json", "w") as f:
            json.dump(events_xai, f, indent=2)
            
        vis_args_nodef = argparse.Namespace(**vars(args))
        vis_args_nodef.seed = 0
        vis_args_nodef.n_clients = 4
        vis_args_nodef.alpha = 0.5
        vis_args_nodef.attack = "scaled_update"
        vis_args_nodef.aggregator = "fedavg"
        vis_args_nodef.rounds = 20
        _, events_nodefense = run_federated_experiment(vis_args_nodef, X_tr, y_tr, X_te, y_te, feature_names, class_names, input_dim, n_classes, visual_mode=True)
        with open("events_nodefense.json", "w") as f:
            json.dump(events_nodefense, f, indent=2)
        print("Visual run complete: events_xai.json and events_nodefense.json created.")
        return

    if args.grid:
        aggregators = ["fedavg", "median", "trimmed_mean", "krum", "xai_guided"]
        attacks = ["none", "label_flip", "scaled_update", "stealth"]
        alphas = [0.1, 0.5, 1.0]
        seeds = [0, 1, 2]
        
        all_results = []
        
        for s in seeds:
            X_raw, y_raw, feature_names, class_names, input_dim, n_classes = load_data(args.dataset, args.max_rows, seed=s)
            X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=s)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            cent_res = train_centralized(X_tr, y_tr, X_te, y_te, input_dim, n_classes, epochs=args.rounds * args.local_epochs, lr=args.lr, seed=s)
            cent_row = {
                "dataset": args.dataset,
                "aggregator": "centralized",
                "attack": "none",
                "alpha": 0.5,
                "seed": s,
                "xai_mode": "none",
                "attack_impact": 0.0,
                "rounds_to_0_90_f1": 1,
                "bytes_communicated": 0,
                "shap_dist_honest_mean": 0.0,
                "shap_dist_mal_mean": 0.0,
                "tpr": 0.0,
                "fpr": 0.0,
                "acc_history": "",
                **cent_res
            }
            all_results.append(cent_row)
            
            for a_val in alphas:
                baseline_f1s = {}
                for agg in aggregators:
                    run_cfg = argparse.Namespace(**vars(args))
                    run_cfg.seed = s
                    run_cfg.alpha = a_val
                    run_cfg.aggregator = agg
                    run_cfg.attack = "none"
                    run_cfg.xai_mode = "shap"
                    r_res, _ = run_federated_experiment(run_cfg, X_tr, y_tr, X_te, y_te, feature_names, class_names, input_dim, n_classes)
                    baseline_f1s[agg] = r_res["macro_f1"]
                    r_res["attack_impact"] = 0.0
                    all_results.append(r_res)
                
                for att in ["label_flip", "scaled_update", "stealth"]:
                    for agg in aggregators:
                        run_cfg = argparse.Namespace(**vars(args))
                        run_cfg.seed = s
                        run_cfg.alpha = a_val
                        run_cfg.aggregator = agg
                        run_cfg.attack = att
                        run_cfg.xai_mode = "shap"
                        r_res, _ = run_federated_experiment(run_cfg, X_tr, y_tr, X_te, y_te, feature_names, class_names, input_dim, n_classes)
                        r_res["attack_impact"] = float(round(baseline_f1s[agg] - r_res["macro_f1"], 4))
                        all_results.append(r_res)
                        
                for abl_att in ["stealth", "scaled_update"]:
                    for abl_mode in ["weight_dist", "both"]:
                        run_cfg = argparse.Namespace(**vars(args))
                        run_cfg.seed = s
                        run_cfg.alpha = a_val
                        run_cfg.aggregator = "xai_guided"
                        run_cfg.attack = abl_att
                        run_cfg.xai_mode = abl_mode
                        r_res, _ = run_federated_experiment(run_cfg, X_tr, y_tr, X_te, y_te, feature_names, class_names, input_dim, n_classes)
                        r_res["attack_impact"] = float(round(baseline_f1s["xai_guided"] - r_res["macro_f1"], 4))
                        all_results.append(r_res)

        df_full = pd.DataFrame(all_results)
        df_full.to_csv("results_full.csv", index=False)
        
        num_cols = ["final_acc", "macro_f1", "attack_impact", "ece", "tpr", "fpr"]
        df_summary = df_full.groupby(["dataset", "aggregator", "attack", "alpha", "xai_mode"])[num_cols].agg(["mean", "std"]).reset_index()
        df_summary.columns = ["_".join(c).strip("_") for c in df_summary.columns]
        df_summary.to_csv("results_summary.csv", index=False)
        print("Experiment grid complete: results_full.csv and results_summary.csv created.")
        return

    X_raw, y_raw, feature_names, class_names, input_dim, n_classes = load_data(args.dataset, args.max_rows, seed=args.seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=args.seed)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    res, _ = run_federated_experiment(args, X_tr, y_tr, X_te, y_te, feature_names, class_names, input_dim, n_classes)
    print(f"Run finished: Acc: {res['final_acc']:.4f}, Macro-F1: {res['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
