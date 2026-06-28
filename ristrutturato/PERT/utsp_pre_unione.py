# -*- coding: utf-8 -*-
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import (
    OUTPUT_DIR, VALIDATION_SEED, N_VALIDATION_SCENARIOS,
    N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
    UTSP2_HIDDEN, UTSP2_NLAYERS, UTSP2_EPOCHS, UTSP2_LR,
    UTSP2_STEP_LR, UTSP2_LOG_FREQ, UTSP2_LAMBDA1, UTSP2_LAMBDA2,
    UTSP2_LAMBDA_E, UTSP2_ALPHA_DECODE,UTSP2_ALPHA_LOSS ,UTSP2_TEMP_MODE, UTSP2_TEMP_SCALE,
    UTSP2_TEMP_FIXED, UTSP2_DIST_SCALE_MODE,
    UTSP_LS_MAX_ACTIONS, UTSP_LS_ACTIONS_PER_ROUND,UTSP2_INCLUDE_PENALTY,UTSP2_INCLUDE_ENTROPY,
    UTSP_LS_MAX_RESTARTS, UTSP_LS_M, UTSP_LS_K,
    UTSP2_LS_ALPHA, UTSP_LS_BETA, UTSP_LS_RANDOM_SEED,
    UTSP_LS_APPLY_INITIAL_2OPT,UTSP2_LAMBDA_D,N_TRAINING_SCENARIOS_UTSP, UTSP_TRAINING_SEED,
    UTSP_BATCH_SIZE,
)
from tsp_utils import get_edge_value
from gurobi_models import solve_reservation_tsp, solve_exact_tsp
from scenarios import generate_scenarios, generate_scenario_batches
from evaluation import (
    validate_policies, genera_grafici_utsp,
    plot_utsp_heatmap, plot_utsp_graph_weights, plot_cost_distributions,
    compute_pi_with_booking_costs,
)
from two_stage_utsp_loss import (
    two_stage_utsp_loss,
    build_I_tensors,
    normalize_dist_tensor,
    decode_booking_policy,
    compute_heatmap,
    format_loss_components,
    check_booking_coverage,
)

_leaky = F.leaky_relu


# Prende il grafo (matrice di adiacenza W) e "propaga" le feature dei nodi attraverso i vicini.
# Ad ogni passo, ogni nodo aggrega le informazioni dei suoi vicini, pesate per il grado (la normalizzazione D). 
# L'ordine 3 significa che fai 3 passi di propagazione, quindi ogni nodo "vede" fino a 3 salti di distanza. Ritorna i risultati di ogni passo.
def _gcn_diffusion(W, order, feature, device):
    I_n = torch.eye(W.size(1), device=device).unsqueeze(0).expand(W.size(0), -1, -1)
    A   = W + I_n
    deg = torch.sum(A, 2, keepdim=True)
    D   = torch.pow(deg.clamp(min=1e-9), -0.5)
    res, x = [], feature
    for _ in range(order):
        x = D * x; x = torch.bmm(A, x); x = D * x
        res.append(x)
    return res


#Fa qualcosa di simile ma con una logica diversa: invece di passi discreti, fa una diffusione "lenta" (media pesata 0.5/0.5) 
#per 16 iterazioni e cattura le differenze tra scale diverse (buf[0]-buf[1], ecc.).
# È come una wavelet: cattura struttura locale a diverse risoluzioni del grafo.
def _scattering_diffusion(W, feature):
    deg, D = torch.sum(W, 2, keepdim=True).clamp(min=1e-9), None
    D   = torch.pow(deg, -1)
    buf, x = [], feature
    for i in range(16):
        x = 0.5 * x + 0.5 * torch.bmm(W, D * x)
        if i in [0, 1, 3, 7]:
            buf.append(x)
    return buf[0]-buf[1], buf[1]-buf[2], buf[2]-buf[3], buf[3]-feature*0


#Combina i due tipi di diffusione sopra tramite attention: per ogni nodo calcola quanto è rilevante ciascuna delle 6 rappresentazioni
# (2 da GCN + 4 da scattering) e le combina con pesi appresi. Poi passa attraverso due layer lineari. È il cuore della GNN
class _SCTConv(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.a       = nn.Parameter(torch.zeros(2 * hidden_dim, 1))
    
    def forward(self, X, adj, device):
        h_A, h_A2, _ = _gcn_diffusion(adj, 3, X, device)
        h_A, h_A2    = _leaky(h_A), _leaky(h_A2)
        s1, s2, s3, s4 = _scattering_diffusion(adj, X)
        s1, s2, s3, s4 = [torch.abs(s) for s in (s1, s2, s3, s4)]
        parts    = [h_A, h_A2, s1, s2, s3, s4]
        a_inputs = torch.stack([torch.cat([X, p], dim=2) for p in parts], dim=1)
        e        = torch.matmul(F.relu(a_inputs), self.a).squeeze(-1)
        attn     = F.softmax(e, dim=1).unsqueeze(-1)
        h_prime  = (attn * torch.stack(parts, dim=1)).sum(dim=1) # prima era mean
        return _leaky(self.linear2(_leaky(self.linear1(h_prime))))
    

class UTSP_GNN(nn.Module):
    def __init__(self, n_nodes, hidden_dim, n_layers):
        super().__init__()
        #self.bn0     = nn.BatchNorm1d(2) # pensavo fosse utile per le distanze normalizzate ma non cambia nulla, quindi non lo uso
        self.in_proj = nn.Linear(2, hidden_dim)
        self.convs   = nn.ModuleList([_SCTConv(hidden_dim) for _ in range(n_layers)])
        self.mlp1    = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim)
        self.mlp2    = nn.Linear(hidden_dim, n_nodes)
        self.softmax = nn.Softmax(dim=1)
        
# ORIGINALE
#    def forward(self, xy, adj, device):
#      B, N, _ = xy.shape
#      x       = self.bn0(xy.reshape(B * N, 2)).reshape(B, N, 2)
#      x       = _leaky(self.in_proj(x))
#      hidden  = x
#      for conv in self.convs:
#          x = conv(x, adj, device)
#          hidden = torch.cat([hidden, x], dim=-1)
#      return self.softmax(self.mlp2(_leaky(self.mlp1(hidden))))
      
    #NUOVA
    def forward(self, xy, adj, device):
        B, N, _ = xy.shape
        #x       = self.bn0(xy.reshape(B * N, 2)).reshape(B, N, 2)
        x       = _leaky(self.in_proj(xy))
        hidden  = x
        for conv in self.convs:
            x = conv(x, adj, device)
            hidden = torch.cat([hidden, x], dim=-1)
        logits = self.mlp2(_leaky(self.mlp1(hidden)))              # (B, N, N)
        return self.softmax(logits)                                # no mask diagonale

# Normalizzo le coordinate in [0,1]
def _normalize_coords(nodes, coords, device): 
    xs = np.array([coords[v][0] for v in nodes], dtype=np.float32)
    ys = np.array([coords[v][1] for v in nodes], dtype=np.float32)
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)
    xy = torch.from_numpy(np.stack([xs, ys], axis=1)).float().to(device)
    return xy   # (n, 2)

# calcolo temp per la adj
def _compute_temperature(dist_stack, mode, scale, fixed): 
    if mode == "fixed":
        return float(fixed)
    n    = dist_stack.size(-1)
    mask = ~torch.eye(n, dtype=torch.bool, device=dist_stack.device)
    mask = mask.unsqueeze(0).expand_as(dist_stack)
    vals = dist_stack[mask]
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return float(fixed)
    T = float(torch.median(vals).item()) * float(scale)
    return max(T, 1e-9)

#Costruisco xy e le matrici di distanza normalizzate per tutti gli scenari.
#xy_tile    : (K, n, 2)  — coordinate normalizzate, replicate per ogni scenario
#dist_raw   : list di K tensori (n, n) — distanze reali (non normalizzate)
#dist_model : (K, n, n)  — distanze normalizzate per GNN/loss
#dist_scale :   scala usata per la normalizzazione
#temperature:  temperatura T per il kernel gaussiano
def _build_input_tensors(scenario_ids, results, nodes, coords, device): 
    K = len(scenario_ids)
    n = len(nodes)
    xy = _normalize_coords(nodes, coords, device)  # (n, 2)
    xy_tile = xy.unsqueeze(0).expand(K, -1, -1).contiguous() # (K, n, 2)

    # Distanze reali per ogni scenario
    dist_raw = []
    for sid in scenario_ids:
        sd = results[sid]["scenario_dist"]
        D  = torch.zeros(n, n, device=device)
        for ii, i in enumerate(nodes):
            for jj, j in enumerate(nodes):
                if i != j:
                    D[ii, jj] = float(sd[i][j])
        dist_raw.append(D)

    dist_stack = torch.stack(dist_raw, dim=0)  # (K, n, n)

    # Normalizzazione interna UTSP
    dist_model, dist_scale = normalize_dist_tensor(
        dist_stack, mode=UTSP2_DIST_SCALE_MODE
    )

    temperature = _compute_temperature(
        dist_model, UTSP2_TEMP_MODE, UTSP2_TEMP_SCALE, UTSP2_TEMP_FIXED
    )

    return xy_tile, dist_raw, dist_model, dist_scale, temperature
    


#altra possibile formulazione AL MOMENTO NON LA STO USANDO PERCHE QUELLA DEL PAPER FUNZIONA
#tau : temperatura del kernel dopo riscalatura robusta
# eps : peso minimo morbido per ogni arco fuori diagonale
# q_scale : quantile usato come scala di riga
# clip_max : massimo valore normalizzato prima del kernel
def _build_adj_robust_floor( dist_stack, tau=UTSP2_TEMP_SCALE,
    eps=0.02, q_scale=0.90, clip_max=3.0, ): 

    K, n, _ = dist_stack.shape
    device = dist_stack.device

    D_scaled = torch.zeros_like(dist_stack)

    for s in range(K):
        for i in range(n):
            row = dist_stack[s, i].clone()
            row[i] = float("inf")

            vals = row[torch.isfinite(row)]
            vals = vals[vals > 0]

            if vals.numel() == 0:
                scale_i = torch.tensor(1.0, device=device)
            else:
                scale_i = torch.quantile(vals, q_scale).clamp(min=1e-9)

            D_scaled[s, i] = dist_stack[s, i] / scale_i

    D_scaled = torch.clamp(D_scaled, min=0.0, max=clip_max)
    adj = torch.exp(-D_scaled / max(tau, 1e-9))
    # soglia minima morbida: ogni arco fuori diagonale resta visibile
    adj = eps + (1.0 - eps) * adj
    return adj


# Addestro la GNN su batch di scenari
def _train_utsp_2stage(
    nodes, coords, scenario_ids, results, scenario_probs,
    I_mask, p_mat, C_mat, device,
    batches=None,
): 
    K = len(scenario_ids)
    n = len(nodes)

    # Costruisco tensori per TUTTI gli scenari (normalizzazione globale)
    xy_tile, dist_raw, dist_model, dist_scale, temperature = _build_input_tensors(
        scenario_ids, results, nodes, coords, device
    )
    probs_t = torch.tensor(
        [scenario_probs[sid] for sid in scenario_ids],
        dtype=torch.float32, device=device
    )

    print(f"[SEED_CHECK] torch.initial_seed()={torch.initial_seed()}")
    print(f"[SEED_CHECK] np.random state[0]={np.random.get_state()[1][0]}")
   
    # Seed locale dedicato all'inizializzazione GNN, indipendente dal flusso globale
    torch.manual_seed(UTSP_TRAINING_SEED)
    model = UTSP_GNN(n, UTSP2_HIDDEN, UTSP2_NLAYERS).to(device)
    
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(), lr=UTSP2_LR)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=UTSP2_STEP_LR, gamma=0.8
    )

    # Adiacenza per ogni scenario: adj_k = exp(-D^ω_norm / T)
    adj_stack = torch.exp(-dist_model / temperature)  # (K, n, n)

    # ── Costruzione batch slices ──────────────────────────────────────
    if batches is not None:
        # Mappa scenario_id → indice nel tensore globale
        sid_to_idx = {sid: k for k, sid in enumerate(scenario_ids)}
        batch_slices = []
        for batch in batches:
            b_sids = batch["scenario_ids"]
            b_indices = [sid_to_idx[sid] for sid in b_sids]
            b_probs = torch.tensor(
                [batch["scenario_probs"][sid] for sid in b_sids],
                dtype=torch.float32, device=device,
            )
            batch_slices.append({
                "indices": b_indices,
                "probs_t": b_probs,
                "K": len(b_sids),
            })
        n_batches = len(batch_slices)
    else:
        # Nessun batching: un unico batch con tutti gli scenari
        batch_slices = [{
            "indices": list(range(K)),
            "probs_t": probs_t,
            "K": K,
        }]
        n_batches = 1

    K_batch = batch_slices[0]["K"]
    print(f"\n  Training UTSP 2-stage | device={device} | n_params={n_par:,}")
    print(f"  GNN({n}→{UTSP2_HIDDEN}×{UTSP2_NLAYERS}) | "
          f"K_total={K} ({n_batches} batch × {K_batch}) | T={temperature:.4f} | scale={dist_scale:.4f}")
    print(f"  Epoche={UTSP2_EPOCHS}  lr={UTSP2_LR}  "
          f"λ1={UTSP2_LAMBDA1}  λ2={UTSP2_LAMBDA2}  λe={UTSP2_LAMBDA_E}  ")

    history = {"loss": [], "components": []}
    best_loss, best_state = float("inf"), None
    t0 = time.time()

    for epoch in range(1, UTSP2_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_comps = None

        for bs in batch_slices:
            idx = bs["indices"]
            K_b = bs["K"]
            xy_b   = xy_tile[idx]       # (K_b, n, 2)
            adj_b  = adj_stack[idx]     # (K_b, n, n)
            dist_b = dist_model[idx]    # (K_b, n, n)

            T_batch = model(xy_b, adj_b, device)                  # (K_b, n, n)
            T_list    = [T_batch[k:k+1] for k in range(K_b)]
            dist_list = [dist_b[k:k+1]  for k in range(K_b)]

            loss, comps = two_stage_utsp_loss(
                T_list, dist_list, I_mask, p_mat, C_mat, bs["probs_t"], 
                alpha=UTSP2_LS_ALPHA,
                lambda1=UTSP2_LAMBDA1,
                lambda2=UTSP2_LAMBDA2,
                lambda_e=UTSP2_LAMBDA_E,
                lambda_d=UTSP2_LAMBDA_D,
                include_penalty=UTSP2_INCLUDE_PENALTY,
                include_entropy=UTSP2_INCLUDE_ENTROPY,
                return_components=True,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_comps = comps       # ultimo batch come riferimento

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)
        history["components"].append(epoch_comps)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % UTSP2_LOG_FREQ == 0 or epoch == 1:
            marker = " ★" if abs(avg_loss - best_loss) < 1e-9 else ""
            print(f"  {format_loss_components(epoch_comps, epoch)} "
                  f"avg={avg_loss:.5f}{marker}")

    elapsed = time.time() - t0
    print(f"\n{'═'*65}")
    print(f"  TRAINING COMPLETATO")
    print(f"  Tempo totale          = {elapsed:.2f}s  ({elapsed/UTSP2_EPOCHS*1000:.1f}ms/epoca)")
    print(f"  Miglior loss          = {best_loss:.5f}  (ep {int(np.argmin(history['loss']))+1})")
    print(f"  Loss iniziale→finale  = {history['loss'][0]:.5f} → {history['loss'][-1]:.5f}")
    print(f"  Riduzione loss        = {(history['loss'][0]-best_loss)/history['loss'][0]*100:.1f}%")
    print(f"{'═'*65}")

    model.load_state_dict(best_state)
    model.eval()
    return model, history, adj_stack, dist_model, xy_tile, probs_t, temperature,  dist_scale
    
    
def _decode_policy(model, adj_stack, xy_tile, I, nodes, I_mask, probs_t, device):
    K = adj_stack.size(0)
    model.eval()
    with torch.no_grad():
        T_batch = model(xy_tile, adj_stack, device)        # (K, n, n)

    H_list = [compute_heatmap(T_batch[k:k+1]) for k in range(K)]

    x_reserved, x_scores = decode_booking_policy(
    H_list, I, nodes, I_mask, probs_t, alpha=UTSP2_LS_ALPHA )
    return x_reserved, x_scores, H_list, T_batch 
    
# Diagnostica numerica su T e H dopo il passaggio nella GNN per possibili problemi     
def debug_T_H(T_batch, H_list, name="TRAIN"): 

    with torch.no_grad():
        T = T_batch.detach().cpu()
        H = torch.stack([h.squeeze(0).detach().cpu() for h in H_list], dim=0)

        Tm = T.mean(dim=0).numpy()
        Hm = H.mean(dim=0).numpy()

        def stats(A, label):
            A = np.asarray(A)
            n = A.shape[0]
            mask = ~np.eye(n, dtype=bool)
            vals = A[mask]

            print(f"\n[{name}] {label}")
            print(f"  min fuori diagonale   = {vals.min():.6e}")
            print(f"  max fuori diagonale   = {vals.max():.6e}")
            print(f"  media fuori diagonale = {vals.mean():.6e}")
            print(f"  std fuori diagonale   = {vals.std():.6e}")
            print(f"  p50 fuori diagonale   = {np.percentile(vals, 50):.6e}")
            print(f"  p90 fuori diagonale   = {np.percentile(vals, 90):.6e}")
            print(f"  p99 fuori diagonale   = {np.percentile(vals, 99):.6e}")
            print(f"  somma righe min/max   = {A.sum(axis=1).min():.6f} / {A.sum(axis=1).max():.6f}")
            print(f"  somma colonne min/max = {A.sum(axis=0).min():.6f} / {A.sum(axis=0).max():.6f}")
            print(f"  diagonale somma       = {np.trace(A):.6e}")

        stats(Tm, "T medio")
        stats(Hm, "H medio")  

# Stampo la matrice di adiacenza media con valori numerici
def _print_adj_matrix(adj_stack, nodes, label="Matrice adiacenza (media scenari)"):
    adj_avg = adj_stack.detach().cpu().numpy().mean(axis=0)
    n = len(nodes)
    sep = "─" * (9 * n + 14)
    print(f"\n{'═'*65}")
    print(f"  {label}")
    print(sep)
    header = f"  {'i→j':>6} |" + "".join(f" {str(v):>7}" for v in nodes)
    print(header)
    print(sep)
    for ii, i in enumerate(nodes):
        row = f"  {str(i):>6} |" + "".join(
            f" {adj_avg[ii, jj]:>7.4f}" if ii != jj else f" {'—':>7}"
            for jj in range(n)
        )
        print(row)
    print(sep)
    # Statistiche sintetiche
    mask = ~np.eye(n, dtype=bool)
    vals = adj_avg[mask]
    print(f"  min={vals.min():.4f}  max={vals.max():.4f}  "
          f"media={vals.mean():.4f}  std={vals.std():.4f}")
    print(f"{'═'*65}")

#Entropia e copertura della heatmap media
def _print_heatmap_diagnostics(H_list, nodes, scenario_ids, label="Heatmap diagnostica"): 
    H_avg = None
    for H in H_list:
        arr = H.detach().squeeze(0).cpu().numpy()
        H_avg = arr.copy() if H_avg is None else H_avg + arr
    H_avg /= max(len(H_list), 1)

    n = len(nodes)
    mask = ~np.eye(n, dtype=bool)
    vals = H_avg[mask]
    vals_pos = vals[vals > 1e-9]

    # Entropia di Shannon normalizzata
    if vals_pos.sum() > 0:
        p = vals_pos / vals_pos.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        max_entropy = float(np.log(len(vals_pos)))
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        entropy, entropy_norm = 0.0, 0.0

    # Copertura: % archi con H > soglie
    thresholds = [0.05, 0.10, 0.20]
    print(f"\n{'═'*65}")
    print(f"  {label}")
    print(f"  Entropia Shannon         = {entropy:.4f}")
    print(f"  Entropia normalizzata    = {entropy_norm:.4f}  "
          f"(1.0=uniforme, 0.0=concentrata)")
    print(f"  H_avg: min={vals.min():.4f}  max={vals.max():.4f}  "
          f"media={vals.mean():.4f}")
    for thr in thresholds:
        n_above = int((vals > thr).sum())
        pct = 100.0 * n_above / len(vals)
        print(f"  Archi con H > {thr:.2f}        = {n_above}/{len(vals)} ({pct:.1f}%)")
    print(f"{'═'*65}")
  
    
# evaluation con stesso seme degli altri modelli gurobi
def _evaluate_policy_oos(
    policy_name, x_policy, nodes, E, base_dist, root, env,
    I, p, C, frequent_arcs, n_val, n_extra, mean_frac, sigma_frac):
    scenario_ids_val = list(range(1, n_val + 1))
    reservation = sum(get_edge_value(p, i, j) for (i, j) in x_policy)

    t0_val = time.time()
    results_val, _, _ = generate_scenarios(
        scenario_ids_val, nodes, E, base_dist, I, frequent_arcs,
        n_extra, mean_frac, sigma_frac, VALIDATION_SEED,
        root=root, env=env, p=p, C=C
    )
    t_gen = time.time() - t0_val
    print(f"  [val] Generazione {n_val} scenari: {t_gen:.2f}s")

    costs, tc_dict, pc_dict = {}, {}, {}
    t0_val2 = time.time()
    for sid in scenario_ids_val:
        sd  = results_val[sid]["scenario_dist"]
        sol = solve_reservation_tsp(
            nodes, E, I, sd, root, p, C, env,
            fixed_reservations=list(x_policy), output_flag=0,
            model_name=f"val_{policy_name}_{sid}"
        )
        tc  = sol["tour_cost"]    or 0.0
        pc  = sol["penalty_paid"] or 0.0
        costs[sid]   = reservation + tc + pc
        tc_dict[sid] = tc
        pc_dict[sid] = pc

    t_solve = time.time() - t0_val2
    print(f"  [val] Risoluzione {n_val} scenari (Gurobi): {t_solve:.2f}s  "
          f"({t_solve/n_val:.2f}s/scenario)")
    mean = sum(costs.values()) / len(costs)
    return costs, tc_dict, pc_dict, mean, results_val


 
# DIPENDENZE E LOCAL SEARCH 
 
# Costo di un tour su una matrice/dizionario di distanze orientate
def tour_cost(tour, dist): 
    n = len(tour)
    return sum(dist[tour[k]][tour[(k + 1) % n]] for k in range(n))

# decodifica H e tour con gurobi e non local search
def _decode_tour_gurobi(H, nodes, E, root, env):
    node_idx = {v: k for k, v in enumerate(nodes)}
    dist_neg = {
        i: {j: -float(H[node_idx[i], node_idx[j]])
            for j in nodes if j != i}
        for i in nodes
    }
    return solve_exact_tsp(nodes, E, dist_neg, root, env)


 
# LOCAL SEARCH STILE UTSP PAPER, ADATTATA AL CASO ORIENTATO 
# Ruota un tour senza ripetizione finale in modo che inizi da start.
def _rotate_tour_to_start(tour, start):
    
    if not tour or start not in tour:
        return list(tour)
    k = tour.index(start)
    return list(tour[k:] + tour[:k])

# Mantiene una rappresentazione canonica del tour con root in prima posizione.
def _rotate_tour_to_root(tour, root):
    return _rotate_tour_to_start(list(tour), root)

# Archi orientati del tour, incluso l'arco di ritorno all'inizio
def _tour_edges(tour):
    n = len(tour)
    return [(tour[i], tour[(i + 1) % n]) for i in range(n)]

# Costo di un tour su una matrice/dizionario di distanze orientate
def _tour_cost_on_dist(tour, dist):
    return tour_cost(tour, dist)


# 2-opt con ricalcolo completo del costo, quindi valido anche con costi orientati.
# Nel TSP asimmetrico l'inversione cambia i versi degli archi: non uso formule
# incrementali simmetriche, ma rivaluto tutto il tour.

def _two_opt_descent_directed(tour, dist, root, max_passes=50):
    
    if not tour or len(tour) <= 3:
        return list(tour), float("inf")

    best = _rotate_tour_to_root(tour, root)
    best_cost = _tour_cost_on_dist(best, dist)
    n = len(best)

    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 2, n + 1):
                if i == 1 and j == n:
                    continue
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                cand = _rotate_tour_to_root(cand, root)
                cand_cost = _tour_cost_on_dist(cand, dist)
                if cand_cost + 1e-9 < best_cost:
                    best, best_cost = cand, cand_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best, best_cost


# Or-opt (single-node relocation) per ATSP.
# Sposta un nodo alla volta nella posizione che riduce il costo.
# Non inverte segmenti: valida per grafi orientati.
def _or_opt_descent_directed(tour, dist, root, max_passes=50):
    
    if not tour or len(tour) <= 3:
        return list(tour), float("inf")

    best = _rotate_tour_to_root(list(tour), root)
    best_cost = _tour_cost_on_dist(best, dist)
    n = len(best)

    for _ in range(max_passes):
        improved = False
        for i in range(n):
            node = best[i]
            remaining = best[:i] + best[i + 1:]
            for j in range(len(remaining)):
                cand = remaining[:j + 1] + [node] + remaining[j + 1:]
                cand = _rotate_tour_to_root(cand, root)
                cand_cost = _tour_cost_on_dist(cand, dist)
                if cand_cost + 1e-9 < best_cost:
                    best, best_cost = cand, cand_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best, best_cost

#Per ogni nodo i costruisce i candidati j da provare nella local search.
# Priorità: top-M valori di H[i,j]. Se la riga è tutta nulla, fallback sui nodi
#più vicini secondo la distanza media reale.
def _build_heatmap_candidates(H, nodes, M, avg_dist=None):   
    idx = {v: k for k, v in enumerate(nodes)}
    candidates = {}

    for i in nodes:
        ii = idx[i]
        vals = []
        for j in nodes:
            if i == j:
                continue
            jj = idx[j]
            vals.append((float(H[ii, jj]), j))

        vals_sorted = sorted(vals, key=lambda x: x[0], reverse=True)
        top = [j for val, j in vals_sorted[:max(1, min(M, len(vals_sorted)))] if val > 0]

        if not top and avg_dist is not None:
            near = sorted(
                [(avg_dist[i][j], j) for j in nodes if j != i],
                key=lambda x: x[0]
            )
            top = [j for _, j in near[:max(1, min(M, len(near)))]]

        if not top:
            top = [j for _, j in vals_sorted[:max(1, min(M, len(vals_sorted)))]]

        candidates[i] = top

    return candidates

#Selezione stocastica guidata dalla heatmap, coerente con il criterio del paper:
    # valore heatmap + termine di esplorazione alpha * sqrt(log(S+1)/(N+1)).
def _select_heatmap_candidate(a, candidates, H_work, chosen_times, idx, rng, alpha, total_actions):
    
    cand = [b for b in candidates.get(a, []) if b != a]
    if not cand:
        return None

    ia = idx[a]
    scores = []
    for b in cand:
        ib = idx[b]
        explore = 0.0
        if alpha > 0:
            explore = alpha * math.sqrt(math.log(total_actions + 2.0) / (chosen_times[ia, ib] + 1.0))
        score = max(float(H_work[ia, ib]) + explore, 1e-12)
        scores.append(score)

    ssum = sum(scores)
    r = rng.random() * ssum
    acc = 0.0
    for b, sc in zip(cand, scores):
        acc += sc
        if r <= acc:
            chosen_times[ia, idx[b]] += 1
            return b

    b = cand[-1]
    chosen_times[ia, idx[b]] += 1
    return b

# Sposta b immediatamente dopo a
def _move_relocate_after(tour, a, b, root):
    if a == b or a not in tour or b not in tour:
        return list(tour)
    t = list(tour)
    t.remove(b)
    pos_a = t.index(a)
    t.insert(pos_a + 1, b)
    return _rotate_tour_to_root(t, root)


# Mossa tipo 2-opt che prova a rendere a→b un arco del tour.
# È una versione orientata/adattata: rivaluto sempre il costo completo
def _move_two_opt_make_edge(tour, a, b, root):
    if a == b or a not in tour or b not in tour:
        return list(tour)

    rot = _rotate_tour_to_start(list(tour), a)
    pos_b = rot.index(b)
    if pos_b == 1:
        return _rotate_tour_to_root(rot, root)
    if pos_b <= 0:
        return _rotate_tour_to_root(rot, root)

    cand = rot[:1] + list(reversed(rot[1:pos_b + 1])) + rot[pos_b + 1:]
    return _rotate_tour_to_root(cand, root)

# Scambia due nodi, lasciando poi root in prima posizione 
def _move_swap_nodes(tour, a, b, root):
    
    if a == b or a not in tour or b not in tour:
        return list(tour)
    t = list(tour)
    ia, ib = t.index(a), t.index(b)
    t[ia], t[ib] = t[ib], t[ia]
    return _rotate_tour_to_root(t, root)


def _random_tour(nodes, root, rng):
    rest = [v for v in nodes if v != root]
    rng.shuffle(rest)
    return [root] + rest

#  Ricerca locale guidata dalla heatmap
def _utsp_paper_style_local_search(tour_seed, H_decode, nodes, root, avg_dist):
    if not tour_seed or len(tour_seed) <= 3:
        return list(tour_seed), float("inf"), {"actions": 0, "restarts": 0, "improvements": 0}

    rng = random.Random(UTSP_LS_RANDOM_SEED)
    idx = {v: k for k, v in enumerate(nodes)}
    n = len(nodes)
    M = max(1, min(UTSP_LS_M, n - 1))

    H_work = np.array(H_decode, dtype=float, copy=True)
    chosen_times = np.zeros_like(H_work, dtype=float)
    candidates = _build_heatmap_candidates(H_work, nodes, M, avg_dist=avg_dist)

    current = _rotate_tour_to_root(tour_seed, root)
    current_cost = _tour_cost_on_dist(current, avg_dist)

    if UTSP_LS_APPLY_INITIAL_2OPT:
      current, current_cost = _or_opt_descent_directed(current, avg_dist, root)

    best = list(current)
    best_cost = current_cost

    total_actions = 0
    restarts = 0
    improvements = 0
    t0 = time.time()

    while total_actions < UTSP_LS_MAX_ACTIONS and restarts <= UTSP_LS_MAX_RESTARTS:
        best_round = None
        best_round_cost = current_cost
        best_round_added_edges = []

        for _ in range(UTSP_LS_ACTIONS_PER_ROUND):
            total_actions += 1
            if total_actions > UTSP_LS_MAX_ACTIONS:
                break

            a = rng.choice(nodes)
            b = _select_heatmap_candidate(
                a, candidates, H_work, chosen_times, idx, rng,
                UTSP2_LS_ALPHA, total_actions
            )
            if b is None:
                continue

            # Piccolo insieme di mosse locali. La candidate list da H decide cosa provare;
            # la bontà viene misurata sul costo medio reale.
            proposals = [
                _move_relocate_after(current, a, b, root),
                _move_two_opt_make_edge(current, a, b, root),
                _move_swap_nodes(current, a, b, root),
            ]

            # Profondità K: per n piccoli uso mosse composte relocate-after ripetute,
            # guidate da candidati successivi della heatmap.
            if UTSP_LS_K > 2:
                comp = list(current)
                aa = a
                for _depth in range(min(UTSP_LS_K, 4)):
                    bb = _select_heatmap_candidate(
                        aa, candidates, H_work, chosen_times, idx, rng,
                        UTSP2_LS_ALPHA, total_actions
                    )
                    if bb is None:
                        break
                    comp = _move_relocate_after(comp, aa, bb, root)
                    aa = bb
                proposals.append(comp)

            cur_edges = set(_tour_edges(current))
            for cand in proposals:
                if len(set(cand)) != n:
                    continue
                cand = _rotate_tour_to_root(cand, root)
                cand_cost = _tour_cost_on_dist(cand, avg_dist)
                if cand_cost + 1e-9 < best_round_cost:
                    cand_edges = set(_tour_edges(cand))
                    added = list(cand_edges - cur_edges)
                    best_round = cand
                    best_round_cost = cand_cost
                    best_round_added_edges = added

        if best_round is not None:
            before = max(current_cost, 1e-9)
            gain = current_cost - best_round_cost
            current = best_round
            current_cost = best_round_cost
            improvements += 1

            # Backpropagation/update stile paper: aumenta peso degli archi che hanno
            # prodotto miglioramento.
            inc = UTSP_LS_BETA * (math.exp(max(gain, 0.0) / before) - 1.0)
            if inc > 0 and best_round_added_edges:
                for i, j in best_round_added_edges:
                    if i in idx and j in idx and i != j:
                        H_work[idx[i], idx[j]] += inc
                candidates = _build_heatmap_candidates(H_work, nodes, M, avg_dist=avg_dist)

            if current_cost + 1e-9 < best_cost:
                best = list(current)
                best_cost = current_cost
        else:
            restarts += 1
            current = _random_tour(nodes, root, rng)
            current_cost = _tour_cost_on_dist(current, avg_dist)
            if UTSP_LS_APPLY_INITIAL_2OPT:
               current, current_cost = _or_opt_descent_directed(current, avg_dist, root)
            if current_cost + 1e-9 < best_cost:
                best = list(current)
                best_cost = current_cost

    info = {
        "actions": total_actions,
        "restarts": restarts,
        "improvements": improvements,
        "seconds": time.time() - t0,
        "final_heatmap_max": float(H_work.max()) if H_work.size else 0.0,
    }
    return best, best_cost, info

# Stampa quanto A è diversa dalla sua trasposta
def _matrix_asymmetry_stats(A, name):
    
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        print(f"  Asimmetria {name}: matrice non quadrata, salto diagnostica.")
        return

    n = A.shape[0]
    mask = ~np.eye(n, dtype=bool)
    diff = A - A.T
    abs_diff = np.abs(diff[mask])
    abs_vals = np.abs(A[mask])

    denom = float(np.mean(abs_vals)) + 1e-12
    print(
        f"  Asimmetria {name}: "
        f"mean|A-A.T|={float(np.mean(abs_diff)):.6e}  "
        f"p90={float(np.percentile(abs_diff, 90)):.6e}  "
        f"max={float(np.max(abs_diff)):.6e}  "
        f"rel_mean={float(np.mean(abs_diff))/denom:.6f}"
    )


def _build_mean_dist_from_results(results, scenario_ids, nodes):
    avg = {i: {} for i in nodes}
    for i in nodes:
        for j in nodes:
            if i == j:
                avg[i][j] = 0.0
            else:
                avg[i][j] = float(np.mean([results[sid]["scenario_dist"][i][j] for sid in scenario_ids]))
    return avg
 # Tour greedy nearest-neighbor come seed per la local search   
def _greedy_tour(nodes, root, dist):
    unvisited = set(nodes) - {root}
    tour, cur = [root], root
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[cur].get(j, float("inf")))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour

# Orginale: Costruisce la matrice di adiacenza per un singolo scenario
def _build_adj_single(D, n, device, dist_scale, temperature):
    dist_norm = D / max(dist_scale, 1e-9)
    return torch.exp(-dist_norm / max(temperature, 1e-9))  # (n, n)
#def _build_adj_single(D, n, device, dist_scale, temperature, k=3, min_weight=0.2): 
#    """
#    Costruisce la matrice di adiacenza per un singolo scenario.
#    Usa la stessa normalizzazione (dist_scale, temperature) del training,
#    in modo che la GNN veda input nella stessa scala.
#    D: (n, n) tensor delle distanze reali dello scenario.
 
#    Applica la stessa garanzia k-NN di _build_adj_with_knn_guarantee,
#    così ogni nodo ha almeno k vicini con peso >= min_weight anche in inferenza.
#    """
#    dist_norm = D / max(dist_scale, 1e-9)
    # Riusa la funzione di training: aggiunge dimensione batch (1, n, n) poi la rimuove
#    adj = _build_adj_with_knn_guarantee(
#        dist_norm.unsqueeze(0), max(temperature, 1e-9), k=k, min_weight=min_weight
#    )
#    return adj.squeeze(0)  # (n, n)


# Forward pass della GNN su un singolo scenario nuovo
def _gnn_heatmap_single(model, xy, adj_single, device): 
    model.eval()
    with torch.no_grad():
        T_out = model(xy.unsqueeze(0), adj_single.unsqueeze(0), device)  # (1,n,n)
        H_t   = compute_heatmap(T_out)
    H = H_t.squeeze(0).cpu().numpy().copy()
    return H

# Distanze effettive per il secondo stadio della LS: aggiungo la penale C[i,j] agli archi di I non prenotat
def _effective_dist_for_ls(dist_s, nodes, I, x_ls, C): 
    from tsp_utils import canon_edge as _ce
    x_set = {_ce(i, j) for (i, j) in x_ls}
    eff   = {i: dict(dist_s[i]) for i in nodes}
    for (i, j) in I:
        if _ce(i, j) not in x_set:
            pen      = get_edge_value(C, i, j)
            eff[i][j] = dist_s[i][j] + pen
            eff[j][i] = dist_s[j][i] + pen
    return eff

# costo reale secondo stadio con percorrenze vere e penale degli archi
def _ls_cost_with_penalty(tour, dist_s, nodes, I, x_ls, C): 
    from tsp_utils import canon_edge as _ce
    x_set = {_ce(i, j) for (i, j) in x_ls}
    I_set = {_ce(i, j) for (i, j) in I}
    arcs  = _tour_edges(tour)
    tc    = tour_cost(tour, dist_s)
    pc    = sum(
        get_edge_value(C, i, j)
        for (i, j) in arcs
        if _ce(i, j) in I_set and _ce(i, j) not in x_set
    )
    return tc, pc

#Esegue il secondo stadio local search per ogni scenario
def _run_ls_on_scenarios(
    model, xy, nodes, root, I, p, C,
    results_scenarios, scenario_ids, scenario_probs,
    H_list_precomputed, dist_scale, temperature, device, x_ls, label="train",
):
    n      = len(nodes)
    reserv = sum(get_edge_value(p, i, j) for (i, j) in x_ls)
    costs, tc_dict, pc_dict, tours, solutions = {}, {}, {}, {}, {}

    t0_ls = time.time()
    for k, sid in enumerate(scenario_ids):
        t0_sid = time.time()
        dist_s = results_scenarios[sid]["scenario_dist"]

        # Heatmap scenario-specifica  
        if H_list_precomputed is not None:
            H_t = H_list_precomputed[k]             # (1, n, n) tensor training
            H_s = H_t.detach().squeeze(0).cpu().numpy().copy()
        else:
            D = torch.zeros(n, n, device=device)
            for ii, i in enumerate(nodes):
                for jj, j in enumerate(nodes):
                    if i != j:
                        D[ii, jj] = float(dist_s[i][j])
            #originale
            adj_s = _build_adj_single(D, n, device, dist_scale, temperature)
            
             
            #nuova versione
            # DOPO
            #adj_s = _build_adj_robust_floor(  D.unsqueeze(0) / max(dist_scale, 1e-9), tau=UTSP2_TEMP_SCALE, eps=0.02, q_scale=0.90, clip_max=3.0, ).squeeze(0)
            H_s = _gnn_heatmap_single(model, xy, adj_s, device)
            
              
        # Distanze effettive + LS  
        dist_eff = _effective_dist_for_ls(dist_s, nodes, I, x_ls, C)
        seed     = _greedy_tour(nodes, root, dist_eff)
        tour_s, _, ls_info = _utsp_paper_style_local_search(
            seed, H_s, nodes, root, dist_eff
        )

        # Costo reale (distanze vere + penale) 
        tc, pc = _ls_cost_with_penalty(tour_s, dist_s, nodes, I, x_ls, C)
        arcs_s = _tour_edges(tour_s)

        elapsed_sid = time.time() - t0_sid
        costs[sid]   = reserv + tc + pc
        tc_dict[sid] = tc
        pc_dict[sid] = pc
        tours[sid]   = tour_s
        solutions[sid] = {
            "arcs":             arcs_s,
            "y_used":           arcs_s,
            "tour":             list(tour_s),
            "tour_cost":        tc,
            "total_cost":       costs[sid],
            "penalty_paid":     pc,
            "reservation_paid": reserv,
            "x_used":           list(x_ls),
        }

        print(f"    [{label}] s={sid}: tc={tc:.4f}  pc={pc:.4f}  "
              f"tot={costs[sid]:.4f}  t={elapsed_sid:.2f}s  tour={tour_s}")

    elapsed_ls = time.time() - t0_ls
    n_scen = len(scenario_ids)
    print(f"  [{label}] Local search completata: {n_scen} scenari in "
          f"{elapsed_ls:.2f}s  ({elapsed_ls/n_scen:.2f}s/scenario)")
    # Report per batch
    cost_list = [costs[sid] for sid in scenario_ids]
    n_batches_report = len(cost_list) // UTSP_BATCH_SIZE
    if n_batches_report > 1:
        for b_idx in range(n_batches_report):
            chunk = cost_list[b_idx*UTSP_BATCH_SIZE : (b_idx+1)*UTSP_BATCH_SIZE]
            print(f"    [{label}] Batch {b_idx+1}/{n_batches_report}: "
                  f"media={sum(chunk)/len(chunk):.4f}  "
                  f"min={min(chunk):.4f}  max={max(chunk):.4f}")
    if scenario_probs is not None:
        mean = sum(scenario_probs[sid] * costs[sid] for sid in scenario_ids)
    else:
        mean = sum(costs.values()) / len(costs)

    return costs, tc_dict, pc_dict, tours, solutions, mean

def _heatmap_numpy_from_H_list(H_list):
    H_sum = None
    for H in H_list:
        arr = H.detach().squeeze(0).cpu().numpy()
        H_sum = arr.copy() if H_sum is None else H_sum + arr
    return H_sum / max(len(H_list), 1)

#Costruisco la heatmap aggregata pesata    
def _heatmap_bar_from_H_list(H_list, scenario_ids, scenario_probs):
    H_bar = None

    for sid, H in zip(scenario_ids, H_list):
        p_omega = float(scenario_probs[sid])
        H_omega = H.detach().squeeze(0).cpu().numpy()

        if H_bar is None:
            H_bar = p_omega * H_omega
        else:
            H_bar += p_omega * H_omega

    return H_bar


def _evaluate_fixed_tour_on_scenarios(tour, results, scenario_ids):
    costs = {}
    arcs = _tour_edges(tour) if tour else []
    solutions = {}
    for sid in scenario_ids:
        dist = results[sid]["scenario_dist"]
        cost = tour_cost(tour, dist) if tour else None
        costs[sid] = cost
        solutions[sid] = {
            "arcs": arcs,
            "y_used": arcs,
            "tour": list(tour),
            "tour_cost": cost,
            "total_cost": cost,
            "penalty_paid": 0.0,
            "reservation_paid": 0.0,
            "x_used": [],
        }
    mean_cost = float(np.mean([c for c in costs.values() if c is not None])) if costs else None
    return costs, solutions, mean_cost


def _run_heatmap_local_search(nodes, E, root, env, results, scenario_ids, H_list):
    H_raw = _heatmap_numpy_from_H_list(H_list)
    H_decode = H_raw.copy()

    _matrix_asymmetry_stats(H_raw, "H_raw UTSP 2-stage")
    _matrix_asymmetry_stats(H_decode, "H_decode UTSP 2-stage")

    avg_dist = _build_mean_dist_from_results(results, scenario_ids, nodes)

    print("\n  Decodifica tour da heatmap con Gurobi ...")
    decoded = _decode_tour_gurobi(H_decode, nodes, E, root, env)
    if isinstance(decoded, dict):
        tour_seed = decoded.get("tour", [])
        arcs_seed = decoded.get("arcs", [])
        seed_cost = tour_cost(tour_seed, avg_dist) if tour_seed else None
    else:
        seed_cost, arcs_seed = decoded
        tour_seed = []

    if not tour_seed and arcs_seed:
        succ = {i: j for (i, j) in arcs_seed}
        tour_seed = [root]
        cur = root
        for _ in range(len(nodes)):
            nxt = succ.get(cur)
            if nxt is None or nxt == root:
                break
            tour_seed.append(nxt)
            cur = nxt
        seed_cost = tour_cost(tour_seed, avg_dist) if tour_seed else None

    print(f"  Tour iniziale heatmap: {tour_seed}")
    print(f"  Costo su distanza media training: {seed_cost if seed_cost is not None else 'N/A'}")

    print("\n  Local search UTSP guidata da H ...")
    tour_ls, ls_cost_avg, ls_info = _utsp_paper_style_local_search(
        tour_seed, H_decode, nodes, root, avg_dist
    )
    arcs_ls = _tour_edges(tour_ls) if tour_ls else []

    print(f"  Tour UTSP-LS: {tour_ls}")
    print(f"  Costo UTSP-LS su distanza media training: {ls_cost_avg:.4f}")
    print(f"  Info local search: {ls_info}")

    return {
        "H_raw": H_raw,
        "H_decode": H_decode,
        "avg_dist": avg_dist,
        "tour_seed": tour_seed,
        "arcs_seed": arcs_seed,
        "seed_cost_avg": seed_cost,
        "tour_ls": tour_ls,
        "arcs_ls": arcs_ls,
        "ls_cost_avg": ls_cost_avg,
        "ls_info": ls_info,
    }

# per far andare l esperimento intero
def run_esperimento_B_UTSP(nodes, coords, base_dist, E, root, env, res_B, mode="local search"):
    """
    mode:
      - "policy"       : x_utsp + secondo stadio Gurobi;
      - "local_search" : H_avg + decodifica + local search UTSP;
      - "both"         : entrambe le valutazioni con un solo training.
    """
    mode = (mode or "both").lower()
    if mode not in {"policy", "local_search", "both"}:
        raise ValueError("mode deve essere 'policy', 'local_search' oppure 'both'.")

    do_policy = mode in {"policy", "both"}
    do_ls = mode in {"local_search", "both"}

    print("\n" + "=" * 70)
    print(f"ESPERIMENTO B — UTSP 2-STADI | mode={mode}")

    needed = ["I", "p", "C", "b", "results", "scenario_probs",
              "frequent_arcs", "PI", "STO", "EEV",
              "stoch_costs", "eev_costs", "x_ev", "x_used_sto",
              "eev_solutions", "stoch_solutions"]
    missing = [k for k in needed if k not in res_B]
    if missing:
        raise KeyError(f"res_B manca delle chiavi: {missing}")

    I = res_B["I"]
    p = res_B["p"]
    C = res_B["C"]
    b = res_B["b"]
    frequent_arcs = res_B["frequent_arcs"]

    # Scenari piccoli già generati da Esperimento B:
    # servono per benchmark, grafici e riepilogo.
    results_B = res_B["results"]
    scenario_probs_B = res_B["scenario_probs"]
    scenario_ids_B = list(results_B.keys())

    # Alias di compatibilità, così il codice sotto che usa ancora
    # results / scenario_probs / scenario_ids non si rompe.
    results = results_B
    scenario_probs = scenario_probs_B
    scenario_ids = scenario_ids_B

    PI_train = res_B["PI"]
    STO_train = res_B["STO"]
    EEV_train = res_B["EEV"]

    # ── Scenari UTSP training: N_TRAINING_SCENARIOS_UTSP / UTSP_BATCH_SIZE batch ──
    batches_utsp = generate_scenario_batches(
        nodes, E, base_dist, I, frequent_arcs,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, UTSP_TRAINING_SEED,
        root, env, p, C,
        n_scenarios=N_TRAINING_SCENARIOS_UTSP,
        batch_size=UTSP_BATCH_SIZE,
    )

    # Combino tutti i risultati per calcoli globali (dist_scale, decode, ecc.)
    results_utsp = {}
    scenario_ids_utsp = []
    scenario_probs_utsp = {}
    for batch in batches_utsp:
        results_utsp.update(batch["results"])
        scenario_ids_utsp.extend(batch["scenario_ids"])
        scenario_probs_utsp.update(batch["scenario_probs"])
    
    
    frequent_arcs = res_B["frequent_arcs"]
    PI_train = res_B["PI"]
    STO_train = res_B["STO"]
    EEV_train = res_B["EEV"]

    n = len(nodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    I_mask, p_mat, C_mat, node_idx = build_I_tensors(I, nodes, p, C, device)


    _, _, _, dist_scale, _ = _build_input_tensors( scenario_ids_utsp, results_utsp, nodes, coords, device )
    

    p_mat = p_mat / dist_scale
    C_mat = C_mat / dist_scale
    
    assert I_mask.sum() > 0, f"I_mask è vuota! Controlla build_I_tensors e la lista I passata"
    print(f"Archi in I: {I_mask.sum().item() // 2} coppie non orientate")
    print(f"p_mat nonzero: {(p_mat > 0).sum().item()} celle")
    print(f"p_mat range: [{p_mat[p_mat>0].min().item():.4f}, {p_mat.max().item():.4f}]")
    print(f"dist_scale: {dist_scale:.4f}")
    
    
    
    
    print(f"\n  Setup: |I|={len(I)}  n_nodi={n}  "
          f"K_train_UTSP={len(scenario_ids_utsp)} ({len(batches_utsp)} batch × {UTSP_BATCH_SIZE})  "
          f"device={device}")
    print(f"  Scenari B per grafici/benchmark = {len(scenario_ids_B)}")
    print(f"  I = {I}")

    (model, history, adj_stack, dist_model,
     xy_tile, probs_t, temperature, dist_scale) = _train_utsp_2stage(
        nodes,
        coords,
        scenario_ids_utsp,
        results_utsp,
        scenario_probs_utsp,
        I_mask,
        p_mat,
        C_mat,
        device,
        batches=batches_utsp,
    )

    x_utsp, x_scores, H_list, T_batch = _decode_policy(
        model, adj_stack, xy_tile, I, nodes, I_mask, probs_t, device,
    )
    
    debug_T_H(T_batch, H_list, name="dopo training UTSP")

    # Diagnostica adiacenza e heatmap  
    _print_adj_matrix(adj_stack, nodes, label="Matrice adiacenza GNN (media scenari training)")
    _print_heatmap_diagnostics(H_list, nodes, scenario_ids_utsp,
                               label="Heatmap diagnostica (media scenari training)")

    reservation_utsp = sum(get_edge_value(p, i, j) for (i, j) in x_utsp)
    # Visualizzazioni heatmap e grafo pesato 
    _H_avg = _heatmap_numpy_from_H_list(H_list)
    _H_decode_vis = _H_avg.copy()
    
    plot_utsp_heatmap(
        "espB_UTSP", nodes, _H_decode_vis,
        title_suffix=f"(media {len(H_list)} scenari)",
    )
    plot_utsp_graph_weights(
        "espB_UTSP", nodes, coords, _H_avg,
        title_suffix=f"(media {len(H_list)} scenari)",
    )

    print(f"\n{check_booking_coverage(x_scores, I)}")
    print(f"\n  Costo prenotazione UTSP : {reservation_utsp:.4f}")
        # Visualizzazione dell'adiacenza dopo il kernel 
    _adj_avg = adj_stack.detach().cpu().numpy().mean(axis=0)

    plot_utsp_heatmap(
        "espB_UTSP_adj_kernel",
        nodes,
        _adj_avg,
        title_suffix=f"(adj_stack media su {adj_stack.shape[0]} scenari — dopo kernel)",
    )

    plot_utsp_graph_weights(
        "espB_UTSP_adj_kernel",
        nodes,
        coords,
        _adj_avg,
        title_suffix=f"(adj_stack media su {adj_stack.shape[0]} scenari — dopo kernel)",
    )
    output = {
        "model": model,
        "history": history,
        "x_utsp": x_utsp,
        "x_scores": x_scores,
        "I_mask": I_mask,
    }

    if do_policy:
        policy_out = _run_policy_branch(
            nodes,
            coords,
            E,
            base_dist,
            root,
            env,
            res_B,
            I,
            p,
            C,
            b,
            results_B,
            scenario_ids_B,
            scenario_probs_B,
            frequent_arcs,
            PI_train,
            STO_train,
            EEV_train,
            x_utsp,
            x_scores,
            reservation_utsp,
            history,
            temperature,
        )
        output.update(policy_out)

    if do_ls:
        ls_out = _run_local_search_branch(
            nodes, coords, E, root, env, res_B,
            results, scenario_ids, H_list,
            PI_train, STO_train, EEV_train,
            history, temperature,
            model=model,
            xy=xy_tile[0],        # (n, 2) — coordinate normalizzate
            dist_scale=dist_scale,
            device=device,
            x_utsp=x_utsp,
            base_dist=base_dist,
        )
        output.update({"local_search": ls_out})

    return output


def _run_policy_branch(
    nodes, coords, E, base_dist, root, env, res_B,
    I, p, C, b, results, scenario_ids, scenario_probs,
    frequent_arcs, PI_train, STO_train, EEV_train,
    x_utsp, x_scores, reservation_utsp,
    history, temperature,
):
    utsp_costs_train, utsp_tc_train, utsp_pc_train = {}, {}, {}
    utsp_solutions_train = {}

    for sid in scenario_ids:
        sd = results[sid]["scenario_dist"]
        sol = solve_reservation_tsp(
            nodes, E, I, sd, root, p, C, env,
            fixed_reservations=x_utsp, output_flag=0
        )
        tc = sol["tour_cost"] or 0.0
        pc = sol["penalty_paid"] or 0.0
        utsp_costs_train[sid] = reservation_utsp + tc + pc
        utsp_tc_train[sid] = tc
        utsp_pc_train[sid] = pc
        utsp_solutions_train[sid] = sol

    UTSP_train = sum(scenario_probs[sid] * utsp_costs_train[sid] for sid in scenario_ids)

    print(f"\n  Validazione out-of-sample ({N_VALIDATION_SCENARIOS} scenari, seme={VALIDATION_SEED}) ...")
    utsp_val, utsp_tc_val, utsp_pc_val, UTSP_val, results_val = _evaluate_policy_oos(
        "utsp", x_utsp, nodes, E, base_dist, root, env,
        I, p, C, frequent_arcs,
        N_VALIDATION_SCENARIOS, N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC
    )

    val_results = validate_policies(
        nodes, E, base_dist, root, env, I, p, C,
        res_B["x_used_sto"], res_B["x_ev"],
        frequent_arcs, N_VALIDATION_SCENARIOS,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
        exp_name="espB"
    )
    
    scenario_ids_test_8 = scenario_ids_val[:8]

    genera_grafici_utsp(
        exp_name="espB_UTSP_LS_TEST",
        nodes=nodes,
        coords=coords,
        scenario_ids=scenario_ids_test_8,
        results=results_val,
    
        eev_costs=val_bench["eev_costs"],
        eev_solutions=val_bench["eev_solutions_val"],
    
        stoch_costs=val_bench["sto_costs"],
        stoch_solutions=val_bench["sto_solutions_val"],
    
        utsp_costs=costs_val,
        utsp_solutions=solutions_val,
    
        x_ev=res_B["x_ev"],
        x_sto=res_B["x_used_sto"],
        x_utsp=x_ls,
    
        utsp_label="UTSP local search",
        save=True,
    )
    
    plot_cost_distributions(
        eev_costs=val_bench["eev_costs"],
        stoch_costs=val_bench["sto_costs"],
        utsp_costs=costs_val,
        exp_name="espB",
        split_label="test",
    )
    
    PI_val = val_results["PI_val"]
    STO_val = val_results["STO_val"]
    EEV_val = val_results["EEV_val"]

    gap_utsp_sto = (UTSP_val - STO_val) / abs(STO_val) * 100 if STO_val else float("nan")
    gap_utsp_eev = (UTSP_val - EEV_val) / abs(EEV_val) * 100 if EEV_val else float("nan")
    gap_utsp_pi = (UTSP_val - PI_val) / abs(PI_val) * 100 if PI_val else float("nan")

    print("\n" + "─" * 65)
    print("RIEPILOGO ESPERIMENTO B — UTSP 2-STADI")
    print(f"  Tratte prenotate  : {sorted(x_utsp)}  ({len(x_utsp)}/{len(I)})")
    print(f"  Costo prenotazione: {reservation_utsp:.4f}")
    print()
    print("  TRAINING (8 scenari)")
    print(f"    PI    = {PI_train:.4f}")
    print(f"    STO   = {STO_train:.4f}")
    print(f"    EEV   = {EEV_train:.4f}")
    print(f"    UTSP  = {UTSP_train:.4f}")
    print()
    print(f"  VALIDAZIONE ({N_VALIDATION_SCENARIOS} scenari, seme={VALIDATION_SEED})")
    print(f"    PI_val   = {PI_val:.4f}")
    print(f"    STO_val  = {STO_val:.4f}")
    print(f"    EEV_val  = {EEV_val:.4f}")
    print(f"    UTSP_val = {UTSP_val:.4f}")
    print()
    print(f"  Gap UTSP vs STO (val) = {gap_utsp_sto:+.2f}%")
    print(f"  Gap UTSP vs EEV (val) = {gap_utsp_eev:+.2f}%")
    print(f"  Gap UTSP vs PI  (val) = {gap_utsp_pi:+.2f}%")
    print("─" * 65)

    _save_utsp_summary(
        exp_name="espB_UTSP",
        scenario_ids=scenario_ids,
        results=results,
        I=I, p=p, C=C, b=b,
        x_utsp=x_utsp,
        x_scores=x_scores,
        utsp_costs_train=utsp_costs_train,
        utsp_tc_train=utsp_tc_train,
        utsp_pc_train=utsp_pc_train,
        UTSP_train=UTSP_train,
        PI_train=PI_train,
        STO_train=STO_train,
        EEV_train=EEV_train,
        UTSP_val=UTSP_val,
        PI_val=PI_val,
        STO_val=STO_val,
        EEV_val=EEV_val,
        utsp_val=utsp_val,
        utsp_tc_val=utsp_tc_val,
        utsp_pc_val=utsp_pc_val,
        gap_utsp_sto=gap_utsp_sto,
        gap_utsp_eev=gap_utsp_eev,
        gap_utsp_pi=gap_utsp_pi,
        history=history,
        temperature=temperature,
    )

    

    return {
        "utsp_costs_train": utsp_costs_train,
        "utsp_tc_train": utsp_tc_train,
        "utsp_pc_train": utsp_pc_train,
        "utsp_solutions_train": utsp_solutions_train,
        "UTSP_train": UTSP_train,
        "utsp_val": utsp_val,
        "UTSP_val": UTSP_val,
        "PI_val": PI_val,
        "STO_val": STO_val,
        "EEV_val": EEV_val,
        "gap_utsp_sto": gap_utsp_sto,
        "gap_utsp_eev": gap_utsp_eev,
        "gap_utsp_pi": gap_utsp_pi,
    }


def _compute_bookings_from_tours(tours, scenario_ids, nodes, I, p, C):
    """
    Prenotazioni ottimali post-LS: prenoto arco (i,j) ∈ I se la frequenza
    d'uso nei tour supera la soglia analitica p/C.
    """
    from tsp_utils import canon_edge as _ce

    I_set = {_ce(i, j) for (i, j) in I}
    n_scenarios = len(scenario_ids)

    usage_count = {edge: 0 for edge in I_set}
    for sid in scenario_ids:
        tour = tours[sid]
        arcs = [(tour[k], tour[(k + 1) % len(tour)]) for k in range(len(tour))]
        for (a, b) in arcs:
            ce = _ce(a, b)
            if ce in I_set:
                usage_count[ce] += 1

    x_opt = []
    print("    Prenotazioni post-LS (f > p/C):")
    for edge in sorted(I_set):
        i, j = edge
        f = usage_count[edge] / n_scenarios
        p_val = get_edge_value(p, i, j)
        C_val = get_edge_value(C, i, j)
        soglia = p_val / C_val if C_val > 0 else float('inf')
        book = f > soglia
        if book:
            x_opt.append(edge)
        flag = "✓ PRENOTA" if book else "✗ no"
        print(f"      {{{i},{j}}}  f={f:.3f}  soglia={soglia:.3f}  {flag}")
    print(f"    Totale: {len(x_opt)}/{len(I_set)}")
    return x_opt


def _run_local_search_branch(
    nodes, coords, E, root, env, res_B,
    results, scenario_ids, H_list,
    PI_train, STO_train, EEV_train,
    history, temperature,
    model, xy, dist_scale, device, x_utsp, base_dist,
):
    print("\n" + "=" * 70)
    print("ESPERIMENTO B — UTSP HEATMAP + LOCAL SEARCH (2-stadi, per-scenario)")

    I              = res_B["I"]
    p              = res_B["p"]
    C              = res_B["C"]
    scenario_probs = res_B["scenario_probs"]
    frequent_arcs  = res_B["frequent_arcs"]

    # LS senza prenotazioni — decise post-hoc con formula f > p/C
    print(f"\n  Primo stadio: prenotazioni decise post-LS (formula f > p/C)")

    # Secondo stadio LS su scenari di training (senza prenotazioni)
    print(f"\n  LS su {len(scenario_ids)} scenari di training ...")
    (costs_train, tc_train, pc_train,
     tours_train, solutions_train, UTSP_LS_train) = _run_ls_on_scenarios(
        model, xy, nodes, root, I, p, C,
        results, scenario_ids, scenario_probs,
        H_list_precomputed=H_list,
        dist_scale=dist_scale, temperature=temperature,
        device=device, x_ls=[], label="train",
    )

    # Prenotazioni ottimali dai tour di training
    x_ls = _compute_bookings_from_tours(tours_train, scenario_ids, nodes, I, p, C)
    reserv = sum(get_edge_value(p, i, j) for (i, j) in x_ls)
    print(f"  Costo prenotazione post-LS: {reserv:.4f}")

    # Ricalcolo costi training con prenotazioni ottimali
    for sid in scenario_ids:
        tc, pc = _ls_cost_with_penalty(tours_train[sid], results[sid]["scenario_dist"],
                                        nodes, I, x_ls, C)
        costs_train[sid] = reserv + tc + pc
        tc_train[sid] = tc
        pc_train[sid] = pc
    UTSP_LS_train = sum(scenario_probs[sid] * costs_train[sid] for sid in scenario_ids)

    # Validazione: batch indipendenti, booking post-LS per batch
    print(f"\\n  LS out-of-sample ({N_VALIDATION_SCENARIOS} scenari, "
          f"1 batch \u00d7 {N_VALIDATION_SCENARIOS}, "
          f"seme={VALIDATION_SEED}) ...")
    batches_val = generate_scenario_batches(
        nodes, E, base_dist, I, frequent_arcs,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, VALIDATION_SEED,
        root, env, p, C,
        n_scenarios=N_VALIDATION_SCENARIOS,
        batch_size=N_VALIDATION_SCENARIOS,
    )

    costs_val, tc_val, pc_val = {}, {}, {}
    tours_val, solutions_val = {}, {}
    results_val = {}
    batch_means = []
    n = len(nodes)

    for b_idx, batch in enumerate(batches_val):
        b_sids    = batch["scenario_ids"]
        b_results = batch["results"]
        K_b       = len(b_sids)

        # Forward GNN batchato
        dist_tensors = []
        for sid in b_sids:
            sd = b_results[sid]["scenario_dist"]
            D  = torch.zeros(n, n, device=device)
            for ii, i_node in enumerate(nodes):
                for jj, j_node in enumerate(nodes):
                    if i_node != j_node:
                        D[ii, jj] = float(sd[i_node][j_node])
            dist_tensors.append(D)
        dist_stack_b = torch.stack(dist_tensors, dim=0)
        adj_stack_b  = torch.exp(-(dist_stack_b / max(dist_scale, 1e-9)) / temperature)
        xy_tile_b    = xy.unsqueeze(0).expand(K_b, -1, -1).contiguous()

        model.eval()
        with torch.no_grad():
            T_batch  = model(xy_tile_b, adj_stack_b, device)
            H_list_b = [compute_heatmap(T_batch[k:k+1]) for k in range(K_b)]

        # LS senza prenotazioni
        (c_b, tc_b, pc_b, t_b, sol_b, _) = _run_ls_on_scenarios(
            model, xy, nodes, root, I, p, C,
            b_results, b_sids, batch["scenario_probs"],
            H_list_precomputed=H_list_b,
            dist_scale=dist_scale, temperature=temperature,
            device=device, x_ls=[], label=f"val_b{b_idx}",
        )

        # Prenotazioni ottimali post-LS per questo batch
        x_batch = _compute_bookings_from_tours(t_b, b_sids, nodes, I, p, C)
        reserv_b = sum(get_edge_value(p, i, j) for (i, j) in x_batch)

        # Ricalcolo costi con prenotazioni ottimali
        batch_costs = []
        for sid in b_sids:
            tc, pc = _ls_cost_with_penalty(
                t_b[sid], b_results[sid]["scenario_dist"],
                nodes, I, x_batch, C
            )
            c_b[sid]  = reserv_b + tc + pc
            tc_b[sid] = tc
            pc_b[sid] = pc
            batch_costs.append(c_b[sid])

        mean_b = sum(batch_costs) / len(batch_costs)
        costs_val.update(c_b)
        tc_val.update(tc_b)
        pc_val.update(pc_b)
        tours_val.update(t_b)
        solutions_val.update(sol_b)
        results_val.update(b_results)
        batch_means.append(mean_b)
        print(f"    Val batch {b_idx+1}/{len(batches_val)}: "
              f"media={mean_b:.4f}  x_opt={sorted(x_batch)}")

    UTSP_LS_val = sum(batch_means) / len(batch_means)
    print(f"\n  Val globale ({N_VALIDATION_SCENARIOS} scenari, "
          f"{len(batches_val)} batch): media={UTSP_LS_val:.4f}")
    scenario_ids_val = list(costs_val.keys())

    #  Benchmark PI/STO/EEV di validazione  
    val_bench = validate_policies(
        nodes, E, base_dist, root, env, I, p, C,
        res_B["x_used_sto"], res_B["x_ev"],
        frequent_arcs, N_VALIDATION_SCENARIOS,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
        exp_name="espB"
    )
    PI_val  = val_bench["PI_val"]
    STO_val = val_bench["STO_val"]
    EEV_val = val_bench["EEV_val"]

    gap_ls_sto = (UTSP_LS_val - STO_val) / abs(STO_val) * 100 if STO_val else float("nan")
    gap_ls_eev = (UTSP_LS_val - EEV_val) / abs(EEV_val) * 100 if EEV_val else float("nan")
    gap_ls_pi  = (UTSP_LS_val - PI_val)  / abs(PI_val)  * 100 if PI_val  else float("nan")
    pi_pren_train_d = compute_pi_with_booking_costs(results, scenario_ids, I, p)
    pi_pren_train_vals = [v for v in pi_pren_train_d.values() if v is not None]
    PI_pren_train = sum(pi_pren_train_vals) / len(pi_pren_train_vals) if pi_pren_train_vals else float("nan")

    pi_pren_val_d = compute_pi_with_booking_costs(results_val, scenario_ids_val, I, p)
    pi_pren_val_vals = [v for v in pi_pren_val_d.values() if v is not None]
    PI_pren_val = sum(pi_pren_val_vals) / len(pi_pren_val_vals) if pi_pren_val_vals else float("nan")

    print("\n" + "─" * 65)
    print("RIEPILOGO ESPERIMENTO B — UTSP LOCAL SEARCH")
    print(f"  x_ls = {sorted(x_ls)}")
    print()
    print(f"  TRAINING ({len(scenario_ids)} scenari)")
    print(f"    PI      = {PI_train:.4f}")
    print(f"    PI+pren = {PI_pren_train:.4f}")
    print(f"    STO     = {STO_train:.4f}")
    print(f"    EEV     = {EEV_train:.4f}")
    print(f"    UTSP_LS = {UTSP_LS_train:.4f}")
    print()
    print(f"  VALIDAZIONE ({N_VALIDATION_SCENARIOS} scenari, seme={VALIDATION_SEED})")
    print(f"    PI_val      = {PI_val:.4f}")
    print(f"    PI+pren_val = {PI_pren_val:.4f}")
    print(f"    STO_val     = {STO_val:.4f}")
    print(f"    EEV_val     = {EEV_val:.4f}")
    print(f"    UTSP_LS_val = {UTSP_LS_val:.4f}")
    print()
    print(f"  Gap LS vs STO (val) = {gap_ls_sto:+.2f}%")
    print(f"  Gap LS vs EEV (val) = {gap_ls_eev:+.2f}%")
    print(f"  Gap LS vs PI  (val) = {gap_ls_pi:+.2f}%")
    print("─" * 65)

    _save_utsp_ls_summary(
        exp_name="espB_UTSP_LS",
        scenario_ids=scenario_ids,
        results=results,
        x_ls=x_ls,
        costs_train=costs_train,
        tc_train=tc_train,
        pc_train=pc_train,
        tours_train=tours_train,
        UTSP_LS_train=UTSP_LS_train,
        UTSP_LS_val=UTSP_LS_val,
        PI_train=PI_train, STO_train=STO_train, EEV_train=EEV_train,
        PI_val=PI_val, STO_val=STO_val, EEV_val=EEV_val,
        gap_ls_sto=gap_ls_sto, gap_ls_eev=gap_ls_eev, gap_ls_pi=gap_ls_pi,
        history=history, temperature=temperature,
    )

    genera_grafici_utsp(
        exp_name="espB_UTSP_LS",
        nodes=nodes,
        coords=coords,
        scenario_ids=scenario_ids,
        results=results,
        eev_costs=res_B["eev_costs"],
        eev_solutions=res_B["eev_solutions"],
        stoch_costs=res_B["stoch_costs"],
        stoch_solutions=res_B["stoch_solutions"],
        utsp_costs=costs_train,
        utsp_solutions=solutions_train,
        x_ev=res_B["x_ev"],
        x_sto=res_B["x_used_sto"],
        x_utsp=x_ls,
        utsp_label="UTSP local search",
        save=True,
    )
    
    plot_cost_distributions(
        eev_costs=res_B["eev_costs"],
        stoch_costs=res_B["stoch_costs"],
        utsp_costs=costs_train,
        exp_name="espB",
        split_label="train",
    )

    return {
        "x_ls":            x_ls,
        "costs_train":     costs_train,
        "tc_train":        tc_train,
        "pc_train":        pc_train,
        "tours_train":     tours_train,
        "solutions_train": solutions_train,
        "UTSP_LS_train":   UTSP_LS_train,
        "costs_val":       costs_val,
        "tours_val":       tours_val,
        "UTSP_LS_val":     UTSP_LS_val,
        "PI_val":          PI_val,
        "STO_val":         STO_val,
        "EEV_val":         EEV_val,
        "gap_ls_sto":      gap_ls_sto,
        "gap_ls_eev":      gap_ls_eev,
        "gap_ls_pi":       gap_ls_pi,
    }



def _save_utsp_summary(
    exp_name, scenario_ids, results,
    I, p, C, b,
    x_utsp, x_scores,
    utsp_costs_train, utsp_tc_train, utsp_pc_train,
    UTSP_train, PI_train, STO_train, EEV_train,
    UTSP_val, PI_val, STO_val, EEV_val,
    utsp_val, utsp_tc_val, utsp_pc_val,
    gap_utsp_sto, gap_utsp_eev, gap_utsp_pi,
    history, temperature,
):
    def fmt(x): return f"{x:.4f}" if x is not None else "N/A"

    reservation = sum(get_edge_value(p, i, j) for (i, j) in x_utsp)
    n_val = len(utsp_val)
    lines = [
        "=" * 65,
        "ESPERIMENTO B — UTSP 2-STADI",
        "=" * 65,
        "",
        "POLITICA DI PRENOTAZIONE",
        f"  Tratte prenotate : {sorted(x_utsp)}",
        f"  N prenotazioni   : {len(x_utsp)}/{len(I)}",
        f"  Costo prenotazione: {fmt(reservation)}",
        "",
    ]
    lines += [
        "",
        "TRAINING (8 scenari)",
        f"  {'Scen':>4} | {'PI':>10} | {'UTSP':>10} [perc, multa]",
    ]
    for sid in scenario_ids:
        pi = results[sid]["exact_free"]["length"]
        uc = utsp_costs_train[sid]
        lines.append(
            f"  {sid:>4} | {fmt(pi):>10} | {fmt(uc):>10} "
            f"[{fmt(utsp_tc_train[sid])}, {fmt(utsp_pc_train[sid])}]"
        )

    lines += [
        "",
        f"  PI   (train) = {fmt(PI_train)}",
        f"  STO  (train) = {fmt(STO_train)}",
        f"  EEV  (train) = {fmt(EEV_train)}",
        f"  UTSP (train) = {fmt(UTSP_train)}",
        "",
        f"VALIDAZIONE ({n_val} scenari, seme={VALIDATION_SEED})",
        f"  {'Scen':>4} | {'UTSP_val':>10} [perc, multa]",
    ]
    for sid in sorted(utsp_val.keys())[:30]:
        lines.append(
            f"  {sid:>4} | {fmt(utsp_val[sid]):>10} "
            f"[{fmt(utsp_tc_val[sid])}, {fmt(utsp_pc_val[sid])}]"
        )
    if n_val > 30:
        lines.append(f"  ... ({n_val - 30} scenari omessi)")

    lines += [
        "",
        f"  PI_val   = {fmt(PI_val)}",
        f"  STO_val  = {fmt(STO_val)}",
        f"  EEV_val  = {fmt(EEV_val)}",
        f"  UTSP_val = {fmt(UTSP_val)}",
        "",
        f"  Gap UTSP vs STO (val) = {gap_utsp_sto:+.4f}%",
        f"  Gap UTSP vs EEV (val) = {gap_utsp_eev:+.4f}%",
        f"  Gap UTSP vs PI  (val) = {gap_utsp_pi:+.4f}%",
        "",
        "TRAINING GNN",
        f"  Epoche         = {UTSP2_EPOCHS}",
        f"  Temperatura T  = {temperature:.6f}",
        f"  Loss iniziale  = {history['loss'][0]:.5f}",
        f"  Loss finale    = {history['loss'][-1]:.5f}",
        f"  Loss minima    = {min(history['loss']):.5f} (ep {int(np.argmin(history['loss'])) + 1})",
        "",
        "TRATTE I",
    ]
    for (i, j) in I:
        p_val = get_edge_value(p, i, j)
        C_val = get_edge_value(C, i, j)
        b_val = get_edge_value(b, i, j)
        lines.append(f"  {{{i},{j}}}  b={fmt(b_val)}  p={fmt(p_val)}  C={fmt(C_val)}")

    lines.append("=" * 65)
    text = "\n".join(lines)
    print("\n" + text)

    fname = os.path.join(OUTPUT_DIR, f"risultati_{exp_name}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  → Salvato: {fname}")


def _save_utsp_ls_summary(
    exp_name, scenario_ids, results,
    x_ls, costs_train, tc_train, pc_train, tours_train,
    UTSP_LS_train, UTSP_LS_val,
    PI_train, STO_train, EEV_train,
    PI_val, STO_val, EEV_val,
    gap_ls_sto, gap_ls_eev, gap_ls_pi,
    history, temperature,
):
    def fmt(x): return f"{x:.4f}" if x is not None else "N/A"

    lines = [
        "=" * 65,
        "ESPERIMENTO B — UTSP HEATMAP + LOCAL SEARCH (2-stadi, per-scenario)",
        "=" * 65,
        "",
        f"Primo stadio x_ls : {sorted(x_ls)}",
        "",
        "TEST",
        f"  {'Scen':>4} | {'PI':>10} | {'UTSP_LS':>10} [perc, multa] | tour",
    ]
    for sid in scenario_ids:
        pi = results[sid]["exact_free"]["length"]
        lines.append(
            f"  {sid:>4} | {fmt(pi):>10} | {fmt(costs_train[sid]):>10} "
            f"[{fmt(tc_train[sid])}, {fmt(pc_train[sid])}] | "
            f"{tours_train.get(sid, [])}"
        )
    lines += [
        "",
        f"  PI      (train) = {fmt(PI_train)}",
        f"  STO     (train) = {fmt(STO_train)}",
        f"  EEV     (train) = {fmt(EEV_train)}",
        f"  UTSP_LS (train) = {fmt(UTSP_LS_train)}",
        "",
        f"VALIDAZIONE ({N_VALIDATION_SCENARIOS} scenari, seme={VALIDATION_SEED})",
        f"  UTSP_LS_val = {fmt(UTSP_LS_val)}",
        f"  PI_val      = {fmt(PI_val)}",
        f"  STO_val     = {fmt(STO_val)}",
        f"  EEV_val     = {fmt(EEV_val)}",
        "",
        f"  Gap LS vs STO (val) = {gap_ls_sto:+.4f}%",
        f"  Gap LS vs EEV (val) = {gap_ls_eev:+.4f}%",
        f"  Gap LS vs PI  (val) = {gap_ls_pi:+.4f}%",
        "",
        "TRAINING GNN",
        f"  Epoche         = {UTSP2_EPOCHS}",
        f"  Temperatura T  = {temperature:.6f}",
        f"  Loss iniziale  = {history['loss'][0]:.5f}",
        f"  Loss finale    = {history['loss'][-1]:.5f}",
        f"  Loss minima    = {min(history['loss']):.5f} "
        f"(ep {int(np.argmin(history['loss'])) + 1})",
        "=" * 65,
    ]

    text = "\n".join(lines)
    print("\n" + text)
    fname = os.path.join(OUTPUT_DIR, f"risultati_{exp_name}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  → Salvato: {fname}")