# -*- coding: utf-8 -*-
import os
import time
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
    UTSP2_LAMBDA_E, UTSP2_GAMMA, UTSP2_TEMP_MODE, UTSP2_TEMP_SCALE,
    UTSP2_TEMP_FIXED, UTSP2_DIST_SCALE_MODE, UTSP2_THRESHOLD,
)
from tsp_utils import get_edge_value
from gurobi_models import solve_reservation_tsp
from scenarios import generate_scenarios
from evaluation import validate_policies
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

def _scattering_diffusion(W, feature):
    deg, D = torch.sum(W, 2, keepdim=True).clamp(min=1e-9), None
    D   = torch.pow(deg, -1)
    buf, x = [], feature
    for i in range(16):
        x = 0.5 * x + 0.5 * torch.bmm(W, D * x)
        if i in [0, 1, 3, 7]:
            buf.append(x)
    return buf[0]-buf[1], buf[1]-buf[2], buf[2]-buf[3], buf[3]-feature*0

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
        h_prime  = (attn * torch.stack(parts, dim=1)).mean(dim=1)
        return _leaky(self.linear2(_leaky(self.linear1(h_prime))))

class UTSP_GNN(nn.Module):
    """GNN UTSP — identica a unionev3.py."""
    def __init__(self, n_nodes, hidden_dim, n_layers):
        super().__init__()
        self.bn0     = nn.BatchNorm1d(2)
        self.in_proj = nn.Linear(2, hidden_dim)
        self.convs   = nn.ModuleList([_SCTConv(hidden_dim) for _ in range(n_layers)])
        self.mlp1    = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim)
        self.mlp2    = nn.Linear(hidden_dim, n_nodes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xy, adj, device):
      B, N, _ = xy.shape
      x       = self.bn0(xy.reshape(B * N, 2)).reshape(B, N, 2)
      x       = _leaky(self.in_proj(x))
      hidden  = x
      for conv in self.convs:
          x = conv(x, adj, device)
          hidden = torch.cat([hidden, x], dim=-1)
      return self.softmax(self.mlp2(_leaky(self.mlp1(hidden))))

def _normalize_coords(nodes, coords, device):
    """Normalizza le coordinate in [0,1] come in unionev3.py."""
    xs = np.array([coords[v][0] for v in nodes], dtype=np.float32)
    ys = np.array([coords[v][1] for v in nodes], dtype=np.float32)
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)
    xy = torch.from_numpy(np.stack([xs, ys], axis=1)).float().to(device)
    return xy   # (n, 2)

def _compute_temperature(dist_stack, mode, scale, fixed):
    """
    Calcola la temperatura T per adj = exp(-d/T).
    dist_stack : (K, n, n) tensor — distanze GIÀ normalizzate
    """
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

def _build_input_tensors(scenario_ids, results, nodes, coords, device):
    """
    Costruisce xy e le matrici di distanza normalizzate per tutti gli scenari.

    Ritorna
    -------
    xy_tile    : (K, n, 2)  — coordinate normalizzate, replicate per ogni scenario
    dist_raw   : list di K tensori (n, n) — distanze reali (non normalizzate)
    dist_model : (K, n, n)  — distanze normalizzate per GNN/loss
    dist_scale : float      — scala usata per la normalizzazione
    temperature: float      — temperatura T per il kernel gaussiano
    """
    K = len(scenario_ids)
    n = len(nodes)
    xy = _normalize_coords(nodes, coords, device)         # (n, 2)
    xy_tile = xy.unsqueeze(0).expand(K, -1, -1).contiguous()           # (K, n, 2)

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

    dist_stack = torch.stack(dist_raw, dim=0)              # (K, n, n)

    # Normalizzazione interna UTSP
    dist_model, dist_scale = normalize_dist_tensor(
        dist_stack, mode=UTSP2_DIST_SCALE_MODE
    )

    temperature = _compute_temperature(
        dist_model, UTSP2_TEMP_MODE, UTSP2_TEMP_SCALE, UTSP2_TEMP_FIXED
    )

    return xy_tile, dist_raw, dist_model, dist_scale, temperature

def _train_utsp_2stage(
    nodes, coords, scenario_ids, results, scenario_probs,
    I_mask, p_mat, C_mat, device
):
    """
    Addestra UTSP_GNN con la loss 2-stage su tutti gli scenari di training.

    Strategia di training:
    - Tutti i K scenari vengono processati in un unico forward pass per epoca
      (batch size = K).  Con K=8 è trattabile e mantiene la correlazione
      tra scenari che la loss sfrutta per la decisione di prenotazione.
    - Non serve DataLoader: gli scenari sono fissi e pochi.
    """
    K = len(scenario_ids)
    n = len(nodes)

    xy_tile, dist_raw, dist_model, dist_scale, temperature = _build_input_tensors(
        scenario_ids, results, nodes, coords, device
    )

    probs_t = torch.tensor(
        [scenario_probs[sid] for sid in scenario_ids],
        dtype=torch.float32, device=device
    )

    # Diagonale azzerata — stessa maschera usata in unionev3.py
    off_diag = (1 - torch.eye(n, device=device)).unsqueeze(0)  # (1, n, n)

    model = UTSP_GNN(n, UTSP2_HIDDEN, UTSP2_NLAYERS).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(), lr=UTSP2_LR)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=UTSP2_STEP_LR, gamma=0.8
    )

    print(f"\n  Training UTSP 2-stage | device={device} | n_params={n_par:,}")
    print(f"  GNN({n}→{UTSP2_HIDDEN}×{UTSP2_NLAYERS}) | "
          f"K={K} scenari | T={temperature:.4f} | scale={dist_scale:.4f}")
    print(f"  Epoche={UTSP2_EPOCHS}  lr={UTSP2_LR}  "
          f"λ1={UTSP2_LAMBDA1}  λ2={UTSP2_LAMBDA2}  λe={UTSP2_LAMBDA_E}  γ={UTSP2_GAMMA}")

    history = {"loss": [], "components": []}
    best_loss, best_state = float("inf"), None
    t0 = time.time()

    # Adiacenza per ogni scenario: adj_k = exp(-D^ω_norm / T) con diagonale zero
    adj_stack = torch.exp(-dist_model / temperature) * off_diag  # (K, n, n)

    for epoch in range(1, UTSP2_EPOCHS + 1):
        model.train()

        # ── Forward: un unico batch con tutti i K scenari ──────────────
        T_batch = model(xy_tile, adj_stack, device)   # (K, n, n)

        # Separa in lista di K tensori (1, n, n) per la loss
        T_list    = [T_batch[k:k+1] for k in range(K)]
        dist_list = [dist_model[k:k+1] for k in range(K)]   # (1, n, n) norm

        loss, comps = two_stage_utsp_loss(
            T_list, dist_list, I_mask, p_mat, C_mat, probs_t,
            gamma=UTSP2_GAMMA,
            lambda1=UTSP2_LAMBDA1,
            lambda2=UTSP2_LAMBDA2,
            lambda_e=UTSP2_LAMBDA_E,
            return_components=True,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["loss"].append(float(loss.item()))
        history["components"].append(comps)

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % UTSP2_LOG_FREQ == 0 or epoch == 1:
            marker = " ★" if abs(loss.item() - best_loss) < 1e-9 else ""
            print(f"  {format_loss_components(comps, epoch)}{marker}")

    elapsed = time.time() - t0
    print(f"\n  Training completato in {elapsed:.1f}s | "
          f"miglior loss = {best_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    return model, history, adj_stack, dist_model, xy_tile, probs_t, temperature

def _decode_policy(model, adj_stack, xy_tile, I, nodes, I_mask, probs_t, device):
    """
    Calcola x̃_ij per ogni {i,j}∈I dal modello addestrato e applica la soglia.

    Usa tutti i K scenari (lo stesso batch del training) per la stima di x̃.
    """
    K = adj_stack.size(0)
    model.eval()
    with torch.no_grad():
        T_batch = model(xy_tile, adj_stack, device)        # (K, n, n)

    H_list = [compute_heatmap(T_batch[k:k+1]) for k in range(K)]

    x_reserved, x_scores = decode_booking_policy(
        H_list, I, nodes, I_mask, probs_t,
        gamma=UTSP2_GAMMA, threshold=UTSP2_THRESHOLD
    )
    return x_reserved, x_scores, H_list, T_batch

def _evaluate_policy_oos(
    policy_name, x_policy, nodes, E, base_dist, root, env,
    I, p, C, frequent_arcs, n_val, n_extra, mean_frac, sigma_frac
):
    """
    Valuta una politica di prenotazione fissa su n_val scenari out-of-sample.

    Genera gli scenari con VALIDATION_SEED (identico a validate_policies
    di prova_neur.py) così i numeri sono direttamente comparabili.

    Ritorna
    -------
    costs   : dict {sid: costo_totale}
    tc_dict : dict {sid: tour_cost}
    pc_dict : dict {sid: penalty_paid}
    mean    : float — costo medio
    """
    scenario_ids_val = list(range(1, n_val + 1))
    reservation = sum(get_edge_value(p, i, j) for (i, j) in x_policy)

    results_val, _, _ = generate_scenarios(
        scenario_ids_val, nodes, E, base_dist, I, frequent_arcs,
        n_extra, mean_frac, sigma_frac, VALIDATION_SEED,
        root=root, env=env, p=p, C=C
    )

    costs, tc_dict, pc_dict = {}, {}, {}
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

    mean = sum(costs.values()) / len(costs)
    return costs, tc_dict, pc_dict, mean, results_val

def run_esperimento_B_UTSP(nodes, coords, base_dist, E, root, env, res_B):
    """
    Esperimento B — UTSP 2-stadi.

    Parametri
    ---------
    nodes, coords, base_dist, E, root, env : output di load_data() / load_env()
    res_B : dict restituito da run_esperimento_B  (deve contenere I, p, C,
            results, scenario_probs, frequent_arcs, PI, STO, EEV, ...)
    """
    print("\n" + "=" * 70)
    print("ESPERIMENTO B — UTSP 2-STADI")

    # ── Controllo chiavi necessarie ────────────────────────────────────
    needed = ["I", "p", "C", "b", "results", "scenario_probs",
              "frequent_arcs", "PI", "STO", "EEV",
              "stoch_costs", "eev_costs", "x_ev", "x_used_sto"]
    missing = [k for k in needed if k not in res_B]
    if missing:
        raise KeyError(f"res_B manca delle chiavi: {missing}")

    # ── Recupero dati da res_B  ────────────────────────────────────────
    I              = res_B["I"]
    p              = res_B["p"]
    C              = res_B["C"]
    b              = res_B["b"]
    results        = res_B["results"]          # 8 scenari di training
    scenario_probs = res_B["scenario_probs"]
    scenario_ids   = list(results.keys())
    frequent_arcs  = res_B["frequent_arcs"]
    PI_train       = res_B["PI"]
    STO_train      = res_B["STO"]
    EEV_train      = res_B["EEV"]

    n = len(nodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Maschere e costi per la loss ───────────────────────────────────
    I_mask, p_mat, C_mat, node_idx = build_I_tensors(I, nodes, p, C, device)

    print(f"\n  Setup: |I|={len(I)}  n_nodi={n}  "
          f"K_train={len(scenario_ids)}  device={device}")
    print(f"  I = {I}")

    # ── Training ───────────────────────────────────────────────────────
    (model, history, adj_stack, dist_model,
     xy_tile, probs_t, temperature) = _train_utsp_2stage(
        nodes, coords, scenario_ids, results, scenario_probs,
        I_mask, p_mat, C_mat, device
    )

    # ── Decodifica politica x_utsp ─────────────────────────────────────
    x_utsp, x_scores, H_list, T_batch = _decode_policy(
        model, adj_stack, xy_tile, I, nodes, I_mask, probs_t, device
    )

    reservation_utsp = sum(get_edge_value(p, i, j) for (i, j) in x_utsp)

    print(f"\n{check_booking_coverage(x_scores, I, UTSP2_THRESHOLD)}")
    print(f"\n  Costo prenotazione UTSP : {reservation_utsp:.4f}")

    # ── Valutazione training (same 8 scenari, per confronto diretto) ──
    utsp_costs_train, utsp_tc_train, utsp_pc_train = {}, {}, {}
    for sid in scenario_ids:
        sd  = results[sid]["scenario_dist"]
        sol = solve_reservation_tsp(
            nodes, E, I, sd, root, p, C, env,
            fixed_reservations=x_utsp, output_flag=0
        )
        tc  = sol["tour_cost"]    or 0.0
        pc  = sol["penalty_paid"] or 0.0
        utsp_costs_train[sid]  = reservation_utsp + tc + pc
        utsp_tc_train[sid]     = tc
        utsp_pc_train[sid]     = pc

    UTSP_train = sum(
        scenario_probs[sid] * utsp_costs_train[sid] for sid in scenario_ids
    )

    # ── Validazione out-of-sample (30 scenari, VALIDATION_SEED=99) ────
    print(f"\n  Validazione out-of-sample ({N_VALIDATION_SCENARIOS} scenari, "
          f"seme={VALIDATION_SEED}) ...")

    utsp_val, utsp_tc_val, utsp_pc_val, UTSP_val, results_val = \
        _evaluate_policy_oos(
            "utsp", x_utsp, nodes, E, base_dist, root, env,
            I, p, C, frequent_arcs,
            N_VALIDATION_SCENARIOS, N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC
        )

    # PI, STO, EEV sugli stessi 30 scenari di validazione
    # (chiamiamo validate_policies che li rigenera con lo stesso seme)
    val_results = validate_policies(
        nodes, E, base_dist, root, env, I, p, C,
        res_B["x_used_sto"], res_B["x_ev"],
        frequent_arcs, N_VALIDATION_SCENARIOS,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
        exp_name="espB"     # appende al txt esistente di B
    )
    PI_val  = val_results["PI_val"]
    STO_val = val_results["STO_val"]
    EEV_val = val_results["EEV_val"]

    # ── Riepilogo numerico ─────────────────────────────────────────────
    gap_utsp_sto = (UTSP_val - STO_val) / abs(STO_val) * 100 if STO_val else float("nan")
    gap_utsp_eev = (UTSP_val - EEV_val) / abs(EEV_val) * 100 if EEV_val else float("nan")
    gap_utsp_pi  = (UTSP_val - PI_val)  / abs(PI_val)  * 100 if PI_val  else float("nan")

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

    # ── Salva txt ──────────────────────────────────────────────────────
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
        "model":              model,
        "history":            history,
        "x_utsp":             x_utsp,
        "x_scores":           x_scores,
        "I_mask":             I_mask,
        "utsp_costs_train":   utsp_costs_train,
        "UTSP_train":         UTSP_train,
        "utsp_val":           utsp_val,
        "UTSP_val":           UTSP_val,
        "PI_val":             PI_val,
        "STO_val":            STO_val,
        "EEV_val":            EEV_val,
        "gap_utsp_sto":       gap_utsp_sto,
        "gap_utsp_eev":       gap_utsp_eev,
        "gap_utsp_pi":        gap_utsp_pi,
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
        "SCORE CONTINUI x̃_ij  (soglia={:.2f})".format(UTSP2_THRESHOLD),
    ]
    for edge, score in sorted(x_scores.items()):
        flag = "✓" if score >= UTSP2_THRESHOLD else "✗"
        lines.append(f"  {{{edge[0]},{edge[1]}}}  x̃={score:.4f}  {flag}")

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
        f"  Loss minima    = {min(history['loss']):.5f} "
        f"(ep {int(np.argmin(history['loss'])) + 1})",
        "",
        "TRATTE I",
    ]
    for (i, j) in I:
        p_val = get_edge_value(p, i, j)
        C_val = get_edge_value(C, i, j)
        b_val = get_edge_value(b, i, j)
        lines.append(
            f"  {{{i},{j}}}  b={fmt(b_val)}  p={fmt(p_val)}  C={fmt(C_val)}"
        )

    lines.append("=" * 65)
    text = "\n".join(lines)
    print("\n" + text)

    fname = os.path.join(OUTPUT_DIR, f"risultati_{exp_name}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  → Salvato: {fname}")
