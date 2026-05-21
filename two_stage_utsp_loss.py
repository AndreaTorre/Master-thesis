# -*- coding: utf-8 -*-
"""
two_stage_utsp_loss.py
======================
Loss function per il TSP stocastico a 2 stadi adattata da UTSP
(Yimeng Min et al., NeurIPS 2023).

Formulazione proposta in: droni_2_stage_formulazione_1.pdf

LOSS FINALE
-----------
L = λ₁ · row-wise
  + λ₂ · no-self-loop
  + minimize distance        (distanze perturbate per scenario)
  + booking cost             (costo prenotazione archi in I, via sigmoide)
  + penalty                  (multa per archi in I usati ma non prenotati)
  + λₑ · entropy             (regolarizzazione binaria della decisione x̃)

Compatibile con i parametri di prova_neur.py (Esperimento B):
  - I    : archi importanti non orientati canonici (i < j)
  - p    : costi di prenotazione {(i,j): p_ij}  (PRENOTAZIONE_FRAC * b_ij)
  - C    : costi multa          {(i,j): C_ij}   (PENALTY_FRAC * b_ij)
  - nodes: lista ID nodi (non necessariamente 0..n-1)
  - perturbazioni: dict {(i,j): delta_ij} per scenario

NOTAZIONE
---------
  T^ω   : output GNN dopo softmax per lo scenario ω,  shape (B, n, n)
  H^ω   : heatmap arco  H^ω = T^ω · roll(T^{ωT}, -1),  shape (B, n, n)
  x̃_ij  : decisione rilassata di prenotazione  = σ_γ( Σ_ω (H^ω_ij + H^ω_ji) )
           (aggregata su entrambe le direzioni perché I è non orientato)
  d^ω_ij: distanza perturbata = D_ij + ξ^ω_ij
"""

import math
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETRI DEFAULT  (specchio di prova_neur.py)
# ═══════════════════════════════════════════════════════════════════════════

# Pesi della loss
DEFAULT_LAMBDA1   = 20.0   # row-wise constraint  (come UTSP_C1_PENALTY)
DEFAULT_LAMBDA2   =  5.0   # no-self-loop         (come UTSP_DIAG_LOSS)
DEFAULT_LAMBDA_E  =  0.1   # entropia

# Sigmoide per la decisione di prenotazione
DEFAULT_GAMMA     = 5.0    # steepness sigmoide:
                            #   γ → ∞  = scalino duro  (prenoto se arco compare almeno una volta)
                            #   γ piccolo = curva morbida (prenoto se compare con certa freq.)


# ═══════════════════════════════════════════════════════════════════════════
# COSTRUZIONE MASCHERE / TENSORI STATICI
# (chiamate una volta sola, poi riutilizzate ad ogni forward)
# ═══════════════════════════════════════════════════════════════════════════

def canon_edge(i, j):
    """Forma canonica non orientata: coppia (min, max)."""
    return (i, j) if i < j else (j, i)


def build_I_tensors(I, nodes, p, C, device):
    """
    Costruisce le maschere e i tensori di costo per gli archi importanti I.

    Parametri
    ---------
    I      : lista di coppie canoniche (i, j) con i < j  — da prova_neur.py
    nodes  : lista ordinata degli ID nodo
    p      : dict {(i,j): p_ij}  costo prenotazione   — da prova_neur.py
    C      : dict {(i,j): C_ij}  costo multa           — da prova_neur.py
    device : torch.device

    Ritorna
    -------
    I_mask : (n, n) BoolTensor — True nelle posizioni (i,j) e (j,i) per ogni {i,j} ∈ I
    p_mat  : (n, n) FloatTensor — costi di prenotazione (0 fuori da I)
    C_mat  : (n, n) FloatTensor — costi multa           (0 fuori da I)
    idx    : dict {node_id: posizione_in_nodes}
    """
    n   = len(nodes)
    idx = {v: k for k, v in enumerate(nodes)}

    I_mask = torch.zeros(n, n, dtype=torch.bool,  device=device)
    p_mat  = torch.zeros(n, n, dtype=torch.float32, device=device)
    C_mat  = torch.zeros(n, n, dtype=torch.float32, device=device)

    for edge in I:
        i, j   = canon_edge(*edge)
        ii, jj = idx[i], idx[j]

        # Recupera il valore dal dict (supporta sia (i,j) che (j,i) come chiave)
        p_val = p.get((i, j), p.get((j, i), 0.0))
        C_val = C.get((i, j), C.get((j, i), 0.0))

        # I è non orientato: sia (ii,jj) che (jj,ii) appartengono alla prenotazione
        for a, b in [(ii, jj), (jj, ii)]:
            I_mask[a, b] = True
            p_mat[a, b]  = p_val
            C_mat[a, b]  = C_val

    return I_mask, p_mat, C_mat, idx


def build_dist_tensor(scenario_dist, nodes, device):
    """
    Converte la matrice di distanza di uno scenario (dict of dicts) in un tensore.

    Parametri
    ---------
    scenario_dist : dict {i: {j: d_ij}}  — da build_scenario_dist() in prova_neur.py
    nodes         : lista ordinata degli ID nodo
    device        : torch.device

    Ritorna
    -------
    D : (n, n) FloatTensor con D[ii, jj] = scenario_dist[nodes[ii]][nodes[jj]]
    """
    n = len(nodes)
    D = torch.zeros(n, n, dtype=torch.float32, device=device)
    for ii, i in enumerate(nodes):
        for jj, j in enumerate(nodes):
            if i != j:
                D[ii, jj] = float(scenario_dist[i][j])
    return D


def normalize_dist_tensor(D, mode="mean_positive"):
    """
    Normalizza la matrice di distanza per la loss/GNN (non cambia i costi reali).

    mode : "mean_positive"   — divide per la media delle distanze positive fuori diagonale
           "median_positive" — divide per la mediana
           "none"            — nessuna normalizzazione
    """
    if mode == "none":
        return D, 1.0

    n = D.size(-1)
    mask = ~torch.eye(n, dtype=torch.bool, device=D.device)
    if D.dim() == 3:
        mask = mask.unsqueeze(0).expand_as(D)

    vals = D[mask]
    vals = vals[vals > 0]

    if vals.numel() == 0:
        return D, 1.0

    scale = float(vals.mean()) if mode == "mean_positive" else float(torch.median(vals))
    scale = max(scale, 1e-9)
    return D / scale, scale


# ═══════════════════════════════════════════════════════════════════════════
# CALCOLO HEATMAP  H^ω  DA  T^ω
# (stessa formula di UTSP paper e unionev3.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_heatmap(T_omega):
    """
    H^ω = T^ω · roll(T^{ωT}, -1)

    Parametri
    ---------
    T_omega : (B, n, n) — output GNN dopo softmax per lo scenario ω

    Ritorna
    -------
    H_omega : (B, n, n)
    """
    return torch.bmm(T_omega, torch.roll(T_omega.transpose(1, 2), -1, 1))


# ═══════════════════════════════════════════════════════════════════════════
# DECISIONE RILASSATA DI PRENOTAZIONE
# ═══════════════════════════════════════════════════════════════════════════

def compute_booking_decision(H_list, I_mask, scenario_probs, gamma):
    """
    x̃_ij = σ_γ( Σ_ω p_ω · (H^ω_ij + H^ω_ji) )   ∀ {i,j} ∈ I

    La somma delle due direzioni riflette la struttura non orientata di I
    (un arco viene prenotato se è usato in qualunque direzione).

    Parametri
    ---------
    H_list         : lista di K tensori (B, n, n) — heatmap per ogni scenario
    I_mask         : (n, n) BoolTensor — True nelle celle di I
    scenario_probs : (K,) FloatTensor  — p_ω per ogni scenario
    gamma          : float — steepness della sigmoide

    Ritorna
    -------
    x_tilde : (B, n, n) FloatTensor — decisione in [0,1], nonzero solo su I (entrambe direzioni)
              Il valore è lo stesso in (i,j) e (j,i) perché I è non orientato.
    H_sum   : (B, n, n) FloatTensor — Σ_ω p_ω (H^ω_ij + H^ω_ji), usato nella penalty
    """
    B, n, _ = H_list[0].shape
    device  = H_list[0].device

    # Somma pesata delle heatmap direzionali su tutti gli scenari
    H_weighted_sum = torch.zeros(B, n, n, device=device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        H_weighted_sum += p_w * H_omega          # accumula Σ_ω p_ω H^ω_ij

    # Per ogni edge {i,j} ∈ I aggiungo entrambe le direzioni (arco non orientato)
    H_sym_sum = H_weighted_sum + H_weighted_sum.transpose(1, 2)
    # → H_sym_sum[b, ii, jj] = Σ_ω p_ω (H^ω_ij + H^ω_ji)

    # Sigmoide: σ_γ(x) = 1 / (1 + exp(-γ·x))
    arg       = H_sym_sum                        # shape (B, n, n)
    x_tilde   = torch.sigmoid(gamma * arg)       # ∈ (0, 1)

    # Azzero fuori da I_mask (non serve la prenotazione per altri archi)
    I_float = I_mask.float().unsqueeze(0)        # (1, n, n)
    x_tilde = x_tilde * I_float
    H_sum   = H_sym_sum * I_float

    return x_tilde, H_sum


# ═══════════════════════════════════════════════════════════════════════════
# LOSS COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def _loss_row_wise(T_list, scenario_probs):
    """
    λ₁ · Σ_ω p_ω Σᵢ (Σⱼ T^ω_ij - 1)²

    Penalizza le righe di T che non sommano a 1 (vincolo di assegnamento per riga).
    Identico al termine C1 di UTSP, qui pesato sugli scenari.
    """
    loss = torch.tensor(0.0, device=T_list[0].device)
    for T_omega, p_w in zip(T_list, scenario_probs):
        row_sums = T_omega.sum(dim=2)        # (B, n)
        loss     = loss + p_w * ((row_sums - 1.0) ** 2).sum(dim=1).mean()
    return loss


def _loss_self_loop(H_list, scenario_probs):
    """
    λ₂ · Σ_ω p_ω Σᵢ H^ω_ii

    Penalizza la diagonale della heatmap (auto-loop impossibili nel TSP).
    Identico al termine DIAG_LOSS di UTSP, qui pesato sugli scenari.
    """
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        diag = torch.diagonal(H_omega, dim1=1, dim2=2).sum(dim=1)  # (B,)
        loss = loss + p_w * diag.mean()
    return loss


def _loss_distance(H_list, dist_list, scenario_probs):
    """
    Σ_ω p_ω Σ_{(i,j)∈E} d^ω_ij · H^ω_ij

    Minimizza il costo atteso del tour usando le distanze perturbate.
    dist_list contiene le distanze GIÀ normalizzate per la loss.
    """
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, D_omega, p_w in zip(H_list, dist_list, scenario_probs):
        # D_omega : (n, n) oppure (B, n, n)
        if D_omega.dim() == 2:
            D_omega = D_omega.unsqueeze(0)   # → (1, n, n)
        cost = (H_omega * D_omega).sum(dim=(1, 2))   # (B,)
        loss = loss + p_w * cost.mean()
    return loss


def _loss_booking(x_tilde, p_mat):
    """
    Σ_{(i,j)∈I} c_ij · x̃_ij

    Costo atteso di prenotazione degli archi importanti.
    c_ij = p_ij in prova_neur.py (= PRENOTAZIONE_FRAC * b_ij).

    Poiché x_tilde e p_mat sono già simmetrici su I e nulli fuori da I,
    divido per 2 per non contare ogni arco due volte (sia (i,j) che (j,i)).
    """
    # p_mat : (n, n),  x_tilde : (B, n, n)
    cost = (p_mat.unsqueeze(0) * x_tilde).sum(dim=(1, 2))  # (B,)
    return (cost / 2.0).mean()


def _loss_penalty(H_list, x_tilde, C_mat, I_mask, scenario_probs):
    """
    Σ_ω p_ω Σ_{(i,j)∈I} C_ij · H^ω_ij · (1 - x̃_ij)

    Multa per gli archi in I che vengono usati (H^ω_ij > 0) ma non prenotati
    (1 - x̃_ij ≈ 1).

    La moltiplicazione H^ω_ij · (1 - x̃_ij) è differenziabile: quando x̃_ij → 1
    il termine si annulla, quando x̃_ij → 0 pago tutta la multa.
    """
    loss    = torch.tensor(0.0, device=H_list[0].device)
    C_broad = C_mat.unsqueeze(0)      # (1, n, n)
    I_float = I_mask.float().unsqueeze(0)

    for H_omega, p_w in zip(H_list, scenario_probs):
        # Restringo a I (già zero fuori da I_mask grazie a I_float)
        penalty = C_broad * H_omega * (1.0 - x_tilde) * I_float  # (B, n, n)
        loss    = loss + p_w * penalty.sum(dim=(1, 2)).mean()

    return loss / 2.0   # ÷ 2 per doppio conteggio direzioni (I non orientato)


def _loss_entropy(x_tilde, I_mask):
    """
    λₑ · Σ_{(i,j)∈I} H(x̃_ij)
       = λₑ · Σ_{(i,j)∈I} [-x̃_ij log(x̃_ij) - (1-x̃_ij) log(1-x̃_ij)]

    Regolarizzazione: spinge x̃_ij verso 0 o 1 (decisioni nette).
    Un'entropia alta indica incertezza nella decisione di prenotazione.

    Nota: x̃_ij = x̃_ji per costruzione (I non orientato),
    per cui divido per 2 per non contare ogni arco due volte.
    """
    eps  = 1e-9
    x    = x_tilde.clamp(eps, 1.0 - eps)
    ent  = -x * torch.log(x) - (1.0 - x) * torch.log(1.0 - x)   # (B, n, n)

    I_float = I_mask.float().unsqueeze(0)
    ent_I   = (ent * I_float).sum(dim=(1, 2))   # (B,)
    return (ent_I / 2.0).mean()


# ═══════════════════════════════════════════════════════════════════════════
# LOSS PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════

def two_stage_utsp_loss(
    T_list,
    dist_list,
    I_mask,
    p_mat,
    C_mat,
    scenario_probs,
    gamma     = DEFAULT_GAMMA,
    lambda1   = DEFAULT_LAMBDA1,
    lambda2   = DEFAULT_LAMBDA2,
    lambda_e  = DEFAULT_LAMBDA_E,
    return_components = False,
):
    """
    Loss completa per il TSP stocastico 2-stadi (formulazione droni_2_stage).

    Parametri
    ---------
    T_list         : lista di K tensori (B, n, n) — output GNN softmax per scenario ω
    dist_list      : lista di K tensori (n, n) o (B, n, n) — distanze normalizzate per loss
    I_mask         : (n, n) BoolTensor  — True nelle celle (i,j) e (j,i) per {i,j}∈I
    p_mat          : (n, n) FloatTensor — costi prenotazione (da build_I_tensors)
    C_mat          : (n, n) FloatTensor — costi multa        (da build_I_tensors)
    scenario_probs : (K,) FloatTensor   — p_ω per ogni scenario
    gamma          : float — steepness sigmoide
    lambda1        : float — peso row-wise
    lambda2        : float — peso no-self-loop
    lambda_e       : float — peso entropia
    return_components : bool — se True ritorna anche il dizionario dei singoli termini

    Ritorna
    -------
    loss : scalar FloatTensor — perdita totale
    (opzionale) components : dict con i singoli termini della loss

    Note implementative
    -------------------
    - dist_list deve contenere le distanze GIÀ normalizzate (via normalize_dist_tensor).
      I costi reali per valutazione out-of-sample non vengono toccati.
    - scenario_probs deve essere un tensore PyTorch sulla stessa device di T_list[0].
    - x̃_ij aggrega entrambe le direzioni perché I è non orientato (come in prova_neur.py).
    """
    # ── Heatmap H^ω da T^ω ────────────────────────────────────────────
    H_list = [compute_heatmap(T_omega) for T_omega in T_list]

    # ── Decisione rilassata di prenotazione x̃ ─────────────────────────
    x_tilde, H_sym_sum = compute_booking_decision(
        H_list, I_mask, scenario_probs, gamma
    )

    # ── Singoli termini della loss ──────────────────────────────────────
    L_row    = _loss_row_wise(T_list,                   scenario_probs)
    L_diag   = _loss_self_loop(H_list,                  scenario_probs)
    L_dist   = _loss_distance(H_list,   dist_list,      scenario_probs)
    L_book   = _loss_booking(x_tilde,   p_mat)
    L_pen    = _loss_penalty(H_list, x_tilde, C_mat, I_mask, scenario_probs)
    L_ent    = _loss_entropy(x_tilde,   I_mask)

    # ── Loss totale ─────────────────────────────────────────────────────
    loss = (
        lambda1  * L_row
      + lambda2  * L_diag
      + L_dist
      + L_book
      + L_pen
      + lambda_e * L_ent
    )

    if return_components:
        return loss, {
            "row_wise"  : L_row.item(),
            "self_loop" : L_diag.item(),
            "distance"  : L_dist.item(),
            "booking"   : L_book.item(),
            "penalty"   : L_pen.item(),
            "entropy"   : L_ent.item(),
            "total"     : loss.item(),
        }

    return loss


# ═══════════════════════════════════════════════════════════════════════════
# DECODIFICA: da x̃ a x ∈ {0,1}
# (usata dopo il training per estrarre la politica di primo stadio)
# ═══════════════════════════════════════════════════════════════════════════

def decode_booking_policy(H_list, I, nodes, I_mask, scenario_probs, gamma, threshold=0.5):
    """
    Decodifica la decisione di prenotazione binaria dal modello addestrato.

    Calcola x̃_ij per ogni {i,j} ∈ I e applica una soglia.

    Parametri
    ---------
    H_list         : lista di K tensori (1, n, n) — un campione, non un batch
    I              : lista di coppie canoniche (i,j) ∈ I  — da prova_neur.py
    nodes          : lista ordinata degli ID nodo
    I_mask         : (n, n) BoolTensor
    scenario_probs : (K,) FloatTensor
    gamma          : float
    threshold      : float — soglia per la decisione binaria (default 0.5)

    Ritorna
    -------
    x_reserved : lista di coppie (i,j) ∈ I per cui x̃_ij ≥ threshold
    x_scores   : dict {(i,j): valore x̃_ij}  — scores continui per analisi
    """
    idx = {v: k for k, v in enumerate(nodes)}

    with torch.no_grad():
        x_tilde, _ = compute_booking_decision(H_list, I_mask, scenario_probs, gamma)
        # x_tilde : (1, n, n)

    x_reserved = []
    x_scores   = {}

    for edge in I:
        i, j   = canon_edge(*edge)
        ii, jj = idx[i], idx[j]
        # Prendo il valore in una direzione (simmetrico per costruzione)
        score  = float(x_tilde[0, ii, jj].item())
        x_scores[(i, j)] = score
        if score >= threshold:
            x_reserved.append((i, j))

    return x_reserved, x_scores


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTICA DELLA LOSS DURANTE IL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def format_loss_components(components, epoch=None):
    """
    Stampa leggibile dei componenti della loss.

    Esempio output:
      Ep  42 | total=4.3210 | dist=3.1200 | book=0.4500 | pen=0.3100 |
              | row=0.3000 | diag=0.0200 | ent=0.1210
    """
    prefix = f"Ep {epoch:>4}" if epoch is not None else "Loss"
    return (
        f"{prefix} | total={components['total']:.4f} "
        f"| dist={components['distance']:.4f} "
        f"| book={components['booking']:.4f} "
        f"| pen={components['penalty']:.4f} "
        f"| row={components['row_wise']:.4f} "
        f"| diag={components['self_loop']:.4f} "
        f"| ent={components['entropy']:.4f}"
    )


def check_booking_coverage(x_scores, I, threshold=0.5):
    """
    Diagnostica post-training: per ogni arco in I stampa lo score e la decisione.
    Utile per verificare se la loss sta producendo decisioni nette.
    """
    lines = ["Decisioni di prenotazione NN:"]
    n_booked = 0
    for edge in sorted(I):
        i, j  = canon_edge(*edge)
        score = x_scores.get((i, j), 0.0)
        flag  = "✓ PRENOTA" if score >= threshold else "✗ non prenota"
        lines.append(f"  {{{i},{j}}}  x̃={score:.4f}  {flag}")
        if score >= threshold:
            n_booked += 1
    lines.append(f"  Totale prenotazioni: {n_booked}/{len(I)}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST (python -m two_stage_utsp_loss  oppure  python two_stage_utsp_loss.py)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test minimale con valori sintetici.
    Verifica:
      - shape dei tensori
      - differenziabilità della loss
      - stampa dei componenti
    """
    import torch

    torch.manual_seed(42)
    device = torch.device("cpu")

    # Parametri sintetici
    n        = 6        # nodi
    K        = 4        # scenari
    B        = 8        # batch size
    n_I      = 3        # archi in I

    # Nodi e archi I fittizi
    nodes    = list(range(n))
    I_fake   = [(0, 2), (1, 3), (2, 4)]   # coppie canoniche

    # Tensori di costo fittizi per I
    p_dict = {(i, j): 0.25 * (i + j + 1) for (i, j) in I_fake}
    C_dict = {(i, j): 0.50 * (i + j + 1) for (i, j) in I_fake}

    I_mask, p_mat, C_mat, idx = build_I_tensors(I_fake, nodes, p_dict, C_dict, device)

    # Output GNN fittizi (softmax per riga, come UTSP_GNN.forward)
    T_list = []
    for _ in range(K):
        logits  = torch.randn(B, n, n, requires_grad=True)
        T_omega = torch.softmax(logits, dim=-1)
        T_list.append(T_omega)

    # Matrici di distanza normalizzate fittizie
    dist_list = []
    for _ in range(K):
        D = torch.rand(n, n) + 0.1
        D.fill_diagonal_(0.0)
        dist_list.append(D)

    # Probabilità scenari uniformi
    probs = torch.full((K,), 1.0 / K, device=device)

    # ── Calcolo loss con componenti ──────────────────────────────────
    loss, comps = two_stage_utsp_loss(
        T_list, dist_list, I_mask, p_mat, C_mat, probs,
        gamma=5.0, lambda1=20.0, lambda2=5.0, lambda_e=0.1,
        return_components=True
    )

    print("=== TEST two_stage_utsp_loss ===")
    print(format_loss_components(comps, epoch=0))
    print(f"\nLoss differenziabile: {loss.requires_grad}")

    # ── Backprop ─────────────────────────────────────────────────────
    loss.backward()
    grad_ok = all(
        T.grad is not None and not torch.isnan(T.grad).any()
        for T in T_list
        if T.requires_grad
    )
    print(f"Gradients OK:        {grad_ok}")

    # ── Decodifica politica ──────────────────────────────────────────
    H_list_single = [compute_heatmap(T_list[k][:1].detach()) for k in range(K)]
    x_res, x_sc   = decode_booking_policy(
        H_list_single, I_fake, nodes, I_mask, probs, gamma=5.0, threshold=0.5
    )
    print(f"\n{check_booking_coverage(x_sc, I_fake)}")
    print("\n=== TEST COMPLETATO ===")