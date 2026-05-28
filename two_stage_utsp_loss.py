# -*- coding: utf-8 -*-
"""
two_stage_utsp_loss.py
======================
Loss function per il TSP stocastico a 2 stadi adattata da UTSP
(Yimeng Min et al., NeurIPS 2023).

LOSS FINALE
-----------
L = λ₁ · row-wise constraint      (T^ω sommano a 1 per riga)
  + λ₂ · no-self-loop             (diagonale di H^ω vicina a 0)
  + minimize distance              (costo atteso tour pesato su scenari)
  + booking cost                   (costo prenotazione via esponenziale: c_ij·(1−e^{−α·H̄_ij}))
  + consistency                    (λ_e · varianza di H^ω_ij rispetto a H̄_ij)
  + asymmetry                      (λ_d · penalizza archi simmetrici in E)
  [+ penalty]                      (opzionale: multa per archi in I usati ma non prenotati)

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
  H̄_ij  : heatmap aggregata  H̄ = Σ_ω p_ω H^ω
  d^ω_ij: distanza perturbata = D_ij + ξ^ω_ij
  α     : tasso di saturazione del booking cost (analogo a una curva di domanda)
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
DEFAULT_LAMBDA_E  =  0.1   # consistency          (varianza H^ω rispetto a H̄)
DEFAULT_LAMBDA_D  =  0.1   # asymmetry            (penalizza archi simmetrici)

# Tasso di saturazione per il booking cost:
#   α → 0  : booking cost ≈ α · c_ij · H̄_ij  (quasi lineare)
#   α grande: satura rapidamente a c_ij        (scalino duro)
DEFAULT_ALPHA     =  5.0


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

def compute_H_bar(H_list, scenario_probs):
    """
    H_bar_ij = Σ_{ω∈Ω} p_ω H^ω_ij

    Parametri
    ---------
    H_list         : lista di K tensori (B, n, n)
    scenario_probs : tensore (K,) con probabilità p_ω

    Ritorna
    -------
    H_bar : (B, n, n)
        Heatmap aggregata pesata sugli scenari.
    """
    H_bar = torch.zeros_like(H_list[0])

    for H_omega, p_w in zip(H_list, scenario_probs):
        H_bar = H_bar + p_w * H_omega

    return H_bar
# ═══════════════════════════════════════════════════════════════════════════
# DECISIONE RILASSATA DI PRENOTAZIONE
# ═══════════════════════════════════════════════════════════════════════════

def compute_booking_activation(H_bar, I_mask, alpha):
    """
    Attivazione continua di prenotazione via esponenziale:

        b̃_ij = 1 − e^{−α · H̄_ij}

    dove H̄_ij = Σ_ω p_ω H^ω_ij è la heatmap aggregata sugli scenari.

    Questa funzione sostituisce la sigmoide della formulazione precedente:
    - è monotona crescente in H̄_ij
    - vale 0 quando H̄_ij = 0  (arco mai usato → non si prenota)
    - tende a 1 quando H̄_ij → ∞  (arco molto usato → si prenota con certezza)
    - α controlla la velocità di saturazione

    Nota: poiché I è non orientato, si lavora già su H̄ che è già la media
    scenario-pesata; non è necessaria la doppia somma (i,j)+(j,i) esplicita
    perché I_mask copre entrambe le direzioni.

    Parametri
    ---------
    H_bar  : (B, n, n) — heatmap aggregata pesata sugli scenari
    I_mask : (n, n) BoolTensor — True nelle celle (i,j) e (j,i) per {i,j}∈I
    alpha  : float — tasso di saturazione

    Ritorna
    -------
    b_tilde : (B, n, n) — attivazione di prenotazione (0 fuori da I)
    """
    I_float = I_mask.float().unsqueeze(0)          # (1, n, n)
    b_tilde = (1.0 - torch.exp(-alpha * H_bar)) * I_float
    return b_tilde

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


def _loss_booking(H_list, p_mat, I_mask, alpha):
    H_stack    = torch.stack([H.squeeze(0) for H in H_list], dim=0)  # (K, n, n)
    H_sum      = H_stack.sum(dim=0)                                   # (n, n)
    I_float    = I_mask.float()
    activation = (1.0 - torch.exp(-alpha * H_sum)) * I_float
    cost       = (p_mat * activation).sum()
    return cost/2.0

def _loss_consistency(H_list, H_bar, I_mask, scenario_probs):
    """
    Consistency: λ_e · Σ_ω p_ω Σ_{(i,j)∈I} (H^ω_ij − H̄_ij)²

    Penalizza la variabilità di H^ω rispetto alla heatmap media H̄
    sugli archi importanti. Favorisce soluzioni coerenti tra scenari.

    Poiché I_mask copre sia (i,j) che (j,i), divido per 2.

    Parametri
    ---------
    H_list         : lista di K tensori (B, n, n)
    H_bar          : (B, n, n) — Σ_ω p_ω H^ω (non serve ri-calcolarlo)
    I_mask         : (n, n) BoolTensor
    scenario_probs : (K,) FloatTensor
    """
    I_float = I_mask.float().unsqueeze(0)
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        diff = ((H_omega - H_bar) ** 2) * I_float
        loss = loss + p_w * diff.sum(dim=(1, 2)).mean()
    return loss 


def _loss_asymmetry(H_list, scenario_probs):
    """
    Asymmetry: λ_d · Σ_ω p_ω Σ_{(i,j)∈E} (1 − (H^ω_ij − H^ω_ji)²)

    Promuove la direzionalità degli archi nella heatmap: se l'arco (i→j)
    è usato, il verso opposto (j→i) non dovrebbe esserlo nello stesso tour.
    Il termine vale 0 quando |H^ω_ij − H^ω_ji| = 1 (perfettamente asimmetrico)
    e vale 1 quando H^ω_ij = H^ω_ji (perfettamente simmetrico → penalizzato).

    L'aggregazione è su tutti gli archi E (non solo I), su entrambe le
    direzioni. Poiché la somma (i,j)+(j,i) conta ogni coppia due volte,
    non divido per 2: il termine è già bilanciato dalla struttura (i,j)∈E.

    Parametri
    ---------
    H_list         : lista di K tensori (B, n, n)
    scenario_probs : (K,) FloatTensor
    """
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        H_T  = H_omega.transpose(1, 2)                    # H^ω_ji
        asym = (H_omega - H_T) ** 2                       # (H^ω_ij - H^ω_ji)²
        # Azzero la diagonale (auto-loop non ha senso di direzionalità)
        n = H_omega.size(-1)
        diag_mask = torch.eye(n, dtype=torch.bool, device=H_omega.device).unsqueeze(0)
        asym = asym.masked_fill(diag_mask, 1.0)           # diag → (1−1)²=0 contributo
        penalty = (1.0 - asym).clamp(min=0.0)             # (B, n, n)
        loss = loss + p_w * penalty.sum(dim=(1, 2)).mean()
    return loss


def _loss_penalty(H_list, C_mat, I_mask, scenario_probs, alpha):
    I_float    = I_mask.float()
    C_broad    = C_mat * I_float
    H_stack    = torch.stack([H.squeeze(0) for H in H_list], dim=0)  # (K, n, n)
    H_sum      = H_stack.sum(dim=0)                                   # (n, n)
    not_booked = torch.exp(-alpha * H_sum) * I_float

    loss = torch.tensor(0.0, device=H_stack.device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        h = H_omega.squeeze(0) if H_omega.dim() == 3 else H_omega
        penalty = C_broad * h * not_booked
        loss = loss + p_w * penalty.sum()
    return loss


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
    alpha          = DEFAULT_ALPHA,
    lambda1        = DEFAULT_LAMBDA1,
    lambda2        = DEFAULT_LAMBDA2,
    lambda_e       = DEFAULT_LAMBDA_E,
    lambda_d       = DEFAULT_LAMBDA_D,
    include_penalty = False,
    return_components = False,
):
    """
    Loss completa per il TSP stocastico 2-stadi (nuova formulazione).

    L = λ₁ · row-wise
      + λ₂ · no-self-loop
      + minimize distance
      + booking cost          Σ_{(i,j)∈I} c_ij · (1 − e^{−α·H̄_ij})
      + λ_e · consistency     Σ_ω p_ω Σ_{(i,j)∈I} (H^ω_ij − H̄_ij)²
      + λ_d · asymmetry       Σ_ω p_ω Σ_{(i,j)∈E} (1 − (H^ω_ij − H^ω_ji)²)
      [+ penalty]             Σ_ω p_ω Σ_{(i,j)∈I} C_ij·H^ω_ij·e^{−α·H̄_ij}  (opzionale)

    Parametri
    ---------
    T_list          : lista di K tensori (B, n, n) — output GNN softmax per scenario ω
    dist_list       : lista di K tensori (n, n) o (B, n, n) — distanze normalizzate
    I_mask          : (n, n) BoolTensor  — True nelle celle (i,j) e (j,i) per {i,j}∈I
    p_mat           : (n, n) FloatTensor — costi prenotazione c_ij (da build_I_tensors)
    C_mat           : (n, n) FloatTensor — costi multa C_ij        (da build_I_tensors)
    scenario_probs  : (K,) FloatTensor   — p_ω per ogni scenario
    alpha           : float — tasso di saturazione esponenziale booking/penalty
    lambda1         : float — peso row-wise constraint
    lambda2         : float — peso no-self-loop
    lambda_e        : float — peso consistency
    lambda_d        : float — peso asymmetry
    include_penalty : bool  — se True aggiunge il termine penalty alla loss
    return_components : bool — se True ritorna anche il dizionario dei singoli termini

    Ritorna
    -------
    loss : scalar FloatTensor — perdita totale
    (opzionale) components : dict con i singoli termini della loss

    Note implementative
    -------------------
    - dist_list deve contenere le distanze GIÀ normalizzate (via normalize_dist_tensor).
    - scenario_probs deve essere un tensore PyTorch sulla stessa device di T_list[0].
    - La penalty è disabilitata di default (include_penalty=False); attivarla
      può aiutare in fasi avanzate del training quando si vuole raffinare la
      coerenza tra decisione di prenotazione e utilizzo effettivo.
    """
    # ── Heatmap H^ω da T^ω ────────────────────────────────────────────
    H_list = [compute_heatmap(T_omega) for T_omega in T_list]

    # ── Heatmap aggregata H̄ = Σ_ω p_ω H^ω ────────────────────────────
    H_bar = compute_H_bar(H_list, scenario_probs)

    # ── Singoli termini della loss ─────────────────────────────────────
    L_row   = _loss_row_wise(T_list,scenario_probs)
    L_diag  = _loss_self_loop(H_list, scenario_probs)
    L_dist  = _loss_distance(H_list, dist_list, scenario_probs)
    L_book  = _loss_booking(H_list, p_mat, I_mask, alpha)
    L_cons  = _loss_consistency(H_list, H_bar, I_mask, scenario_probs)
    L_asym  = _loss_asymmetry(H_list,scenario_probs)

    # ── Loss totale ────────────────────────────────────────────────────
    loss = (
        lambda1   * L_row
        + lambda2 * L_diag
        + L_dist
        + L_book
        + lambda_e * L_cons
        + lambda_d * L_asym
    )

    L_pen = torch.tensor(0.0, device=H_list[0].device)
    if include_penalty:
        L_pen = _loss_penalty(H_list, C_mat, I_mask, scenario_probs, alpha)
        loss  = loss + L_pen

    if return_components:
        return loss, {
            "total":       float(loss.detach().cpu().item()),
            "row_wise":    float(L_row.detach().cpu().item()),
            "self_loop":   float(L_diag.detach().cpu().item()),
            "distance":    float(L_dist.detach().cpu().item()),
            "booking":     float(L_book.detach().cpu().item()),
            "consistency": float(L_cons.detach().cpu().item()),
            "asymmetry":   float(L_asym.detach().cpu().item()),
            "penalty":     float(L_pen.detach().cpu().item()),
        }

    return loss


# ═══════════════════════════════════════════════════════════════════════════
# DECODIFICA: da x̃ a x ∈ {0,1}
# (usata dopo il training per estrarre la politica di primo stadio)
# ═══════════════════════════════════════════════════════════════════════════

def decode_booking_policy(H_list, I, nodes, I_mask, scenario_probs, alpha=DEFAULT_ALPHA, threshold=0.5):
    """
    Decodifica la decisione di prenotazione binaria dal modello addestrato.

    Calcola b̃_ij = 1 − e^{−α·H̄_ij} per ogni {i,j} ∈ I e applica una soglia.

    Parametri
    ---------
    H_list         : lista di K tensori (1, n, n) — un campione, non un batch
    I              : lista di coppie canoniche (i,j) ∈ I  — da prova_neur.py
    nodes          : lista ordinata degli ID nodo
    I_mask         : (n, n) BoolTensor
    scenario_probs : (K,) FloatTensor
    alpha          : float — tasso di saturazione esponenziale
    threshold      : float — soglia per la decisione binaria (default 0.5)

    Ritorna
    -------
    x_reserved : lista di coppie (i,j) ∈ I per cui b̃_ij ≥ threshold
    x_scores   : dict {(i,j): valore b̃_ij}  — scores continui per analisi
    """
    idx = {v: k for k, v in enumerate(nodes)}

    with torch.no_grad():
        H_stack = torch.stack([H.squeeze(0) for H in H_list], dim=0)  # (K, n, n)
        H_sum   = H_stack.sum(dim=0)                                   # (n, n)
        b_tilde = compute_booking_activation(H_sum, I_mask, alpha)

    x_reserved = []
    x_scores   = {}

    for edge in I:
        i, j   = canon_edge(*edge)
        ii, jj = idx[i], idx[j]
        # Prendo il valore in una direzione (simmetrico per costruzione di I_mask)
        score  = float(b_tilde[0, ii, jj].item())
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
      Ep  42 | total=4.3210 | dist=3.1200 | book=0.4500 | cons=0.0800 |
              | asym=0.2100 | pen=0.0000 | row=0.3000 | diag=0.0200
    """
    prefix = f"Ep {epoch:>4}" if epoch is not None else "Loss"
    return (
        f"{prefix} | total={components['total']:.4f} "
        f"| dist={components['distance']:.4f} "
        f"| book={components['booking']:.4f} "
        f"| cons={components['consistency']:.4f} "
        f"| asym={components['asymmetry']:.4f} "
        f"| pen={components['penalty']:.4f} "
        f"| row={components['row_wise']:.4f} "
        f"| diag={components['self_loop']:.4f}"
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
      - stampa dei componenti (nuova formulazione)
    """
    import torch

    torch.manual_seed(42)
    device = torch.device("cpu")

    # Parametri sintetici
    n        = 6        # nodi
    K        = 4        # scenari
    B        = 8        # batch size

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
        alpha=5.0, lambda1=20.0, lambda2=5.0,
        lambda_e=0.1, lambda_d=0.1,
        include_penalty=True,
        return_components=True,
    )

    print("=== TEST two_stage_utsp_loss (nuova formulazione) ===")
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
        H_list_single, I_fake, nodes, I_mask, probs,
        alpha=5.0, threshold=0.5,
    )
    print(f"\n{check_booking_coverage(x_sc, I_fake)}")
    print("\n=== TEST COMPLETATO ===")