# -*- coding: utf-8 -*-
 

import math
import torch
import torch.nn.functional as F


 
# PARAMETRI DEFAULT  
 
# Pesi della loss
DEFAULT_LAMBDA1   = 10.0   # row-wise constraint    
DEFAULT_LAMBDA2   =  10.0   # no-self-loop          
DEFAULT_LAMBDA_E  =  1   # consistency           
DEFAULT_LAMBDA_D  =  1   # asymmetry             
DEFAULT_ALPHA     =  5.0


 
# COSTRUZIONE MASCHERE / TENSORI STATICI
# (chiamate una volta sola, poi riutilizzate ad ogni forward)
 

def canon_edge(i, j):
    return (i, j) if i < j else (j, i)

# costruisco maschere e tensori di costo per archi in I
def build_I_tensors(I, nodes, p, C, device):
    n   = len(nodes)
    idx = {v: k for k, v in enumerate(nodes)}

    I_mask = torch.zeros(n, n, dtype=torch.bool,  device=device) # (n, n) BoolTensor — True nelle posizioni (i,j) e (j,i) per ogni {i,j} ∈ I
    p_mat  = torch.zeros(n, n, dtype=torch.float32, device=device) # (n, n) FloatTensor — costi di prenotazione (0 fuori da I)
    C_mat  = torch.zeros(n, n, dtype=torch.float32, device=device) # (n, n) FloatTensor — costi multa

    for edge in I:
        i, j   = canon_edge(*edge)
        ii, jj = idx[i], idx[j]
        # Recupero il valore dal dict (supporta sia (i,j) che (j,i) come chiave)
        p_val = p.get((i, j), p.get((j, i), 0.0))
        C_val = C.get((i, j), C.get((j, i), 0.0))

        # I è non orientato: sia (ii,jj) che (jj,ii) appartengono alla prenotazione
        for a, b in [(ii, jj), (jj, ii)]:
            I_mask[a, b] = True
            p_mat[a, b]  = p_val
            C_mat[a, b]  = C_val

    return I_mask, p_mat, C_mat, idx # (n, n) FloatTensor — costi multa

# passo da matrice di distanze di scenario omega ad un tensore
def build_dist_tensor(scenario_dist, nodes, device):
    n = len(nodes)
    D = torch.zeros(n, n, dtype=torch.float32, device=device)
    for ii, i in enumerate(nodes):
        for jj, j in enumerate(nodes):
            if i != j:
                D[ii, jj] = float(scenario_dist[i][j])
    return D #FloatTensor con D[ii, jj] = scenario_dist[nodes[ii]][nodes[jj]]


# normalizzo la matrice di distanze con al media da passare alla GNN
def normalize_dist_tensor(D, mode="mean_positive"):
    """
    mode : "mean_positive"     divide per la media delle distanze positive fuori diagonale
           "median_positive"   divide per la mediana
           "none"              nessuna normalizzazione
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


 
# CALCOLO HEATMAP  H^ω  DA  T^ω
#primissima 
#def compute_heatmap(T_omega):
#    """
#    H^ω = T^ω · roll(T^{ωT}, -1)#
#
#    Parametri T_omega : (B, n, n) — output GNN dopo softmax per lo scenario ω
#
 #    Ritorna H_omega : (B, n, n)
  #  """
   # return torch.bmm(T_omega, torch.roll(T_omega.transpose(1, 2), -1, 1))

#terza versione
def compute_heatmap(T_omega):
    B, n, _ = T_omega.shape
    H = torch.zeros(B, n, n, device=T_omega.device)
    for t in range(n - 1):
        pt  = T_omega[:, :, t].unsqueeze(2)
        pt1 = T_omega[:, :, t+1].unsqueeze(1)
        H  += torch.bmm(pt, pt1)
    pn = T_omega[:, :, -1].unsqueeze(2)
    p1 = T_omega[:, :,  0].unsqueeze(1)
    H += torch.bmm(pn, p1)
    
    return H


# H_bar è la heatmap media pesata per la distribuzione di prob degli scenari
def compute_H_bar(H_list, scenario_probs): 
    H_bar = torch.zeros_like(H_list[0])

    for H_omega, p_w in zip(H_list, scenario_probs):
        H_bar = H_bar + p_w * H_omega

    return H_bar
 
    
    
# DECISIONE RILASSATA DI PRENOTAZIONE

# Attivazione delle prenotazioni, segue la struttura 1-exp(alpha*somma di Hij)
def compute_booking_activation(H_sum, I_mask, alpha):
    I_float = I_mask.float().unsqueeze(0)          # (1, n, n)
    b_tilde = (1.0 - torch.exp(-alpha * H_sum)) * I_float
    return b_tilde  # (B, n, n)

 
# LOSS ADATTATA
 
# penalizzo le righe di T che non sommano a 1, serve per avere doppia stocasticità della matrice (per le colonne c'è softmax)
# identico al paper
def _loss_row_wise(T_list, scenario_probs):
    loss = torch.tensor(0.0, device=T_list[0].device)
    for T_omega, p_w in zip(T_list, scenario_probs):
        row_sums = T_omega.sum(dim=2)        # (B, n)
        loss     = loss + p_w * ((row_sums - 1.0) ** 2).sum(dim=1).mean()
    return loss

# evito selfloop, per cui penalizzo la diagonale e cerco di averla a zero 
# identico al paper
def _loss_self_loop(H_list, scenario_probs): 
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        diag = torch.diagonal(H_omega, dim1=1, dim2=2).sum(dim=1)  # (B,)
        loss = loss + p_w * diag.mean()
    return loss

# voglio minimizzare il costo dato dalle distanze
# simile al paper ma per ogni scenario passo delle distanze diverse
def _loss_distance(H_list, dist_list, scenario_probs): 
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, D_omega, p_w in zip(H_list, dist_list, scenario_probs):
        # D_omega : (n, n) oppure (B, n, n)
        if D_omega.dim() == 2:
            D_omega = D_omega.unsqueeze(0)   #  (1, n, n)
        cost = (H_omega * D_omega).sum(dim=(1, 2))   # (B,)
        loss = loss + p_w * cost.mean()
    return loss


# termine per gestire le prenotazioni
# inquesto modo la prenotazione funziona che:
# se prenoto i,j allora pago p; se prenoto j,i pago p; se entrambi sono alti, pago al massimo p
# Nuovo rispetto al paper
def _loss_booking(H_list, p_mat, I_mask, alpha):
    H_stack = torch.stack([H.squeeze(0) for H in H_list], dim=0)  # (K, n, n)
    H_sum = H_stack.sum(dim=0) # (n, n)
    
    n = H_sum.size(0)
    cost = torch.tensor(0.0, device=H_sum.device)  
                
    for ii in range(n):
        for jj in range(ii + 1, n):
            if bool(I_mask[ii, jj].item()):
                S_ij = H_sum[ii, jj] + H_sum[jj, ii]
                activation = 1.0 - torch.exp(-alpha * S_ij)
                cost = cost + p_mat[ii, jj] * activation

    return cost

# penalizzo la variabilità/differenza di una data heatmap in base alla heatmap media
# cerco un accordo tra i valori degli scenari, in particolare archi in I
# Nuovo rispetto al paper
def _loss_consistency(H_list, H_bar, I_mask, scenario_probs): 
    I_float = I_mask.float().unsqueeze(0)
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        diff = ((H_omega - H_bar) ** 2) * I_float
        loss = loss + p_w * diff.sum(dim=(1, 2)).mean()
    return loss 





# nuova versione, in questo modo non penalizzo archi assenti in entrambe le direzione, ma sono quando ij e ji sono entrambi elevati
#Penalizza solo l'uso simultaneo dei due versi i,j e j,i
# nuovo rispetto al paper
def _loss_asymmetry(H_list, scenario_probs):
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        H_T = H_omega.transpose(1, 2)
        reciprocal = H_omega * H_T
        loss = loss + p_w * reciprocal.sum(dim=(1, 2)).mean()
    return loss

#def _loss_asymmetry(H_list, scenario_probs):
#    loss = torch.tensor(0.0, device=H_list[0].device)
#    for H_omega, p_w in zip(H_list, scenario_probs):
#        H_T = H_omega.transpose(1, 2)
#        s   = H_omega + H_T          # H_ij + H_ji
#        d   = H_omega - H_T          # H_ij - H_ji
#        term = s - d ** 2            # (B, n, n)
#        loss = loss + p_w * term.sum(dim=(1, 2)).mean()
#    return loss

# penalità: se uso un arco in I ma non l ho prenotato pago una penalità
# nuovo rispetto al paper
def _loss_penalty(H_list, C_mat, I_mask, scenario_probs, alpha):
    H_stack = torch.stack([H.squeeze(0) for H in H_list], dim=0)  # (K, n, n)
    H_sum = H_stack.sum(dim=0) # (n, n)

    n = H_sum.size(0)
    loss = torch.tensor(0.0, device=H_sum.device)

    for ii in range(n):
        for jj in range(ii + 1, n):
            if bool(I_mask[ii, jj].item()):
                S_ij = H_sum[ii, jj] + H_sum[jj, ii]
                not_booked = torch.exp(-alpha * S_ij)

                for H_omega, p_w in zip(H_list, scenario_probs):
                    h = H_omega.squeeze(0)
                    usage_omega = h[ii, jj] + h[jj, ii]

                    loss = loss + p_w * C_mat[ii, jj] * usage_omega * not_booked

    return loss



# LOSS PRINCIPALE

def two_stage_utsp_loss( T_list, dist_list, I_mask,
    p_mat, C_mat, scenario_probs,
    alpha          = DEFAULT_ALPHA,
    lambda1        = DEFAULT_LAMBDA1,
    lambda2        = DEFAULT_LAMBDA2,
    lambda_e       = DEFAULT_LAMBDA_E,include_entropy=False,
    lambda_d       = DEFAULT_LAMBDA_D,
    include_penalty = False, return_components = False,):
    """    
    Parametri
   
    T_list          : lista di K tensori (B, n, n)   output GNN softmax per scenario ω
    dist_list       : lista di K tensori (n, n) o (B, n, n)   distanze normalizzate
    I_mask          : (n, n) BoolTensor    True nelle celle (i,j) e (j,i) per {i,j}∈I
    p_mat           : (n, n) FloatTensor   costi prenotazione c_ij (da build_I_tensors)
    C_mat           : (n, n) FloatTensor  costi multa C_ij (da build_I_tensors)
    scenario_probs  : (K,) FloatTensor  p_ω per ogni scenario
    alpha           : float   tasso di saturazione esponenziale booking/penalty
    lambda1         : float   peso row-wise constraint
    lambda2         : float   peso no-self-loop
    lambda_e        : float   peso consistency
    lambda_d        : float   peso asymmetry
    include_penalty : bool    se True aggiunge il termine penalty alla loss
    return_components : bool   se True ritorna anche il dizionario dei singoli termini

    """
    
    
    
    #   Heatmap H^ω da T^ω  
    H_list = [compute_heatmap(T_omega) for T_omega in T_list]

    #   Heatmap aggregata H̄ = Σ_ω p_ω H^ω  
    H_bar = compute_H_bar(H_list, scenario_probs)

    # Singoli termini della loss  
    L_row   = _loss_row_wise(T_list, scenario_probs)
    L_diag  = _loss_self_loop(H_list, scenario_probs)
    L_dist  = _loss_distance(H_list, dist_list, scenario_probs)
    L_book  = _loss_booking(H_list, p_mat, I_mask, alpha)
    L_asym  = _loss_asymmetry(H_list, scenario_probs)

    #   Loss totale  
    loss = (
        lambda1   * L_row
        + lambda2 * L_diag
        + L_dist
        + L_book
        + lambda_d * L_asym
    )

    L_pen = torch.tensor(0.0, device=H_list[0].device)
    if include_penalty:
        L_pen = _loss_penalty(H_list, C_mat, I_mask, scenario_probs, alpha)
        loss  = loss + L_pen

    L_cons = torch.tensor(0.0, device=H_list[0].device)
    if include_entropy:
        L_cons = _loss_consistency(H_list, H_bar, I_mask, scenario_probs)
        loss   = loss + lambda_e * L_cons

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


# DECODIFICA: da x̃ a x ∈ {0,1} (usata dopo il training per estrarre la politica di primo stadio)
  

def decode_booking_policy(H_list, I, nodes, I_mask, scenario_probs, alpha=DEFAULT_ALPHA, threshold=0.8):
    idx = {v: k for k, v in enumerate(nodes)}

    with torch.no_grad():
        H_stack = torch.stack([H.squeeze(0) for H in H_list], dim=0)
        H_sum   = H_stack.sum(dim=0) 
        b_tilde = compute_booking_activation(H_sum, I_mask, alpha)

    x_reserved = []
    x_scores   = {}
    print(f"DEBUG decode_booking_policy: len(H_list)={len(H_list)}, H_list[0].shape={H_list[0].shape}")
    for edge in I:
        #prima 
        # i, j   = canon_edge(*edge)
        #dopo
        i, j   = edge 
        ii, jj = idx[i], idx[j]
        score = max(float(b_tilde[0, ii, jj].item()),
                    float(b_tilde[0, jj, ii].item()))
        x_scores[(i, j)] = score
        if score >= threshold:
            x_reserved.append((i, j))

    return x_reserved, x_scores


# DIAGNOSTICA DELLA LOSS DURANTE IL TRAINING

# stampa delle compoenti della loss durante alcune epoche
def format_loss_components(components, epoch=None):
     
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


# per ogni arco in I stampa lo score e la decisione
def check_booking_coverage(x_scores, I, threshold=0.8): # la threshold la gesticsco a mano qua ma non è coerente 
    lines = ["Decisioni di prenotazione NN:"]
    n_booked = 0
    for edge in sorted(I):
        i, j  = edge
        score = x_scores.get((i, j), x_scores.get((j, i), 0.0))
        flag  = "✓ PRENOTA" if score >= threshold else "✗ non prenota"
        lines.append(f"  {{{i},{j}}}  x̃={score:.4f}  {flag}")
        if score >= threshold:
            n_booked += 1
    lines.append(f"  Totale prenotazioni: {n_booked}/{len(I)}")
    return "\n".join(lines)