#!/usr/bin/env python3
# build_speck32_clat_dataset.py
# Dataset cLAT para UNA ronda de SPECK-32/64 (w=16, ALPHA=7, BETA=2).
# Genera pares de máscaras (alpha,beta)->(gamma,delta) restringidos por peso Hamming
# combinado (sobre 32 bits) y guarda correlación normalizada, bias, y rewards útiles.
#
# Uso:
#   python build_speck32_clat_dataset.py --hw-in 2 --hw-out 2 --out speck32_clat_hw2x2.csv
# Opciones ver al final (argparse).

from __future__ import annotations
import argparse
import csv
import math
import os
from typing import Iterable, Iterator, List, Tuple


W = 16           
MASKW = (1 << W) - 1
ALPHA = 7       
BETA  = 2       


def hw(x: int) -> int:
    return x.bit_count()

def ror(x: int, r: int, w: int = W) -> int:
    r %= w
    return ((x >> r) | ((x & ((1 << r) - 1)) << (w - r))) & ((1 << w) - 1)

def rol(x: int, r: int, w: int = W) -> int:
    r %= w
    return (((x << r) & ((1 << w) - 1)) | (x >> (w - r))) & ((1 << w) - 1)


# Correlación normalizada de z = x + y (mod 2^W) vía DP
def walsh_corr_add(ax: int, ay: int, bz: int, w: int = W) -> float:
    """
    corr = 2^{-2w} * Sum_{x,y} (-1)^{ <ax,x> ⊕ <ay,y> ⊕ <bz, (x+y mod 2^w)> }
    DP exacta por acarreo. Complejidad O(w).
    """
    # state[c] = contribución acumulada con carry=c al procesar los i bits menos significativos
    state0, state1 = 1.0, 0.0  # carry-in al LSB es 0

    for i in range(w):
        ax_i = (ax >> i) & 1
        ay_i = (ay >> i) & 1
        bz_i = (bz >> i) & 1

        # new_state[c'] = sum sobre (c, xi, yi) tal que carry_out=c'
        ns0 = 0.0
        ns1 = 0.0

        # c=0
        if state0 != 0.0:
            s0 = state0
            # (xi, yi) ∈ {(0,0),(0,1),(1,0),(1,1)}
            # c=0 → sume = xi+yi
            # (0,0): z=0, c'=0
            phase = (ax_i & 0) ^ (ay_i & 0) ^ (bz_i & 0)
            ns0 += s0 if phase == 0 else -s0
            # (0,1): z=1, c'=0
            phase = (ax_i & 0) ^ (ay_i & 1) ^ (bz_i & 1)
            ns0 += s0 if phase == 0 else -s0
            # (1,0): z=1, c'=0
            phase = (ax_i & 1) ^ (ay_i & 0) ^ (bz_i & 1)
            ns0 += s0 if phase == 0 else -s0
            # (1,1): z=0, c'=1
            phase = (ax_i & 1) ^ (ay_i & 1) ^ (bz_i & 0)
            ns1 += s0 if phase == 0 else -s0

        # c=1
        if state1 != 0.0:
            s1 = state1
            # (0,0): z=1, c'=0
            phase = (ax_i & 0) ^ (ay_i & 0) ^ (bz_i & 1)
            ns0 += s1 if phase == 0 else -s1
            # (0,1): z=0, c'=1
            phase = (ax_i & 0) ^ (ay_i & 1) ^ (bz_i & 0)
            ns1 += s1 if phase == 0 else -s1
            # (1,0): z=0, c'=1
            phase = (ax_i & 1) ^ (ay_i & 0) ^ (bz_i & 0)
            ns1 += s1 if phase == 0 else -s1
            # (1,1): z=1, c'=1  (porque 1+1+1 = 3 -> z=1, c'=1)
            phase = (ax_i & 1) ^ (ay_i & 1) ^ (bz_i & 1)
            ns1 += s1 if phase == 0 else -s1

        state0, state1 = ns0, ns1

    total = state0 + state1
    return total / (1 << (2 * w))

# Correlación cLAT de UNA ronda de SPECK 32/64
def corr_speck32_round(alpha: int, beta: int, gamma: int, delta: int) -> float:
    """
    corr( <alpha,x0> ⊕ <beta,y0>, <gamma,x1> ⊕ <delta,y1> )
    Ecuaciones de ronda (sin clave):
      x1 = ROR(x0,7) + y0 (mod 2^16)
      y1 = ROL(y0,2) ^ x1
    Manipulando máscaras:
      b_z = gamma ^ delta
      a_x = ROR(alpha, 7)
      a_y = beta ^ ROR(delta, 2)
    """
    b_z = gamma ^ delta
    a_x = ror(alpha, ALPHA, W)
    a_y = beta ^ ror(delta, BETA, W)
    return walsh_corr_add(a_x, a_y, b_z, W)

# Generación de máscaras por peso Hamming combinado (32b)
def masks_32_by_combined_hw(max_hw: int, include_zero: bool = False) -> List[int]:
    """
    Devuelve todos los enteros m de 32 bits con HW(m) ∈ [1..max_hw] (o [0..max_hw] si include_zero).
    Se usa para generar (alpha,beta) y (gamma,delta) al partir m en low/high 16 bits.
    """
    out: List[int] = []
    limit = 1 << 32
    # Generación combinatoria eficiente por posiciones
    positions = list(range(32))
    if include_zero and max_hw >= 0:
        out.append(0)

    # hw = 1
    if max_hw >= 1:
        for i in positions:
            out.append(1 << i)
    # hw = 2
    if max_hw >= 2:
        for i in range(32):
            for j in range(i + 1, 32):
                out.append((1 << i) | (1 << j))
    # hw >= 3 (opcional; cuidado con explosión combinatoria)
    if max_hw >= 3:
        from itertools import combinations
        for k in range(3, max_hw + 1):
            for comb in combinations(positions, k):
                m = 0
                for i in comb:
                    m |= (1 << i)
                out.append(m)
    return out

def split32_to_pair(m: int) -> Tuple[int, int]:
    alpha = m & MASKW
    beta  = (m >> 16) & MASKW
    return alpha, beta

# Construcción del dataset y guardado
def build_dataset(hw_in: int, hw_out: int) -> List[Tuple[int,int,int,int,float,float,float,float,int,int,int,int]]:
    """
    Genera todas las filas para HW combinado ≤ hw_in (entrada) y ≤ hw_out (salida).
    Retorna lista de tuplas con:
      (alpha, beta, gamma, delta, corr, bias, abs_corr, corr_sq,
       hwa, hwb, hwg, hwd)
    """
    inputs_m  = masks_32_by_combined_hw(hw_in, include_zero=False)
    outputs_m = masks_32_by_combined_hw(hw_out, include_zero=False)

    rows: List[Tuple[int,int,int,int,float,float,float,float,int,int,int,int]] = []
    for mi in inputs_m:
        alpha, beta = split32_to_pair(mi)
        hwa, hwb = hw(alpha), hw(beta)
        for mo in outputs_m:
            gamma, delta = split32_to_pair(mo)
            hwg, hwd = hw(gamma), hw(delta)

            corr = corr_speck32_round(alpha, beta, gamma, delta)
            bias = 0.5 * corr
            abs_corr = abs(corr)
            corr_sq = corr * corr

            rows.append((
                alpha, beta, gamma, delta,
                corr, bias, abs_corr, corr_sq,
                hwa, hwb, hwg, hwd
            ))
    return rows

def save_csv(rows, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "alpha","beta","gamma","delta",
            "corr","bias","abs_corr","corr_sq",
            "hw_alpha","hw_beta","hw_gamma","hw_delta"
        ])
        w.writerows(rows)

def save_npz(rows, out_path: str):
    import numpy as np
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr = np.array(rows, dtype=np.float64)
    # Guardamos columnas separadas con nombres
    np.savez(out_path,
             alpha=arr[:,0].astype(np.uint16),
             beta=arr[:,1].astype(np.uint16),
             gamma=arr[:,2].astype(np.uint16),
             delta=arr[:,3].astype(np.uint16),
             corr=arr[:,4],
             bias=arr[:,5],
             abs_corr=arr[:,6],
             corr_sq=arr[:,7],
             hw_alpha=arr[:,8].astype(np.uint8),
             hw_beta=arr[:,9].astype(np.uint8),
             hw_gamma=arr[:,10].astype(np.uint8),
             hw_delta=arr[:,11].astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description="Construye dataset cLAT de una ronda SPECK-32/64 para RL.")
    ap.add_argument("--hw-in", type=int, default=2, help="Peso Hamming combinado máximo (32b) para (alpha,beta). Default: 2")
    ap.add_argument("--hw-out", type=int, default=2, help="Peso Hamming combinado máximo (32b) para (gamma,delta). Default: 2")
    ap.add_argument("--out", type=str, default="speck32_clat_hw2x2.csv", help="Ruta de salida (ext .csv o .npz).")
    args = ap.parse_args()

    print(f"[i] Generando máscaras entrada (≤{args.hw_in}) y salida (≤{args.hw_out})…")
    rows = build_dataset(args.hw_in, args.hw_out)
    n = len(rows)
    print(f"[i] Filas: {n}")

    if args.out.lower().endswith(".csv"):
        save_csv(rows, args.out)
        print(f"[✓] Guardado CSV: {args.out}")
    elif args.out.lower().endswith(".npz"):
        save_npz(rows, args.out)
        print(f"[✓] Guardado NPZ: {args.out}")
    else:
        # Default CSV si extensión desconocida
        save_csv(rows, args.out + ".csv")
        print(f"[✓] Extensión no reconocida, guardado como CSV: {args.out + '.csv'}")

if __name__ == "__main__":
    main()
