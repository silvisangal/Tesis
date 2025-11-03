# Command for building:: python clat_speck32.py --build --out "<name>.pkl.gz"
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import pickle, gzip, os, argparse

def mask(n: int) -> int: return (1 << n) - 1

@dataclass
class CLAT8:
    m: int = 8
    cLATmin: Dict[int, Dict[int, int]] = None
    cLATN: Dict[int, Dict[int, Dict[int, int]]] = None
    cLATu: Dict[int, Dict[int, Dict[int, List[int]]]] = None
    cLATw: Dict[int, Dict[int, Dict[int, List[int]]]] = None
    cLATb: Dict[int, Dict[int, Dict[int, Dict[int, int]]]] = None

    def __post_init__(self):
        self.cLATmin = defaultdict(lambda: defaultdict(lambda: self.m))
        self.cLATN   = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.cLATu   = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.cLATw   = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.cLATb   = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def build(self):
        m = self.m
        M = mask(m)
        for b in (0, 1):
            for v in range(1 << m):
                for w in range(1 << m):
                    for u in range(1 << m):
                        A = u ^ v
                        B = u ^ w
                        C = u ^ v ^ w
                        Cb = [(C >> (m - 1 - j)) & 1 for j in range(m)]
                        Cw = 0
                        MT = [0] * m
                        Z  = (1 << (m - 1)) if b == 1 else 0
                        if b == 1:
                            Cw += 1
                            MT[0] = 1
                        for i in range(1, m):
                            MT[i] = (Cb[i - 1] + MT[i - 1]) & 1
                            if MT[i] == 1:
                                Z  |= (1 << (m - 1 - i))
                                Cw += 1
                        F1 = A & (~(A & Z) & M)
                        F2 = B & (~(B & Z) & M)
                        if F1 == 0 and F2 == 0:
                            idx = self.cLATN[v][b][Cw]
                            self.cLATN[v][b][Cw] = idx + 1
                            self.cLATu[v][b][Cw].append(u)
                            self.cLATw[v][b][Cw].append(w)
                            self.cLATb[u].setdefault(v, {}).setdefault(w, {})[b] = (MT[m - 1] + Cb[m - 1]) & 1
                            if self.cLATmin[v][b] > Cw:
                                self.cLATmin[v][b] = Cw

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "m": self.m,
            "cLATmin": {v: dict(bd) for v, bd in self.cLATmin.items()},
            "cLATN":   {v: {b: dict(cd) for b, cd in bd.items()} for v, bd in self.cLATN.items()},
            "cLATu":   {v: {b: {cw: ul for cw, ul in cd.items()} for b, cd in bd.items()} for v, bd in self.cLATu.items()},
            "cLATw":   {v: {b: {cw: wl for cw, wl in cd.items()} for b, cd in bd.items()} for v, bd in self.cLATw.items()},
            "cLATb":   {u: {v: {w: dict(bd) for w, bd in vd.items()} for v, vd in ud.items()} for u, ud in self.cLATb.items()},
        }
        if path.endswith(".gz"):
            with gzip.open(path, "wb") as f: pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "wb") as f: pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "CLAT8":
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f: d = pickle.load(f)
        else:
            with open(path, "rb") as f: d = pickle.load(f)
        tbl = CLAT8(m=d["m"])
        # reconstrucción a defaultdicts:
        for v, bd in d["cLATmin"].items():
            tbl.cLATmin[v].update(bd)
        for v, bd in d["cLATN"].items():
            for b, cd in bd.items():
                tbl.cLATN[v][b].update(cd)
        for v, bd in d["cLATu"].items():
            for b, cd in bd.items():
                for cw, ul in cd.items():
                    tbl.cLATu[v][b][cw].extend(ul)
        for v, bd in d["cLATw"].items():
            for b, cd in bd.items():
                for cw, wl in cd.items():
                    tbl.cLATw[v][b][cw].extend(wl)
        for u, vd in d["cLATb"].items():
            for v, wd in vd.items():
                for w, bd in wd.items():
                    tbl.cLATb[u][v][w].update(bd)
        return tbl

# Split–Lookup–Recombination 
def hw(x: int) -> int: return x.bit_count()
def _min_weight(tbl: CLAT8, vk: int) -> int: return min(tbl.cLATmin[vk][0], tbl.cLATmin[vk][1])

def enumerate_u_w_for_v16(tbl: CLAT8, v16: int, max_bound: Optional[int] = None):
    m = tbl.m; assert m == 8
    v_blocks = [(v16 >> 8) & 0xFF, v16 & 0xFF]  # MSB, LSB
    mins = [_min_weight(tbl, vk) for vk in v_blocks]

    def dfs(k: int, b_k: int, acc_u: int, acc_w: int, acc_wt: int):
        if k == 2:
            yield acc_u, acc_w, acc_wt; return
        vk = v_blocks[k]
        for cw in sorted(tbl.cLATN[vk][b_k].keys()):
            if max_bound is not None:
                rest = sum(mins[j] for j in range(k+1, 2))
                if acc_wt + cw + rest > max_bound: 
                    continue
            uL = tbl.cLATu[vk][b_k][cw]
            wL = tbl.cLATw[vk][b_k][cw]
            for idx in range(len(uL)):
                uk, wk = uL[idx], wL[idx]
                b_next = tbl.cLATb[uk][vk][wk][b_k]
                yield from dfs(k+1, b_next, (acc_u<<8)|uk, (acc_w<<8)|wk, acc_wt+cw)
    yield from dfs(0, 0, 0, 0, 0)


def main():
    ap = argparse.ArgumentParser(description="cLAT m=8 para Split-Lookup-Recombination (SPECK-32).")
    ap.add_argument("--build", action="store_true", help="Construir cLAT m=8 y guardarla.")
    ap.add_argument("--out", type=str, default="clat_speck_32.pkl.gz", help="Archivo de salida (.pkl o .pkl.gz).")
    ap.add_argument("--check", type=str, default="", help="Ruta de cLAT para verificar carga.")
    ap.add_argument("--examples", nargs="*", help="Valores v16 en hex (ej: 0x1234 0xabcd) para listar (u,w,Cw).")
    ap.add_argument("--max-bound", type=int, default=None, help="Poda por cota superior de peso total (opcional).")
    args = ap.parse_args()

    if args.build:
        print("[i] Construyendo cLAT m=8… (una sola vez)")
        tbl = CLAT8(m=8)
        tbl.build()
        tbl.save(args.out)
        print(f"[✓] cLAT guardada en {args.out}")

    if args.check:
        print(f"[i] Cargando {args.check}…")
        tbl = CLAT8.load(args.check)
        # reporte mínimo
        v0 = 0
        min0 = min(tbl.cLATmin[v0][0], tbl.cLATmin[v0][1])
        print(f"[✓] Cargada. m={tbl.m}. cLATmin[v=0][b] min={min0}")

    if args.examples:
        if not args.check:
            raise SystemExit("Usa --check <archivo_guardado> junto con --examples para consultar.")
        tbl = CLAT8.load(args.check)
        for vhex in args.examples:
            v = int(vhex, 16)
            print(f"\n[v={v:04x}] ejemplos (u,w,Cw):")
            shown = 0
            for (u,w,cw) in enumerate_u_w_for_v16(tbl, v, args.max_bound):
                print(f"  u={u:04x}  w={w:04x}  Cw={cw}")
                shown += 1
                if shown >= 10:  
                    print("  …")
                    break

if __name__ == "__main__":
    main()
