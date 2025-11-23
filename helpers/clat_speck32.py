# Command for building:: python clat_speck32.py --build --out "<name>.pkl.gz"
# Command for checking intersections: python clat_speck32.py --check "<name>.pkl.gz" --w16-for-u16-v16 0x0004 0x0006 --max-bound 3 --limit <limit>
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

def print_intersections(tbl: CLAT8, v_list, max_bound=None, limit=20):
    for vhex in v_list:
        v = int(vhex, 16)
        print(f"\n[v={v:04x}] (u, w, Cw, u&w, hw(u&w))")
        shown = 0
        for (u, w, cw) in enumerate_u_w_for_v16(tbl, v, max_bound=max_bound):
            inter = u & w
            print(f"  u={u:04x}  w={w:04x}  Cw={cw}  u&w={inter:04x}  hw={inter.bit_count()}")
            shown += 1
            if shown >= limit:
                print("  …")
                break

def list_w_for_u_v8(tbl: CLAT8, u: int, v: int, b: int | None = None):
    m = tbl.m
    assert m == 8
    bs = [b] if b in (0, 1) else [0, 1]
    out = []
    for bb in bs:
        for cw, ulist in tbl.cLATu[v][bb].items():
            wlist = tbl.cLATw[v][bb][cw]
            for idx, uu in enumerate(ulist):
                if uu == u:
                    w = wlist[idx]
                    b_next = tbl.cLATb[u][v][w][bb]
                    out.append((w, cw, bb, b_next))
    out.sort(key=lambda t: (t[1], t[2], t[0]))
    return out

def enumerate_w_for_u16_v16(tbl: CLAT8, u16: int, v16: int, max_bound: int | None = None):
    """
    MSB->LSB con m=8, t=2. Fija u16 y v16; enumera (w16, Cw_total).
    Aplica poda por 'max_bound' si se da.
    """
    m = tbl.m; assert m == 8
    u_blocks = [(u16 >> 8) & 0xFF, u16 & 0xFF]  
    v_blocks = [(v16 >> 8) & 0xFF, v16 & 0xFF] 

    mins = [min(tbl.cLATmin[v_blocks[k]][0], tbl.cLATmin[v_blocks[k]][1]) for k in (0,1)]

    def dfs(k: int, b_in: int, acc_w: int, acc_wt: int):
        if k == 2:
            yield acc_w, acc_wt
            return

        uk = u_blocks[k]
        vk = v_blocks[k]

        for cw in sorted(tbl.cLATN[vk][b_in].keys()):
            if max_bound is not None:
                rest = sum(mins[j] for j in range(k+1, 2))
                if acc_wt + cw + rest > max_bound:
                    continue

            ulist = tbl.cLATu[vk][b_in][cw]
            wlist = tbl.cLATw[vk][b_in][cw]

            for idx, uu in enumerate(ulist):
                if uu != uk:
                    continue
                wk = wlist[idx]
                b_next = tbl.cLATb[uk][vk][wk][b_in]
                dfs_w = (acc_w << 8) | wk
                yield from dfs(k+1, b_next, dfs_w, acc_wt + cw)
    yield from dfs(0, 0, 0, 0)


def main():
    ap = argparse.ArgumentParser(description="cLAT m=8 para Split-Lookup-Recombination (SPECK-32).")
    ap.add_argument("--build", action="store_true", help="Construir cLAT m=8 y guardarla.")
    ap.add_argument("--out", type=str, default="clat_speck_32.pkl.gz", help="Archivo de salida (.pkl o .pkl.gz).")
    ap.add_argument("--check", type=str, default="", help="Ruta de cLAT para verificar carga.")
    ap.add_argument("--examples", nargs="*", help="Valores v16 en hex (ej: 0x1234 0xabcd) para listar (u,w,Cw).")
    ap.add_argument("--max-bound", type=int, default=None, help="Poda por cota superior de peso total (opcional).")
    ap.add_argument("--intersections", nargs="*", help="V hex (ej: 0x0006) para ver u&w.")
    ap.add_argument("--w-for-u-v8", nargs=3, metavar=("u","v","b"), help="Lista w en m=8 para u,v y b∈{0,1|None}.")
    ap.add_argument("--w16-for-u16-v16", nargs=2, metavar=("u16","v16"), help="Lista w16 para u16, v16.")
    ap.add_argument("--limit", type=int, default=20, help="Líneas máx. a imprimir.")
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
    if args.intersections:
        if not args.check:
            raise SystemExit("Usa --check <archivo_guardado> junto con --intersections.")
        tbl = CLAT8.load(args.check)
        print_intersections(tbl, args.intersections, max_bound=args.max_bound, limit=args.limit)
    if args.w_for_u_v8:
        if not args.check: raise SystemExit("Usa --check <archivo> con --w-for-u-v8.")
        tbl = CLAT8.load(args.check)
        u = int(args.w_for_u_v8[0], 0); v = int(args.w_for_u_v8[1], 0)
        b = None if args.w_for_u_v8[2].lower() == "none" else int(args.w_for_u_v8[2])
        rows = list_w_for_u_v8(tbl, u, v, b)
        print(f"[m=8] u={u:02x} v={v:02x} (b={b})  ->  {len(rows)} coincidencias")
        for i,(w,cw,bb,bn) in enumerate(rows[:args.limit]):
            print(f"  w={w:02x}  Cw={cw}  b_in={bb}  b_next={bn}")
        if len(rows) > args.limit: print("  …")

    if args.w16_for_u16_v16:
        if not args.check: raise SystemExit("Usa --check <archivo> con --w16-for-u16-v16.")
        tbl = CLAT8.load(args.check)
        u16 = int(args.w16_for_u16_v16[0], 0); v16 = int(args.w16_for_u16_v16[1], 0)
        cnt = 0
        print(f"[16b] u={u16:04x} v={v16:04x}")
        for w16, cw in enumerate_w_for_u16_v16(tbl, u16, v16, max_bound=args.max_bound):
            print(f"  w={w16:04x}  Cw={cw}")
            cnt += 1
            if cnt >= args.limit: print("  …"); break
if __name__ == "__main__":
    main()
