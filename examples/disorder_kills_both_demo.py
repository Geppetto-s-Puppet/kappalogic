"""
disorder_kills_both_demo.py
===========================
v0.86: 乱れは「色」と「伝導」を同時に殺す —— そして原因は同じ**局在**。

同じ乱れた鎖から、kappalogic の**同じ2つの検出器**で2つの物性が出る:
    光学 ε₂  = Σ |x_ij|² · δ_ξ(ΔE-ω) · 占有差        (縦の遷移)
    伝導 σ   = Σ |v_ij|² · δ_η(E_i-E_j) · 輸送窓w    (横の伝導)
                        ────────        ──────
                    デルタ検出器      NOT 検出器

実行: py -3.14 examples/disorder_kills_both_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappalogic.disorder import disordered_chain, inverse_participation_ratio
from kappalogic.transport import kubo_dc_conductivity
from kappalogic.optics import (
    eps2_from_hamiltonian, HC_EV_NM, spectrum_to_xyz, xyz_to_srgb,
    srgb_to_hex, color_saturation, absorbance,
)


def ssh_disordered(N, t1, t2, W, rng):
    H = np.zeros((N, N))
    for i in range(N - 1):
        H[i, i + 1] = H[i + 1, i] = -(t1 if i % 2 == 0 else t2)
    H[np.arange(N), np.arange(N)] = W * (rng.random(N) - 0.5)
    return H


def main():
    rng = np.random.default_rng(0)
    N = 160
    lam = np.linspace(380, 780, 141)
    Eg = HC_EV_NM / lam

    print("同じ乱れた鎖から『色』と『伝導』を同じ検出器で計算する")
    print(f"(N={N}、光学は交替ホッピング鎖 gap=2.4eV、伝導は一様鎖のバンド中心)\n")
    print("  W      色         彩度     σ_DC      局在長ζ    σ/ζ")
    print("  " + "-" * 62)
    for W in [0.0, 1.0, 2.0, 4.0, 8.0]:
        # --- 色(交替ホッピング鎖、アンサンブル平均スペクトル) ---
        nreal = 1 if W == 0.0 else 8
        acc = np.zeros_like(Eg)
        for _ in range(nreal):
            acc += eps2_from_hamiltonian(ssh_disordered(N, 1.9, 0.7, W, rng), Eg, 0.06)
        T = np.exp(-absorbance(Eg, acc / nreal) * 8.0)
        rgb = xyz_to_srgb(*spectrum_to_xyz(lam, T))

        # --- 伝導と局在長(一様鎖、バンド中心) ---
        sig, zet = [], []
        for _ in range(6):
            H = disordered_chain(N, W, rng)
            sig.append(kubo_dc_conductivity(H))
            E, V = np.linalg.eigh(H)
            mid = np.argsort(np.abs(E))[:20]
            zet.append(np.mean([1.0 / inverse_participation_ratio(V[:, m]) for m in mid]))
        s, z = float(np.mean(sig)), float(np.mean(zet))
        print(f"  {W:<6} {srgb_to_hex(rgb)}   {color_saturation(rgb):.3f}   "
              f"{s:8.4f}  {z:8.2f}  {s/z:8.4f}")

    print("\n読み方 —— ここが面白いところ:")
    print(" ① **乱れは色と伝導を同時に殺す**。彩度が単調に落ち、σ_DC は3桁落ちる。")
    print("    どちらも原因は同じ:状態が局在して、遠くまで届かなくなること。")
    print(" ② **σ/ζ が弱〜中程度の乱れでほぼ一定** = σ_DC ∝ ζ。")
    print("    伝導(久保公式)と波動関数の広がり(IPR)という**完全に独立な2つの")
    print("    測り方**が、同じ局在長を見ている——内的な整合性のアンカー。")
    print(" ③ 光学(縦の遷移)と伝導(横の伝導)が、kappalogic の**同じ2つの検出器**")
    print("    (デルタ検出器 δ_ξ と NOT 検出器の窓)で書ける。物性の別々の顔が、")
    print("    同じ部品の組み替えで出てくる。")
    print("\n 正直な注記:")
    print(" - σ/ζ の比例は強乱れ(ζ が数サイト)で崩れる——拡散的輸送から強局在")
    print("   (ホッピング)へ移るためで、物理的に正しい振る舞い。")
    print(" - σ の絶対値は任意単位(e²/ħ 等の前因子は付けていない)。意味があるのは")
    print("   乱れ依存性と ζ との比例関係の方。")


if __name__ == "__main__":
    main()
