"""
disordered_material_demo.py
===========================
v0.85: (a) 命題39 の次元描像を完成(2D=周辺次元)、(b) モジュール合流——
乱れた鎖(disorder.py)を光学(optics.py)へ流し込み、「乱れた物質の色」を出す。

実行: py -3.14 examples/disordered_material_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappalogic.disorder import dimensional_beta_survey
from kappalogic.optics import (
    eps2_from_hamiltonian, material_color_from_hamiltonian, HC_EV_NM,
    spectrum_to_xyz, xyz_to_srgb, srgb_to_hex, color_saturation, absorbance,
)


def ssh_disordered(N, t1, t2, W, rng):
    H = np.zeros((N, N))
    for i in range(N - 1):
        H[i, i + 1] = H[i + 1, i] = -(t1 if i % 2 == 0 else t2)
    H[np.arange(N), np.arange(N)] = W * (rng.random(N) - 0.5)
    return H


def main():
    rng = np.random.default_rng(4)

    print("=== 1. 命題39 の次元描像を完成: β が 0 を横切るか ===")
    s = dimensional_beta_survey(rng, n_slices_2d=15000, n_slices_3d=5000)
    print("  d=2 (周辺次元):")
    for W, b in s["d2"]:
        print(f"    W={W:<6} β = {b:+.4f}   (負のまま)")
    print("  d=3:")
    for W, b in s["d3"]:
        tag = "金属" if b > 0 else "絶縁体"
        print(f"    W={W:<6} β = {b:+.4f}   ({tag})")
    print("  → d=1: β<0(強局在) / d=2: β<0 のまま0に届かない(周辺・転移なし)")
    print("    d=3: 0 を横切る(転移あり、ν=1.602)。**2 が下部臨界次元**。")

    print("\n=== 2. モジュール合流: 乱れた鎖 → 光学 → 色 ===")
    N, t1, t2 = 100, 1.9, 0.7
    print(f"  交替ホッピング鎖 N={N}、清浄時のギャップ = {2*abs(t1-t2):.1f} eV")
    lam = np.linspace(380, 780, 161)
    Eg = HC_EV_NM / lam
    print("\n   W      色         彩度    ギャップ内吸収 ε₂(1.2eV)")
    for W in [0.0, 1.0, 2.0, 3.0, 4.0]:
        nreal = 1 if W == 0.0 else 10
        acc = np.zeros_like(Eg)
        gaps = []
        for _ in range(nreal):
            H = ssh_disordered(N, t1, t2, W, rng)
            acc += eps2_from_hamiltonian(H, Eg, 0.06)
            gaps.append(eps2_from_hamiltonian(H, 1.2, 0.06))
        e2 = acc / nreal
        T = np.exp(-absorbance(Eg, e2) * 8.0)
        rgb = xyz_to_srgb(*spectrum_to_xyz(lam, T))
        print(f"   {W:<6} {srgb_to_hex(rgb)}    {color_saturation(rgb):.3f}   {np.mean(gaps):.5f}")

    print("\n読み方 / 正直な注記:")
    print(" - **乱れは色を洗い流す**: 彩度が単調に低下する。機構はギャップ内に")
    print("   状態が生まれること(Urbach 裾)——清浄時は 0 だった ε₂(1.2eV) が立ち上がる。")
    print(" - 単調なのは**彩度**であって輝度ではない。強い乱れはバンド構造を壊して")
    print("   スペクトル重みを広いエネルギーへ薄く分散させるので、可視域の吸収が")
    print("   むしろ弱まり淡くなる(輝度は非単調)。")
    print(" - 2次元の弱乱れ(W<=2)は転送行列が収束せず扱っていない。局在長が指数的に")
    print("   長いためで、これは数値の都合であると同時に 2D 弱局在の物理そのもの。")


if __name__ == "__main__":
    main()
