"""
bcs_superconductivity_demo.py
=============================
kappalogic のカーネル k(x;ξ)=tanh(x/ξ) で BCS 超伝導ギャップ方程式を
自己無撞着に解く(v0.78)。物性(T_c・ギャップ Δ)を kappalogic の延長で計算。

実行: py -3.14 examples/bcs_superconductivity_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappalogic.superconductivity import (
    bcs_critical_temperature, bcs_gap, bcs_gap_zero_temperature,
    bcs_universal_ratio, BCS_UNIVERSAL_RATIO,
)


def main():
    wD = 1.0
    print("BCS gap equation solved with kappalogic's kernel k=tanh, xi=2*k_B*T")
    print(f"(energies in units of Debye frequency wD=1; BCS universal ratio = {BCS_UNIVERSAL_RATIO:.4f})\n")

    print(" lambda   k_B*Tc       Delta(0)     Delta(0)/kTc   1.134*wD*e^(-1/λ)")
    print(" " + "-" * 70)
    for lam in [0.15, 0.2, 0.3, 0.4, 0.5]:
        Tc = bcs_critical_temperature(lam, wD)
        D0 = bcs_gap_zero_temperature(lam, wD)
        approx = 1.134 * wD * np.exp(-1.0 / lam)
        print(f"  {lam:.2f}    {Tc:.5f}     {D0:.5f}      {D0/Tc:.4f}         {approx:.5f}")

    print("\n Delta(T)/Delta(0) の温度依存(lambda=0.3):")
    lam = 0.3
    Tc = bcs_critical_temperature(lam, wD)
    D0 = bcs_gap_zero_temperature(lam, wD)
    print("  T/Tc :  " + "  ".join(f"{f:.2f}" for f in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0]))
    row = []
    for f in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0]:
        D = bcs_gap(max(f * Tc, 1e-6), lam, wD)
        row.append(f"{D/D0:.2f}")
    print("  Δ/Δ0 :  " + "  ".join(row))

    print("\n読み方:")
    print(" - 弱結合(λ小)で普遍比 Δ(0)/kTc → 1.7639(=π/e^γ)を再現。強結合で増大")
    print("   (強結合補正=BCS弱結合普遍性の破れ、物理的に正しい)。")
    print(" - T_c = 1.134 wD e^(-1/λ): T_c を上げるには λ(結合)か wD(軽い原子=水素化物)")
    print("   を上げる——常温超伝導が高圧水素化物(H3S, LaH10)で追われる理由そのもの。")
    print(" - Δ(T) は T=0 で最大、T_c で √(1-T/Tc) 的に 0 へ。")
    print("\n 正直な位置づけ:")
    print(" - これは教科書 BCS の再現であって新物理ではない。ただし tanh=kappalogic の")
    print("   カーネル、ξ=2kT の熱カーネル(fermi_occupation と地続き)、自己無撞着=命題28")
    print("   と同じ骨格——『物性を kappalogic の延長で解く』最初の具体例。")
    print(" - ユーザーの着想『OR_N 安全定理→常温超伝導』については: OR_N の閾値 t* に")
    print("   BCS の対応物は無く(弱結合BCSは結合閾値を持たない)、命題33 で蓄積遷移は")
    print("   真の相転移でないと確定済み。安全定理そのものは手がかりにならない。共有")
    print("   するのは『自己無撞着 tanh 平均場』という骨格のみ、というのが誠実な答え。")


if __name__ == "__main__":
    main()
