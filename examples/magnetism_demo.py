"""
magnetism_demo.py
=================
磁性(v0.81)。命題37: 自己無撞着写像の傾きが1を横切るか、で臨界性が決まる。

kappalogic のカーネル自身の自己無撞着 m = tanh(Tc·m/T) = k(Tc·m; ξ=T) は
**本物の二次相転移**(β=1/2, γ=1)を持つ——命題33 で「臨界性なし」と否定した
OR_N の蓄積遷移との対比が要点。

実行: py -3.14 examples/magnetism_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappalogic.magnetism import (
    curie_weiss_magnetization, curie_weiss_susceptibility,
    curie_weiss_slope_at_zero, has_critical_point,
    pauli_susceptibility, landau_susceptibility, total_electron_susceptibility,
    MEAN_FIELD_AMPLITUDE,
)
from kappalogic.theory import or_n_kicked_map_slope_at_unstable_point


def main():
    Tc = 1.0
    print("=== Curie-Weiss 強磁性: m = tanh(Tc·m/T) = kappalogic の k(Tc·m; ξ=T) ===")
    print(" T/Tc      m        m/(1-T/Tc)^(1/2)   [→ √3 = %.5f なら β=1/2]" % MEAN_FIELD_AMPLITUDE)
    for t in [0.99, 0.999, 0.9999]:
        m = curie_weiss_magnetization(t * Tc, Tc)
        print(f"  {t:.4f}  {m:.6f}      {m/(1-t)**0.5:.5f}")

    print("\n T>Tc の帯磁率 χ(Curie-Weiss則 χ=1/(T-Tc) なら γ=1):")
    for T in [1.05, 1.2, 1.5]:
        chi = curie_weiss_susceptibility(T, Tc)
        print(f"  T={T:.2f}: χ={chi:8.4f}   χ·(T-Tc)={chi*(T-Tc):.5f}")

    print("\n=== 命題37: 臨界性は「傾きが1を横切るか」で決まる ===")
    below = curie_weiss_slope_at_zero(0.9 * Tc, Tc)
    above = curie_weiss_slope_at_zero(1.1 * Tc, Tc)
    print(f" Curie-Weiss の m=0 での傾き Tc/T: T=0.9Tc → {below:.4f} (>1)、"
          f"T=1.1Tc → {above:.4f} (<1)")
    print(f"   → 1 を横切る: {has_critical_point(below, above)}  ⇒ ピッチフォーク分岐=二次相転移")
    slopes = [or_n_kicked_map_slope_at_unstable_point(xi) for xi in [0.2, 0.05, 0.02]]
    print(f" OR_N の蹴られた写像 G'(t*)(命題33): {['%.2f' % s for s in slopes]} —— 常に >1")
    print(f"   → 1 を横切る: {has_critical_point(slopes[0], slopes[-1])}  ⇒ 臨界性なし(命題33)")
    print(" **同じ tanh の自己無撞着でも、傾きが1を横切るかで臨界性の有無が分かれる**")

    print("\n=== フェルミ面の磁気応答(輸送窓 = kappalogic の NOT)===")
    g = lambda E: np.sqrt(max(E, 0.0))       # 自由電子 DOS
    mu = 1.0
    print("  kT      χ_P/μ_B²    χ_L/μ_B²(反磁性)   χ_total/μ_B²")
    for kT in [0.001, 0.05, 0.1]:
        cp = pauli_susceptibility(g, mu, kT)
        cl = landau_susceptibility(g, mu, kT)
        ct = total_electron_susceptibility(g, mu, kT)
        print(f"  {kT:<7} {cp:.6f}    {cl:+.6f}         {ct:.6f}")
    print("  χ_P→g(μ)=1 (T→0)。χ_L=-(1/3)χ_P は自由電子の標準結果を**借用**")
    print("  (kappalogic から導いたものではない)。正味 χ=(2/3)χ_P で常磁性。")


if __name__ == "__main__":
    main()
