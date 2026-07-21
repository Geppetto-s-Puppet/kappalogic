"""
criticality_survey_demo.py
==========================
命題38(v0.82): kappalogic の自己無撞着系を、たった1つの基準で分類する。

  **不動点における線形化した傾きが、制御パラメータを振って ±1 を横切るか**
     +1 を横切る → ピッチフォーク分岐 → 二次相転移(β=1/2)
     -1 を横切る → 周期倍分岐
     横切らない  → 臨界点なし

実行: py -3.14 examples/criticality_survey_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappalogic.criticality import (
    bifurcation_survey, bcs_beta_amplitude, curie_weiss_beta_amplitude,
    bcs_linear_coefficient,
)
from kappalogic.superconductivity import bcs_critical_temperature
from kappalogic.dynamics import not_map_critical_xi


def main():
    print("=== 命題38: 分岐分類(傾きが ±1 を横切るか)===\n")
    print(f" {'系':<40} {'傾き(低)':>10} {'傾き(高)':>10}   分岐")
    print(" " + "-" * 88)
    for r in bifurcation_survey():
        print(f" {r['system']:<40} {r['slope_low']:10.3f} {r['slope_high']:10.3f}   {r['bifurcation']}")

    print("\n=== 二次相転移組: 臨界指数 β=1/2 の振幅 ===")
    print(f" Curie-Weiss  m/(1-T/Tc)^(1/2) → {curie_weiss_beta_amplitude():.5f}   (理論 √3 = 1.73205)")
    for t in [0.99, 0.995, 0.999]:
        print(f" BCS (t={t})  Δ/Δ0/(1-T/Tc)^(1/2) → {bcs_beta_amplitude(0.3, reduced_t=t):.5f}"
              f"   (既知の BCS 振幅 ~1.74)")

    print("\n=== BCS の線形化係数 λ·I(0,T) が 1 を横切る様子 ===")
    lam = 0.3
    Tc = bcs_critical_temperature(lam)
    for f in [0.8, 0.95, 1.0, 1.05, 1.2]:
        print(f"  T/Tc={f:.2f}: λI(0,T) = {bcs_linear_coefficient(f*Tc, lam):.6f}")

    xi_c = not_map_critical_xi()[0]
    print(f"\n=== 周期倍組: NOT 写像の臨界 ξ_c = {xi_c:.5f}(命題17、乗数がちょうど -1)===")

    print("\n読み方:")
    print(" - 磁性(Curie-Weiss)と超伝導(BCS)は、どちらも傾きが +1 を横切る")
    print("   ピッチフォーク分岐 = **本物の二次相転移**(β=1/2、平均場)。")
    print(" - NOT 写像は -1 を横切る周期倍分岐(相転移ではなく力学系の分岐)。")
    print(" - OR_N の蹴られた walk だけは傾きが常に >1 で **1 を横切らない**")
    print("   → 臨界点を持たない(命題33 の否定的結果の正体)。")
    print(" - 命題28『t* の自己無撞着は Curie-Weiss と同型』と命題33『しかし普遍性")
    print("   クラスは共有しない』の差の正体が、この傾き条件だった。")


if __name__ == "__main__":
    main()
