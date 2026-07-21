"""
transport_demo.py
=================
電子輸送(電気/熱伝導・比熱)を kappalogic の検出器で計算(v0.80)。
核心: 輸送窓 w(E)=-∂f/∂E = kappalogic の NOT(E-μ;2kT)/(4kT)。

実行: py -3.14 examples/transport_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappalogic.transport import (
    transport_moment, lorenz_ratio, wiedemann_franz_check,
    sommerfeld_heat_capacity_coefficient, LORENZ_NUMBER,
)


def main():
    print("電子輸送窓 w(E) = -∂f/∂E = kappalogic の NOT(E-μ; 2kT)/(4kT)")
    print("(フェルミ準位まわり幅 kT の『輸送窓』= OFF検出器 NOT)\n")

    print(" 窓のモーメント L_n = ∫(E-μ)^n w dE  (kT=0.2):")
    for n in range(4):
        print(f"   L_{n} = {transport_moment(n, 0.0, 0.2):.6f}"
              f"{'   (= (π²/3)(kT)² = %.6f)' % ((np.pi**2/3)*0.2**2) if n==2 else ''}")

    print("\n Wiedemann-Franz 則 κ/(σT)(緩和時間・DOS・速度に依らない):")
    for (g, vF, tau) in [(1.0, 1.0, 1.0), (2.5, 0.7, 3.0), (0.4, 1.8, 0.5)]:
        r = wiedemann_franz_check(0.2, g=g, vF=vF, tau=tau)["ratio"]
        print(f"   (g={g}, vF={vF}, τ={tau}): κ/(σT) = {r:.5f}   (π²/3 = {np.pi**2/3:.5f})")

    print(f"\n ローレンツ数 L0 = π²/3 (k_B/e)² = {LORENZ_NUMBER:.4e} W·Ω/K²")
    print(f"   (実験の Wiedemann-Franz 値 ~2.44e-8 と一致)")

    print("\n 電子比熱ゾンマーフェルト係数 γ = (π²/3) k_B² g(μ)  (C_v = γT, 線形):")
    for g in [1.0, 2.0, 3.5]:
        print(f"   g(μ)={g}: γ = {sommerfeld_heat_capacity_coefficient(g):.4f} (k_B=1 単位)")

    print("\n読み方:")
    print(" - 輸送窓は kappalogic の NOT 検出器そのもの(誤差0)。電流・熱を運ぶのは")
    print("   この窓の中の電子だけ、という輸送の描像が kappalogic の言葉で書ける。")
    print(" - 窓の2次モーメント L_2=(π²/3)(kT)² から Wiedemann-Franz のローレンツ数")
    print("   (普遍定数 2.44e-8)と電子比熱の線形則が出る——教科書物性の再現。")
    print(" - superconductivity.py(BCS)と同じく『物性を kappalogic の延長で解く』例。")


if __name__ == "__main__":
    main()
