"""
color_and_disorder_demo.py
==========================
物質の「色」を kappalogic の検出器から計算し、乱れ(ランダムネス)を入れる(v0.83)。

  ε₂(ω) = < |M|² · δ_ξ(ΔE_k - ω) >   ← δ_ξ は kappalogic の delta_approx
                                        (= NOT 検出器の規格化版)
  → 吸収 α ∝ ω ε₂ → 透過率 T=exp(-αd) → CIE → sRGB

実行: py -3.14 examples/color_and_disorder_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappalogic.optics import (
    ssh_transition_energies, dielectric_imaginary, spectral_weight, transmitted_color,
)
from kappalogic.disorder import localization_length, localization_exponent


def bands_for_gap(gap_eV, width=3.0, nk=600):
    tm, tp = gap_eV / 2, (gap_eV + 2 * width) / 2
    return ssh_transition_energies((tp + tm) / 2, (tp - tm) / 2, nk)


def main():
    print("=== 1. 内的アンカー: 総和則は線幅 ξ に依存しない ===")
    print("   (kappalogic の delta_approx が ∫δ=1 に厳密規格化されているため)")
    dE = bands_for_gap(2.0)
    w = np.linspace(0.0, 12.0, 6000)
    print("   ξ       ∫ε₂dω        ∫ω·ε₂dω")
    for xi in [0.02, 0.05, 0.1]:
        e2 = dielectric_imaginary(w, dE, xi)
        print(f"   {xi:<7} {spectral_weight(w,e2,0):.6f}     {spectral_weight(w,e2,1):.6f}")

    print("\n=== 2. バンドギャップから物質の『色』を計算 ===")
    print("   gap(eV)   色      解釈")
    labels = {3.6: "無色透明(可視光を吸わない)",
              3.0: "淡い黄(紫端だけ吸う)",
              2.4: "橙・琥珀(青緑を吸収)",
              2.0: "暗い赤",
              1.6: "ほぼ黒",
              1.0: "黒(可視全域を吸収)"}
    for gap in [3.6, 3.0, 2.4, 2.0, 1.6, 1.0]:
        r = transmitted_color(bands_for_gap(gap), xi=0.08, thickness=10.0)
        print(f"   {gap:<9} {r['hex']}  {labels[gap]}")
    print("   → ZnS(透明)→ CdS(黄橙)→ Si(黒)という実際の半導体の色の進行と同じ。")

    print("\n=== 3. ランダムネス: 乱れと Anderson 局在 ===")
    rng = np.random.default_rng(7)
    print("   1D 一様鎖 N=1200、バンド中心の局在長 ζ = 1/IPR")
    p, pairs = localization_exponent(1200, [1.0, 1.4, 2.0], rng, n_samples=4, n_states=20)
    print("     W        ζ         ζ·W²")
    for W, z in pairs:
        print(f"     {W:<8} {z:8.2f}  {z*W**2:8.2f}")
    print(f"   フィット: ζ ∝ W^(-{p:.2f})   (1D 弱乱れの理論値は -2)")

    print("\n読み方 / 正直な注記:")
    print(" - 光学応答まるごとが kappalogic の部品(デルタ検出器 × tanh 占有差)で書ける。")
    print("   ξ はここでは**スペクトル線幅**——ξ の意味論に5つ目が加わった。")
    print(" - 総和則の ξ 非依存性は、検出器の規格化から**自動的に**従う内的アンカー。")
    print(" - CIE 等色関数・sRGB 変換・直接遷移の吸収の形は**借用**した標準理論。")
    print(" - 乱れの側: 1D では任意の W>0 で全状態が局在し、**金属-絶縁体転移は無い**")
    print("   (命題33 と同種の、正直に記録すべき否定的事実)。真の Anderson 転移")
    print("   (非平均場の普遍性クラス、ν≈1.57)は 3D で現れる——普遍性クラスへの宿題。")


if __name__ == "__main__":
    main()
