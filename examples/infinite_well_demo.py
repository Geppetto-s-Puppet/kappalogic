"""
examples/infinite_well_demo.py
==============================
無限井戸型ポテンシャル(幅L、両端で波動関数0)の量子プロパゲータを、
「鏡像法」(電磁気学の接地導体板の鏡像と同じ発想)で構成し、
教科書通りの固有関数展開と比較する。

実行: python examples/infinite_well_demo.py
"""
from kappalogic import (
    box_heat_kernel, box_heat_kernel_eigen,
    infinite_well_propagator, infinite_well_propagator_eigen,
)

if __name__ == "__main__":
    L, xp = 1.0, 0.3

    print("=== 有限区間[0,L]の熱核(鏡像法 vs 固有関数展開) ===")
    for t in [0.5, 0.05, 0.005]:
        for x in [0.1, 0.5, 0.8]:
            a = box_heat_kernel(x, xp, t, L)
            b = box_heat_kernel_eigen(x, xp, t, L)
            print(f"t={t:<6} x={x}: images={a:.6f}  eigen={b:.6f}  diff={abs(a - b):.2e}")
        print()

    print("=== 無限井戸型ポテンシャルの量子プロパゲータ(鏡像法 vs 固有関数展開) ===")
    print("(ウィック回転 D->i*hbar/(2m), t->T + epsilonプリスクリプションで正則化)")
    for T in [0.3, 0.05]:
        for x in [0.1, 0.5, 0.8]:
            a = infinite_well_propagator(x, xp, T, L)
            b = infinite_well_propagator_eigen(x, xp, T, L)
            print(f"T={T:<6} x={x}: images={a:.4f}  eigen={b:.4f}  |diff|={abs(a - b):.2e}")
        print()
