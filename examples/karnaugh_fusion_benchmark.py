"""
examples/karnaugh_fusion_benchmark.py
======================================
AND(AND(a,b),c,...) のように毎回reg()を呼ぶ「素朴な連結」と、
先に全部の積を取ってreg()を1回だけ呼ぶ「融合版(AND_n)」の速度・数値安定性を比較する。

実行: python examples/karnaugh_fusion_benchmark.py
"""
import time
import numpy as np
from kappalogic import AND, AND_n


def and_naive(vals):
    r = vals[0]
    for v in vals[1:]:
        r = AND(r, v)
    return r


if __name__ == "__main__":
    # 速度比較
    N, M = 50, 20000
    vals = list(np.random.randn(N) + 5)
    t0 = time.time()
    for _ in range(M):
        and_naive(vals)
    t1 = time.time()
    for _ in range(M):
        AND_n(*vals)
    t2 = time.time()
    print(f"N={N}項のAND x {M}回: naive={t1 - t0:.3f}s, AND_n(融合)={t2 - t1:.3f}s, "
          f"speedup={(t1 - t0) / (t2 - t1):.1f}x")

    # 数値安定性の比較(境界ぎりぎりの値が混ざる場合)
    tricky = [3.0, 5.0, -2.0, 7.0, 0.001]  # 最後の項がxi(=1e-3)と同スケール
    print(f"\n境界ぎりぎりの値を含む場合: {tricky}")
    print(f"  naive : {float(and_naive(tricky)):.4f} (個別判定なので曖昧になりやすい)")
    print(f"  AND_n : {float(AND_n(*tricky)):.4f} (全体の積で判定するので安定)")
