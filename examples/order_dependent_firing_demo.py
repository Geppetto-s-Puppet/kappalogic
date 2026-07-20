"""
examples/order_dependent_firing_demo.py
=======================================
命題32(順序依存の発火反転定理、v0.72)のデモ。派生定理ノート#6
(樹状突起の順序依存統合)への回答。

naive fold(飽和する逐次OR = 命題23の蹴られたwalk)を、樹状突起の
非線形統合の最小モデルと見なすと:

  「同じシナプス入力の集合でも、到着タイミングの順序を変えるだけで
   発火/非発火が反転する入力集合が存在し、その集合を全順列を試さず
   O(n log n)の2ソートだけで事前に特定できる」

実行(pip install -e . 済みが前提):
    python3 examples/order_dependent_firing_demo.py
"""
import itertools
import numpy as np
from kappalogic import or_n_firing_is_order_dependent
from kappalogic.theory import _reg

SEP = "=" * 70


def NOT(x, xi):
    return 1 - _reg(x, xi)


def OR2(x, y, xi):
    return NOT(NOT(x, xi) * NOT(y, xi), xi)


def naive_fold(vals, xi):
    acc = vals[0]
    for v in vals[1:]:
        acc = OR2(acc, v, xi)
    return acc


def L(x, xi):
    return 2 * np.log(np.cosh(x / xi))


def demo_single_flip_example():
    print("\n" + SEP)
    print("1. 同じ入力集合が、到着順序次第で発火にも非発火にもなる例")
    print(SEP)
    xi = 1e-2
    vals = np.array([-0.017, 0.0086, -0.0213])
    print(f"  シナプス入力(強度): {vals}")
    print(f"  各入力の log-charge L: {np.round([L(v, xi) for v in vals], 3)}")
    print()
    folds = []
    for p in itertools.permutations(vals):
        f = naive_fold(list(p), xi)
        folds.append(f)
        fired = "発火" if f > 0.5 else "非発火"
        print(f"    到着順 {np.round(p, 4)} -> fold={f:.4f} [{fired}]")
    print(f"\n  => 同じ3入力で fold は [{min(folds):.3f}, {max(folds):.3f}] に割れる")
    r = or_n_firing_is_order_dependent(vals, xi)
    print(f"  予言子の判定: order_dependent = {r['order_dependent']}")
    print(f"    (降順fold={r['fold_descending']:.3f}>0.5 かつ "
          f"昇順fold={r['fold_ascending']:.3f}<=0.5 なので反転しうる)")


def demo_predictor_vs_bruteforce():
    print("\n" + SEP)
    print("2. 予言子(2ソート)vs 全順列(n!)の一致率")
    print(SEP)
    xi = 1e-2
    rng = np.random.default_rng(7)
    Cstar = xi * 0.5 * np.log(4 / xi)
    n_trials = 3000
    correct = tp = fp = fn = 0
    for _ in range(n_trials):
        n = rng.integers(3, 6)
        vals = rng.uniform(-2 * Cstar, 2 * Cstar, n)
        fires = [naive_fold(list(p), xi) > 0.5 for p in itertools.permutations(vals)]
        true_flip = any(fires) and not all(fires)
        pred = or_n_firing_is_order_dependent(vals, xi)["order_dependent"]
        correct += (pred == true_flip)
        tp += (true_flip and pred)
        fp += ((not true_flip) and pred)
        fn += (true_flip and (not pred))
    print(f"  試行数: {n_trials}")
    print(f"  精度:   {100 * correct / n_trials:.2f}%")
    print(f"  適合率: {100 * tp / (tp + fp) if tp + fp else 100:.1f}%  "
          f"(反転と予言したら実際に反転)")
    print(f"  再現率: {100 * tp / (tp + fn) if tp + fn else 100:.1f}%  "
          f"(見逃しは境界の浮動小数点同着)")
    print("\n  => 全 n! 順列を試さず、降順・昇順の2ソートだけで反転しうる")
    print("     入力集合を厳密に近い精度で特定できる。")


def demo_neuroscience_framing():
    print("\n" + SEP)
    print("3. 神経科学的な含意(派生定理ノート#6)")
    print(SEP)
    print("""  標準的な Leaky Integrate-and-Fire ニューロンはシナプス入力を
  線形和で蓄積するので、入力の到着順序に依存しない。
  しかし実際の樹状突起の統合は非線形・飽和的(dendritic saturation)
  であることが実験的に知られている。

  naive fold(飽和する逐次OR)をその最小モデルと見なすと、命題32は
  「同じシナプス入力集合でも、到着順序を変えるだけで発火が反転する
   集合が存在し、それを事前に特定できる」という定量的予測を与える。
  これは原理的にはパッチクランプ実験で検証可能。

  正直な限界: これは飽和的統合の定性的性質を最小モデルで定量化した
  ものであって、実際のニューロンの生物物理を含んだモデルではない。""")


if __name__ == "__main__":
    demo_single_flip_example()
    demo_predictor_vs_bruteforce()
    demo_neuroscience_framing()
    print("\n" + SEP)
    print("デモ完了。")
    print(SEP)
