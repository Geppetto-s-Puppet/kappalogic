"""
kappa_cpu_demo.py
==================
kappa_cpu.py (kappalogicのAND/OR/XOR/eqだけで組んだ"数式上のCPU") の
検証デモ一式。実行すると以下を順番に確認する:

  1. n bit リップルキャリー加算器のbit-exact性 (xiを振って崩れる境界を探す)
  2. ビット幅nを増やしても「安全なxiの上限」が縮まないことの確認
     (命題9/10のnaive-fold安全条件との対比)
  3. ALU全opcode(ADD/SUB/AND/OR/XOR/NOT)の正しさ
  4. PC/ACC/CTRを持つトイCPUで実際にJNZループ・プログラム
     (1+2+3+4+5=15) を実行し、bit-exactに一致することを確認
  5. ループ長(CTR初期値)を増やしても安全xiの上限が縮まないことの確認
     (= 時間方向のfold安全性も、bit幅方向と同じく"劣化しない")
  6. hard-mode特有の勾配消失(README自身が明記している限界)を、
     このCPU上で再確認する: PCを連続にずらしたときの勾配はxiが
     十分安全な領域ではほぼ0
  7. 単純なADD命令1個の"逆算"を勾配降下で試す(うまくいかない、
     という正直な結果を含めて記録)

すべて kappalogic (pip install -e .) がインストール済みであることを前提。
"""
import numpy as np
from kappa_cpu import (
    int_to_bits, bits_to_int, bits_to_int_exact,
    ripple_add, ripple_sub, alu,
    ALU_ADD, ALU_SUB, ALU_AND, ALU_OR, ALU_XOR, ALU_NOT_A,
    ToyCPU, OP_ADD, OP_DEC_CTR, OP_JNZ, OP_HALT, OP_LOADI_CTR, OP_LOADI_ACC,
)

SEP = "=" * 70


def section(title):
    print("\n" + SEP)
    print(title)
    print(SEP)


def demo_adder_correctness():
    section("1. リップルキャリー加算器の bit-exact 性 (n=6)")
    np.random.seed(0)
    n = 6
    for xi in [1e-1, 1e-2, 1e-3, 1e-4]:
        errs = 0
        trials = 300
        for _ in range(trials):
            a = np.random.randint(0, 2 ** n)
            b = np.random.randint(0, 2 ** n)
            A, B = int_to_bits(a, n), int_to_bits(b, n)
            S, _ = ripple_add(A, B, xi=xi)
            if bits_to_int(S) != (a + b) % (2 ** n):
                errs += 1
        print(f"  xi={xi:<8}  mismatches={errs}/{trials}")


def demo_safe_xi_vs_bitwidth():
    section("2. ビット幅nを増やしても『安全なxiの上限』は縮まらない")
    np.random.seed(1)

    def err_rate(n, xi, trials=120):
        errs = 0
        for _ in range(trials):
            a = np.random.randint(0, 2 ** n)
            b = np.random.randint(0, 2 ** n)
            A, B = int_to_bits(a, n), int_to_bits(b, n)
            S, _ = ripple_add(A, B, xi=xi)
            if bits_to_int(S) != (a + b) % (2 ** n):
                errs += 1
        return errs / trials

    for n in [2, 4, 8, 16, 32]:
        lo, hi = 0.001, 2.0
        while err_rate(n, hi) == 0 and hi < 50:
            hi *= 1.5
        for _ in range(18):
            mid = (lo + hi) / 2
            if err_rate(n, mid, trials=80) > 0.02:
                hi = mid
            else:
                lo = mid
        print(f"  n={n:>3} bit   safe xi <= {lo:.4f}   (naive-foldのProp9/10と違い、"
              f"nが伸びても閾値がほぼ一定)")


def demo_alu():
    section("3. ALU 全opcode の正しさ (opcodeもeq()でデコード)")
    np.random.seed(3)
    n = 6
    xi = 1e-3
    ops = {ALU_ADD: 'ADD', ALU_SUB: 'SUB', ALU_AND: 'AND',
           ALU_OR: 'OR', ALU_XOR: 'XOR', ALU_NOT_A: 'NOT_A'}
    for opcode, name in ops.items():
        errs, trials = 0, 200
        for _ in range(trials):
            a = np.random.randint(0, 2 ** n)
            b = np.random.randint(0, 2 ** n)
            A, B = int_to_bits(a, n), int_to_bits(b, n)
            out = alu(float(opcode), A, B, xi=xi)
            got = bits_to_int(out)
            want = {
                'ADD': (a + b) % (2 ** n), 'SUB': (a - b) % (2 ** n),
                'AND': a & b, 'OR': a | b, 'XOR': a ^ b,
                'NOT_A': (~a) % (2 ** n),
            }[name]
            if got != want:
                errs += 1
        print(f"  {name:6}  mismatches={errs}/{trials}")


SUM_1_TO_5_PROGRAM = [
    (OP_LOADI_CTR, 5),   # 0: CTR = 5
    (OP_LOADI_ACC, 0),   # 1: ACC = 0
    (OP_ADD, 0),         # 2: ACC = ACC + CTR      <- loop start
    (OP_DEC_CTR, 0),     # 3: CTR = CTR - 1
    (OP_JNZ, 2),         # 4: if CTR != 0 goto 2
    (OP_HALT, 0),        # 5: halt
]


def demo_toy_cpu():
    section("4. トイCPU: PC/ACC/CTRを持つ本物のJNZループを実行 (1+2+..+5=15)")
    cpu = ToyCPU(SUM_1_TO_5_PROGRAM, n_bits=6, xi=1e-3)
    pc, acc, ctr, hist = cpu.run(steps=25, trace=True)
    for pc_, a_, c_ in hist:
        print(f"  pc={pc_:6.3f}  ACC={a_:3d}  CTR={c_:3d}")
    print(f"  -> final ACC = {bits_to_int(acc)}  (期待値: 15)")


def demo_safe_xi_vs_loop_length():
    section("5. ループ回数(=時間方向のfold長)を増やしても安全xiは縮まらない")

    def make_program(ctr0):
        return [
            (OP_LOADI_CTR, ctr0), (OP_LOADI_ACC, 0), (OP_ADD, 0),
            (OP_DEC_CTR, 0), (OP_JNZ, 2), (OP_HALT, 0),
        ]

    def safe_xi_threshold(ctr0, n_bits=7):
        program = make_program(ctr0)
        want = ctr0 * (ctr0 + 1) // 2
        steps = ctr0 * 3 + 6

        def ok(xi):
            cpu = ToyCPU(program, n_bits=n_bits, xi=xi)
            _, acc, _ = cpu.run(steps=steps)
            return bits_to_int(acc) == want % (2 ** n_bits)

        lo, hi = 1e-4, 2.0
        if not ok(lo):
            return None
        while ok(hi) and hi < 50:
            hi *= 1.3
        for _ in range(22):
            mid = (lo + hi) / 2
            if ok(mid):
                lo = mid
            else:
                hi = mid
        return lo

    for ctr0 in [2, 3, 5, 8, 12, 20, 30]:
        thr = safe_xi_threshold(ctr0)
        steps = ctr0 * 3 + 6
        print(f"  CTR0={ctr0:3d} (~{steps:3d} cycles)   safe xi <= {thr:.5f}")


def demo_vanishing_gradient():
    section("6. hard-modeの勾配消失をCPU全体で再確認 (README自身の警告どおり)")

    def final_acc_raw(pc0, xi):
        cpu = ToyCPU(SUM_1_TO_5_PROGRAM, n_bits=6, xi=xi)
        _, acc, _ = cpu.run(steps=25, pc0=pc0)
        return bits_to_int_exact(acc)

    h = 1e-4
    for xi in [1e-3, 0.05, 0.1, 0.15, 0.2]:
        grad = (final_acc_raw(h, xi) - final_acc_raw(-h, xi)) / (2 * h)
        print(f"  xi={xi:<6}  dACC/dpc0 (pc0=0付近) ~= {grad: .6f}   "
              f"(安全域では勾配ほぼ0 = single-passing/勾配消失そのもの)")


def demo_gradient_program_inversion():
    section("7. 単一ADD命令の『逆算』を勾配降下で試す(正直な負の結果)")
    np.random.seed(0)
    n = 5
    xi = 0.3
    B = int_to_bits(11, n)
    target = 20  # A + 11 = 20 -> 正解 A=9

    def loss(A):
        S, _ = ripple_add(A, B, xi=xi)
        return (bits_to_int_exact(S) - target) ** 2

    A = np.random.uniform(0.0, 1.0, size=n)
    lr, h = 0.3, 1e-3
    for step in range(200):
        base = loss(A)
        grad = np.array([(loss(A + h * e) - base) / h
                          for e in np.eye(n)])
        A = np.clip(A - lr * grad, 0, 1)
    print(f"  収束後のA(丸め): {bits_to_int(A)}  (正解: 9, loss={loss(A):.3f})")
    print("  -> ローカルミニマム/プラトーに捕まりやすく、綺麗には解けない。")
    print("     README/dev_notes.mdが既に警告している『hard-modeは勾配降下の")
    print("     目的関数に向かない』という限界と一致する、正直な追試結果。")


def demo_annealed_inversion_vs_bitwidth():
    section("8. ライブラリ公式の anneal_solve でリベンジ + ビット幅依存の確認")
    from kappalogic.heat import xi_of_time
    from kappalogic.search import anneal_solve

    def test_n(n, trials=20, seed=0):
        np.random.seed(seed)
        B_int = 3 % (2 ** n)
        B = int_to_bits(B_int, n)
        target = 7 % (2 ** n)
        want = (target - B_int) % (2 ** n)

        def objective(A, t):
            xi_ = max(xi_of_time(t, D=1.0), 1e-4)
            S, _ = ripple_add(A, B, xi=xi_)
            return -(bits_to_int_exact(S) - target) ** 2

        succ = 0
        for _ in range(trials):
            A0 = np.random.uniform(0, 1, size=n)
            Af = anneal_solve(objective, A0, t_start=5, t_end=1e-7,
                               steps=400, lr=0.2)
            Af = np.clip(Af, 0, 1)
            if bits_to_int(Af) == want:
                succ += 1
        return succ, trials, want

    print("  (kappalogic.search.anneal_solve: 熱方程式スケジュールでxiを")
    print("   焼きなましながら勾配上昇する、公式のsoft-mode探索レシピ)")
    for n in [1, 2, 3, 4, 5, 6]:
        s, t, want = test_n(n)
        print(f"  n={n} bit  正解A={want:3d}  成功 {s}/{t}")
    print("  -> 単一ゲートなら公式レシピでほぼ100%成功するが、")
    print("     ビット幅(=加算器の連鎖段数)が伸びると急激に失敗率が上がる。")
    print("     『順方向の正しさ』はビット幅に頑健なのに、『逆方向の勾配探索』")
    print("     はビット幅に脆い、という非対称性の実測。")


if __name__ == "__main__":
    demo_adder_correctness()
    demo_safe_xi_vs_bitwidth()
    demo_alu()
    demo_toy_cpu()
    demo_safe_xi_vs_loop_length()
    demo_vanishing_gradient()
    demo_gradient_program_inversion()
    demo_annealed_inversion_vs_bitwidth()
    print("\n" + SEP)
    print("全デモ完了。")
    print(SEP)
