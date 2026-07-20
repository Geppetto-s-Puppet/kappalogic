"""
kappa_cpu.py
============
kappalogic (tanh(x/xi) 一本槍の微分可能離散論理ライブラリ) の
AND/OR/XOR/eq/gt だけを部品にして、"数式上のCPU" ―― ripple-carry加算器
から ALU、さらに PC・ACC・CTR を持つ極小のフォン・ノイマン型トイCPU まで
を組み上げる実験コード。

設計方針(kappalogicの流儀に倣う):
  - if文を一切使わず、すべての分岐・選択を eq()/AND_n/OR_n による
    "重み付き和のマルチプレクサ" として書く。
  - ビットは 0.0/1.0 近傍の実数として表現し、xi->0 で厳密な2進数の
    意味論に収束することを数値的に確認する。
  - PC(プログラムカウンタ)自体も実数値のレジスタとして扱い、
    命令フェッチを eq(pc, i) による重み付き和として実装する
    (= "ソフトなフォン・ノイマン・アーキテクチャ")。
"""
import numpy as np
from kappalogic.core import AND, OR, XOR, NOT, eq, gt, OR_n, DEFAULT_XI

# ---------------------------------------------------------------------
# 1. ビット <-> 整数変換 (テスト・デバッグ専用。回路そのものには使わない)
# ---------------------------------------------------------------------

def int_to_bits(x, n):
    """非負整数 x を n bit (LSBが index 0) の 0.0/1.0 配列にする。mod 2^n。"""
    x = int(x) % (2 ** n)
    return np.array([(x >> i) & 1 for i in range(n)], dtype=float)


def bits_to_int(bits):
    """0/1近傍の実数配列を最も近い整数ビット列とみなして整数に戻す(丸め込み)。"""
    rounded = np.round(np.clip(bits, 0, 1)).astype(int)
    return int(sum(b << i for i, b in enumerate(rounded)))


def bits_to_int_exact(bits):
    """丸めずに実数のまま重み付き和を取る版。ξ>0のときの"ぼやけ具合"を見るのに使う。"""
    return sum(float(b) * (2 ** i) for i, b in enumerate(bits))


# ---------------------------------------------------------------------
# 2. 全加算器 -> n bit リップルキャリー加算器 (kappalogicのAND/XOR/ORそのもの)
# ---------------------------------------------------------------------

def full_adder(a, b, cin, xi=DEFAULT_XI):
    """半加算器2段 + キャリー選択。教科書の全加算器のブール式を
    そのままAND/XOR/ORに置き換えただけ。"""
    s1 = XOR(a, b, xi)
    c1 = AND(a, b, xi)
    s2 = XOR(s1, cin, xi)
    c2 = AND(s1, cin, xi)
    cout = OR(c1, c2, xi)
    return s2, cout


def ripple_add(A, B, cin0=0.0, xi=DEFAULT_XI):
    """A,B: 長さnのビット配列(LSB first)。桁上がりを左から右へ伝播させる
    (これが命題9/10の"naive fold"と全く同じ構造 ―― n段連鎖するAND/ORの
    融合安全性の問題がここでもそのまま起きる、というのが今回の発見)。"""
    n = len(A)
    S = np.zeros(n)
    cin = cin0
    for i in range(n):
        S[i], cin = full_adder(A[i], B[i], cin, xi)
    return S, cin  # cout(オーバーフロー桁)は mod 2^n として捨てる


def invert_bits(bits, xi=DEFAULT_XI):
    return np.array([NOT(b, xi) for b in bits])


def ripple_sub(A, B, xi=DEFAULT_XI):
    """A - B (mod 2^n)。2の補数: A + (~B) + 1。"""
    S, _ = ripple_add(A, invert_bits(B, xi), cin0=1.0, xi=xi)
    return S


def bitwise(op, A, B, xi=DEFAULT_XI):
    return np.array([op(a, b, xi) for a, b in zip(A, B)])


def is_zero_bits(bits, xi=DEFAULT_XI):
    """全ビットが0に近いか(1に近いほど"全部0"、と定義)。
    OR_n(*bits) は「少なくとも1本が非0」の検出器なので、1から引くと"全部0"検出器になる。"""
    any_nonzero = OR_n(*bits, xi=xi)
    return 1.0 - any_nonzero


# ---------------------------------------------------------------------
# 3. n bit ALU: opcodeをeq()でデコードし、Σ eq(op,k)*candidate_k で選択する
#    ―― if文を一切使わないマルチプレクサ
# ---------------------------------------------------------------------

ALU_ADD, ALU_SUB, ALU_AND, ALU_OR, ALU_XOR, ALU_NOT_A = range(6)


def alu(opcode, A, B, xi=DEFAULT_XI):
    """opcode: 実数値(0..5に近い値)。A,B: 長さnのビット配列。
    戻り値: 長さnのビット配列(要素ごとに重み付き和で選択済み)。"""
    n = len(A)
    candidates = {
        ALU_ADD: ripple_add(A, B, xi=xi)[0],
        ALU_SUB: ripple_sub(A, B, xi=xi),
        ALU_AND: bitwise(AND, A, B, xi),
        ALU_OR: bitwise(OR, A, B, xi),
        ALU_XOR: bitwise(XOR, A, B, xi),
        ALU_NOT_A: invert_bits(A, xi),
    }
    weights = {k: eq(opcode, k, xi) for k in candidates}
    out = np.zeros(n)
    for k, cand in candidates.items():
        out = out + weights[k] * cand
    return out


# ---------------------------------------------------------------------
# 4. トイCPU: PC / ACC / CTR を持つ、"ソフトなフォン・ノイマン機械"
#
#    命令フェッチもデコードも実行も、すべて eq() による重み付き和の
#    マルチプレクサだけで書く。PC自身も実数レジスタ(1個の浮動小数点数)
#    として扱うので、原理上は「命令列の上で勾配降下する」ことすら
#    可能になる(この点はデモの最後で試す)。
#
#    命令セット (op, field):
#      OP_ADD        : ACC = ACC + CTR
#      OP_DEC_CTR    : CTR = CTR - 1
#      OP_JNZ  target: CTR != 0 なら pc = target, さもなければ pc += 1
#      OP_HALT       : pc をその場に留める(以後何もしない)
#      OP_LOADI_CTR v: CTR = v (即値)
#      OP_LOADI_ACC v: ACC = v (即値)
# ---------------------------------------------------------------------

(OP_ADD, OP_DEC_CTR, OP_JNZ, OP_HALT,
 OP_LOADI_CTR, OP_LOADI_ACC) = range(6)

N_OPS = 6


class ToyCPU:
    """program: [(opcode:int, field:int), ...] のリスト。
    field の意味は opcode依存 (JNZ='jump先pc', LOADI_*='即値', それ以外は無視)。"""

    def __init__(self, program, n_bits=6, xi=DEFAULT_XI):
        self.program = program
        self.n_bits = n_bits
        self.xi = xi
        self.n_instr = len(program)
        # 命令ROM: 各フィールドを実数配列として持つ("配線"なのでkappalogicの
        # ゲートを通さないハードコード定数。実機のマイクロコードROMに相当)。
        self.rom_opcode = np.array([op for op, _ in program], dtype=float)
        self.rom_field = np.array([f for _, f in program], dtype=float)
        # LOADI系だけは即値をビット列に変換して持っておく(定数配線)。
        self.rom_ctr_bits = np.array([
            int_to_bits(f, n_bits) if op == OP_LOADI_CTR else np.zeros(n_bits)
            for op, f in program
        ])
        self.rom_acc_bits = np.array([
            int_to_bits(f, n_bits) if op == OP_LOADI_ACC else np.zeros(n_bits)
            for op, f in program
        ])

    def fetch(self, pc):
        """pc (実数) から eq()の重み付き和で opcode/field/即値ビットを"フェッチ"する。
        pcが整数値ちょうどなら対応する命令の値にほぼ厳密に収束する。"""
        w = np.array([eq(pc, i, self.xi) for i in range(self.n_instr)])
        active_opcode = float(np.dot(w, self.rom_opcode))
        active_field = float(np.dot(w, self.rom_field))
        active_ctr_const = w @ self.rom_ctr_bits   # (n_instr,) @ (n_instr,n) -> (n,)
        active_acc_const = w @ self.rom_acc_bits
        return active_opcode, active_field, active_ctr_const, active_acc_const

    def step(self, pc, acc, ctr):
        xi = self.xi
        op, field, ctr_const, acc_const = self.fetch(pc)

        ctr_is_zero = is_zero_bits(ctr, xi)  # 現在のCTRで判定(更新前)

        # --- ACC の次状態 (opcodeごとの候補をeq()で選択) ---
        acc_candidates = {
            OP_ADD: ripple_add(acc, ctr, xi=xi)[0],
            OP_DEC_CTR: acc,
            OP_JNZ: acc,
            OP_HALT: acc,
            OP_LOADI_CTR: acc,
            OP_LOADI_ACC: acc_const,
        }
        # --- CTR の次状態 ---
        ctr_candidates = {
            OP_ADD: ctr,
            OP_DEC_CTR: ripple_sub(ctr, int_to_bits(1, self.n_bits), xi=xi),
            OP_JNZ: ctr,
            OP_HALT: ctr,
            OP_LOADI_CTR: ctr_const,
            OP_LOADI_ACC: ctr,
        }
        # --- PC の次状態 ---
        jnz_next = ctr_is_zero * (pc + 1.0) + (1.0 - ctr_is_zero) * field
        pc_candidates = {
            OP_ADD: pc + 1.0,
            OP_DEC_CTR: pc + 1.0,
            OP_JNZ: jnz_next,
            OP_HALT: pc,
            OP_LOADI_CTR: pc + 1.0,
            OP_LOADI_ACC: pc + 1.0,
        }

        weights = {k: eq(op, k, xi) for k in range(N_OPS)}

        new_acc = sum(weights[k] * acc_candidates[k] for k in range(N_OPS))
        new_ctr = sum(weights[k] * ctr_candidates[k] for k in range(N_OPS))
        new_pc = sum(weights[k] * pc_candidates[k] for k in range(N_OPS))
        return new_pc, new_acc, new_ctr

    def run(self, steps, pc0=0.0, acc0=None, ctr0=None, trace=False):
        acc = np.zeros(self.n_bits) if acc0 is None else acc0
        ctr = np.zeros(self.n_bits) if ctr0 is None else ctr0
        pc = pc0
        history = []
        for t in range(steps):
            if trace:
                history.append((pc, bits_to_int(acc), bits_to_int(ctr)))
            pc, acc, ctr = self.step(pc, acc, ctr)
        if trace:
            history.append((pc, bits_to_int(acc), bits_to_int(ctr)))
            return pc, acc, ctr, history
        return pc, acc, ctr
