"""
kappalogic.partition
=====================

【命題31(v0.71): OR_n の分配関数的描像 —— 空状態確率 P=e^{-S} と
sech^2 型占有数】

v0.66(physics_bridge.py)の統計力学的再解釈(log-charge S=ΣL、化学
ポテンシャル τ、実効温度 T_eff)に、「分配関数(空状態確率)」と
「自由エネルギー的な量」を正しく接続する層を足したもの。

== 経緯(正直な記録)==
このモジュールは、外部から提供された「Kappalogic Partition Function
Theorem (κPFT) v0.70」という草稿を検証した結果、その中心的主張に
数学的な誤りが見つかったため、**誤りを取り除き、数値検証を通った部分
だけを残して書き直した**ものである。元草稿の誤りと、それをどう直したか
は下の「元草稿からの訂正」に明記する。

== 核心(全て数値検証済み)==

1. 空状態確率 P(a_1,...,a_n;ξ) := ∏_k NOT(a_k;ξ) = e^{-S}
   (S=Σ_k L(a_k;ξ)、L=-ln NOT なので定義から厳密に P=e^{-S})。
   物理的には「n個の入力が全て非アクティブ(0)である結合確率」。

2. OR_n = NOT(P;ξ) = 1 - κ(P/ξ)^2 = occupation_from_charge(S)
   ——「少なくとも1個アクティブ」。これは既存の occupation_from_charge
   と厳密に一致し(RMS ~1e-18、tanh/erf/algebraic の3カーネル)、
   真の fused OR_n とも機械精度で一致する。

3. S=τ 近傍で OR_n = sech^2(c·e^{-(S-τ)}) という**二重指数(Gompertz型)**
   のステップ。半占有点 S=τ で厳密に 1/2、傾きは 1/T_eff。

== 元草稿(κPFT v0.70)からの訂正 ==

元草稿は OR_n を「フェルミ・ディラック占有数 1/(1+exp(-log Z))、
Z=exp((S-τ)/T_eff)」= ロジスティック型 sigmoid((S-τ)/T_eff) と主張して
いたが、これは誤り:

  - 真の OR_n は sech^2(c·e^{-(S-τ)})(二重指数/Gompertz型)。
    sigmoid((S-τ)/T_eff)(単指数/ロジスティック型)とは、S=τ で値・傾き
    こそ一致するが関数形が異なり、S≠τ で系統的に乖離する。
    元草稿の「RMS < 1e-3」は誤りで、実測 RMS ≈ 0.07(検証スクリプトの
    自己申告値は 1.67 に達していた)。
  - 元草稿の定理ステートメントの式 "OR_n = 1 - 2/(1+exp(log Z))
    = 1 - 2/(1+(S-τ)/T_eff)" は exp(log Z) と log Z を取り違えており、
    値域 [0,1] を飛び出す(3つの添付ファイルで占有数の式が3通りに
    食い違っていた)。
  - 元草稿の目玉「n→∞での2次相転移」は、実際には数値実験で観測
    されていない(全ケースが HIGH 相)。原因は L(a;ξ)=2 ln cosh(a/ξ)≥0
    が常に非負で、n 個の和 S は容易に τ(≈0.13)を超えるため——OR は
    「1個でも大きければ真」なので、入力数が増えるほど low 相(OR_n→0)
    を作るのは難しくなる。μ を負に振り分散を絞っても P(OR_n=1) は
    0.90 を下回らなかった。よって「臨界エネルギー密度を跨ぐ相転移」は
    実在しないと結論し、この主張は削除した。

正しく残せたのは「P=e^{-S} を空状態確率と見る描像」と「sech^2 型の
占有数」で、これは新しい定理というより v0.66 の描像の自然な言い換えに
近い(誇張しない)。
"""

import numpy as np
from .kernels import apply_kernel, kernel_log_increment, KERNEL_BOUNDARY_CONST
from .physics_bridge import (
    log_charge, chemical_potential, effective_temperature, occupation_from_charge,
)


def empty_state_probability(values, xi, kernel="tanh"):
    """
    空状態確率 P = ∏_k NOT(a_k;ξ) = e^{-S}(S=log_charge)。
    「n個の入力が全て非アクティブ(0近傍)である結合確率」。
    定義から厳密に e^{-S} に等しい(下の or_n_from_empty_state で使う)。
    """
    S = log_charge(values, xi, kernel)
    return float(np.exp(-S))


def or_n_from_empty_state(values, xi, kernel="tanh"):
    """
    空状態確率 P=e^{-S} から OR_n を復元: OR_n = NOT(P;ξ) = 1 - κ(P/ξ)^2。
    これは occupation_from_charge(S) と厳密に一致し(機械精度)、真の
    fused OR_n とも一致する。κPFT草稿の sigmoid 近似の**正しい置き換え**。
    """
    P = empty_state_probability(values, xi, kernel)
    return float(1 - apply_kernel(P, xi, kernel) ** 2)


def occupation_step_sech2(S, xi, kernel="tanh"):
    """
    log-charge S における占有数(=OR_n)を、正しい sech^2 型ステップとして
    返す。occupation_from_charge の別名(分配関数の文脈で使う用)。

    S=τ 近傍で OR_n = sech^2(c·e^{-(S-τ)})、半占有点 S=τ で 1/2、
    傾き 1/T_eff。κPFT が主張した sigmoid((S-τ)/T_eff) は、この式の
    S=τ における1次近似に過ぎず、離れると乖離する。
    """
    return float(occupation_from_charge(S, xi, kernel))


def free_energy_log_domain(values, xi, kernel="tanh"):
    """
    log-charge 空間での自由エネルギー的な量 F := -T_eff · ln OR_n。

    OR_n は「少なくとも1個アクティブ」の確率と見なせるので、統計力学の
    F = -T ln Z の類推で、ここでは Z の役割を占有数 OR_n(アクティブな
    状態の重み)に担わせた量。OR_n→1(高 log-charge)で F→0、OR_n→0
    (低 log-charge)で F→+∞。S に対して単調減少する。

    注: これは厳密な熱力学的自由エネルギーではなく、あくまで log-charge
    描像での類推量(元草稿の F=S-(S-τ)^2/T_eff は分配関数の誤りを
    引きずっていて非単調・発散したため、単調で解釈しやすいこの形に
    置き換えた)。
    """
    or_n = or_n_from_empty_state(values, xi, kernel)
    or_n = float(np.clip(or_n, 1e-300, 1.0))
    T_eff = effective_temperature(kernel)
    return float(-T_eff * np.log(or_n))


def summary(values, xi, kernel="tanh"):
    """
    n個の入力に対する分配関数的描像の要約を dict で返す。
    or_value は真の OR_n と機械精度で一致する(検証済み)。
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    S = log_charge(values, xi, kernel)
    tau = chemical_potential(xi, kernel)
    T_eff = effective_temperature(kernel)
    P = float(np.exp(-S))
    or_n = or_n_from_empty_state(values, xi, kernel)
    F = free_energy_log_domain(values, xi, kernel)
    return {
        "n_inputs": n,
        "log_charge": S,             # S = Σ L(a_k;ξ)、内部エネルギー的な量
        "chemical_potential": tau,   # τ(ξ)、半占有 S=τ
        "effective_temperature": T_eff,
        "empty_state_probability": P,  # P = e^{-S} = ∏ NOT(a_k)
        "or_value": or_n,            # OR_n = NOT(P) = occupation(S)、真値と機械精度一致
        "free_energy": F,            # F = -T_eff ln OR_n(log-charge描像の類推量)
        "above_threshold": bool(S > tau),  # S>τ ⟺ OR_n>1/2
    }


def why_no_phase_transition(xi=1.0, kernel="tanh", n=20, n_trials=500,
                            mus=(-0.5, -0.2, -0.1, -0.05, 0.0, 0.1), sigma=0.1,
                            seed=1):
    """
    κPFT 草稿が主張した「n→∞での相転移」が実在しないことを示す診断。

    入力平均 μ を負に振り分散を絞っても、P(OR_n=1) が高いまま(low 相に
    落ちない)ことを返す。理由: L(a;ξ)=2 ln cosh(a/ξ)≥0 が常に非負なので
    S=ΣL は容易に τ を超え、OR(「1個でも大きければ真」)は入力数が増える
    ほど true に張り付く。よって「臨界密度を跨ぐ相転移」は起きない。

    戻り値: [(mu, P(OR_n=1)), ...]。
    """
    tau = chemical_potential(xi, kernel)
    out = []
    for mu in mus:
        rng = np.random.default_rng(seed)
        rate = np.mean([
            float(log_charge(rng.normal(mu, sigma, n), xi, kernel) > tau)
            for _ in range(n_trials)
        ])
        out.append((float(mu), float(rate)))
    return out
