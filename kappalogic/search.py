"""
kappalogic.search
====================
"hardモード"(core.pyのreg/AND/OR)は境界でも遠方でも微分が0になる
「盆地型」構造を持つため、これを直接勾配降下の目的関数にすると
ほぼ必ず勾配消失で動けなくなることを実験で確認した(dev_notes.md参照:
reg(x)=tanh(x/xi)^2 は x=0 でも d(reg)/dx=0 になる。sgn(x)=tanh(x/xi) は
x=0で微分最大)。

このモジュールは、探索(勾配ベースの最適化)専用の"softモード"を提供する:
  - soft_gt/soft_eq などは reg で包まず sgn や heat.gaussian_match を
    直接使うので、境界に微分の「穴」が空かない。
  - anneal_solve は heat.xi_of_time による物理的に妥当なスケジュール
    (xi ∝ sqrt(t)) でsoftモードの目的関数を最適化し、最後にhardモードの
    厳密な指示関数(core.AND等)で結果を検算する。

位置づけ: これはNP完全問題を魔法のように解く道具ではない。悪い初期値からでも
勾配が働くようにするための「探索用の緩和」であり、収束するかどうかは
問題の凸性・初期値に依存する(通常の非凸最適化と同じ限界を持つ)。
"""
import numpy as np
from .core import sgn, DEFAULT_XI
from .heat import xi_of_time, gaussian_match


def soft_gt(a, b, xi=1.0):
    """A>Bに近いほど+1、A<Bに近いほど-1 に近づく、境界で微分が消えない比較。"""
    return sgn(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), xi)


def soft_eq(a, b, t=1.0, D=1.0):
    """A=Bに近いほど1に近づく、なだらかな一致度報酬(熱核=ガウス関数ベース)。"""
    return gaussian_match(a, b, t, D)


def soft_or(vals, xi=1.0):
    """
    「どれか一つでも満たされていればOK」を勾配が働く形で近似するsoft-OR
    (softmax/logsumexpの数値安定版)。値が大きい成分ほど支配的に効く。
    xi->0でmax(vals)に収束(鋭いOR)、xiが大きいほどなだらか。
    """
    vals = np.asarray(vals, dtype=float) / xi
    m = np.max(vals)
    return xi * (m + np.log(np.sum(np.exp(vals - m))))


def soft_and(vals, xi=1.0):
    """soft-AND: 「全部満たされてほしい」= -soft_or(-vals, xi)"""
    return -soft_or([-v for v in vals], xi)


def l2_penalty(x, coeff=0.02):
    """
    探索変数の暴走を防ぐL2正則化項(objective関数の中で引き算して使う)。
    実験の結果:
      - coeff=0で3-SATの探索をすると変数が±140程度まで暴走した(充足はする)。
      - coeff=0.05程度で±5程度まで抑えられ、充足性も保たれた。
      - coeff=0.2まで強めると、SAT式の中で真偽どちらでもよい("don't care")
        変数がちょうど0に張り付いてしまい、真偽判定が境界線上で曖昧になる
        ケースを確認した。強すぎる正則化は避けるのが無難(目安: 0.01〜0.05)。
    """
    x = np.asarray(x, dtype=float)
    return coeff * np.sum(x ** 2)


def is_dont_care_variable(x, i, clauses_fn, xi_final=1e-4, threshold=0.05):
    """
    L2正則化がゼロに張り付かせた変数が、本当に「どちらの真偽値でも
    充足度に影響しない("don't care")」変数かどうかを検出する。

    判定方法: x[i]の符号だけを反転させても、充足度(clauses_fnの値、
    soft_or/soft_andのみで組んだ論理式の値で、l2_penaltyを含まないもの)が
    ほとんど変わらないなら、その変数はdon't careとみなす。

    手動で作った「全変数が本質的」なケースで誤検出しないこと、
    3-SATデモの"don't care"変数(x1)を正しく検出できることを確認済み。

    clauses_fn: callable(x: np.ndarray, xi: float) -> float
        l2_penaltyを含まない、充足度のみを返す目的関数。
    """
    x = np.asarray(x, dtype=float)
    x_flipped = x.copy()
    x_flipped[i] = -x_flipped[i]
    before = clauses_fn(x, xi_final)
    after = clauses_fn(x_flipped, xi_final)
    return abs(before - after) < threshold


def find_dont_care_variables(x, clauses_fn, xi_final=1e-4, threshold=0.05):
    """
    is_dont_care_variableを全変数に適用し、don't careと判定されたインデックスの
    リストを返す。L2正則化で0に張り付いた変数を「バグ」ではなく「自由変数」
    として報告するための実用的なヘルパー。
    """
    x = np.asarray(x, dtype=float)
    return [i for i in range(len(x)) if is_dont_care_variable(x, i, clauses_fn, xi_final, threshold)]


def numeric_grad(f, x, h=1e-6):
    """中心差分による数値勾配。xはnp.array。"""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        grad[i] = (f(xp) - f(xm)) / (2 * h)
    return grad


def anneal_solve(objective, x0, t_start=100.0, t_end=1e-6, steps=500, lr=0.3, D=1.0):
    """
    heat.xi_of_time(t,D) = 2*sqrt(D*t) に従って t を t_start から t_end まで
    (指数的に、つまりxiは sqrt(t)に比例して)減らしながら objective(x, t) を
    勾配上昇で最大化する。

    objective: callable(x: np.ndarray, t: float) -> float
        softモードの目的関数。呼び出し側が t を受け取って
        soft_gt(..., xi=xi_of_time(t))やsoft_eq(...,t=t)などに使う。
    x0: 初期値 (list可)
    戻り値: 最適化後の x (np.ndarray)
    """
    x = np.asarray(x0, dtype=float).copy()
    t_schedule = np.geomspace(t_start, t_end, steps)
    for t in t_schedule:
        grad = numeric_grad(lambda xx: objective(xx, t), x)
        x = x + lr * grad
    return x
