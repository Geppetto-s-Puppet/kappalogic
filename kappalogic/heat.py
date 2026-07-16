"""
kappalogic.heat
=================
kernel="erf" を選んだときの sgn(x,xi,"erf") = erf(x/xi) は、
1次元熱方程式 u_t = D*u_xx を階段状の初期条件(x<0で0, x>0で1)から
時間tだけ発展させた厳密解
    u(x,t) = (1 + erf(x/(2*sqrt(D*t)))) / 2
とちょうど同じ形をしている(xi = 2*sqrt(D*t))。

これを有限差分法によるシミュレーションと比較し、最大誤差6e-5で
数値的に確認済み(dev_notes.md参照)。

このモジュールは、その対応関係を使って
  - xi <-> 時間 t の相互変換
  - 物理的に妥当な(=実際に拡散方程式が従う速さの)アニーリングスケジュール
    (xiはtに比例するのではなく sqrt(t) に比例して縮む)
  - ガウス核(熱核そのもの: デルタ関数初期条件からの厳密解)による
    "ソフトな一致度" 報酬関数
を提供する。
"""
import numpy as np
from scipy.special import erf


def xi_of_time(t, D=1.0):
    """拡散時間tに対応するxi。 xi = 2*sqrt(D*t)"""
    return 2 * np.sqrt(D * np.asarray(t, dtype=float))


def time_of_xi(xi, D=1.0):
    """xiに対応する拡散時間t。 t = xi^2 / (4D)"""
    xi = np.asarray(xi, dtype=float)
    return xi ** 2 / (4 * D)


def heat_step_profile(x, t, D=1.0):
    """
    階段状初期条件(x<0->0, x>0->1)を時間tだけ拡散させた温度分布の厳密解。
    sgn(x, xi_of_time(t,D), kernel="erf") と等価 ( u=(1+sgn)/2 )。
    """
    return 0.5 * (1 + erf(np.asarray(x, dtype=float) / xi_of_time(t, D)))


def heat_kernel_gaussian(x, t, D=1.0):
    """
    デルタ関数初期条件(1点に熱源)からの厳密解=熱核そのもの。
    G(x,t) = 1/sqrt(4*pi*D*t) * exp(-x^2/(4*D*t))
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-x ** 2 / (4 * D * t)) / np.sqrt(4 * np.pi * D * t)


def gaussian_match(a, b, t, D=1.0):
    """
    「a と b がどれだけ一致しているか」を熱核と同じ形のガウス関数で評価する
    "ソフトな一致度" 報酬 (0〜1、a=bで最大)。
    reg(a-b)(=硬い一致判定)とは違い、境界(a=b)でなだらかに減衰するので
    勾配ベースの探索の目的関数に使うのに向く。t->0でreg的な鋭さに近づく。
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    xi = xi_of_time(t, D)
    return np.exp(-(d / xi) ** 2)
