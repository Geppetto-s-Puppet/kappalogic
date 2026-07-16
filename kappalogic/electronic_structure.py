"""
kappalogic.electronic_structure
===================================
密度汎関数理論(DFT)への接続。

発見: フェルミ・ディラック分布(有限温度の電子占有数)
    f(eps) = 1/(1+exp((eps-mu)/kT))
は代数的に 1/(1+e^{-y}) = (1+tanh(y/2))/2 という恒等式そのものなので、
このライブラリの sgn(x,xi)=tanh(x/xi) を使って
    occupation(eps,mu,xi=2*kT) = (1+sgn(mu-eps,xi))/2
と書けば、フェルミ・ディラック分布と機械精度で完全に一致する
(xi = 2*kT が正しい対応関係。検証済み)。

これは、DFT計算コード(VASP, Quantum ESPRESSOなど)で金属を扱う際に
使われる「電子スメアリング」という実務的な数値手法と、このライブラリの
「hard(T=0のステップ関数)/soft(xi>0でなだらか)の二重性」が
数学的に同一であることを意味する。1次元タイトバインディング鎖で、
xi(スメアリング幅)を0に近づけると全電子エネルギーがT=0の厳密値に
収束することを数値実験で確認した。

【訂正】当初、有限鎖(N=200)での収束次数が教科書的なO(kT^2)
(ゾンマーフェルト展開)より速く見える現象を確認したが、これは
有限系サイズ(離散準位)+二分探索の精度限界による数値アーティファクト
だった。連続体の状態密度(tight_binding_dos_continuum)で高精度に
(scipy.integrate.quad + brentq)再計算したところ、xiを半分にする
たびの誤差比が4.00で安定し、標準的なO(xi^2)であることを確認した
(dev_notes.md参照)。
"""
import numpy as np
from .core import sgn


def fermi_occupation(eps, mu, kT):
    """
    フェルミ・ディラック占有数。sgn(x,xi=2*kT)を使って実装。
    kT->0でハードなステップ関数(T=0の占有)に収束する。
    """
    xi = 2 * kT
    return (1 + sgn(mu - eps, xi)) / 2


def tight_binding_chain_energies(N, t_hop=1.0):
    """1次元タイトバインディング鎖(周期境界条件)のエネルギー準位。eps_k=-2*t*cos(k)"""
    j = np.arange(N)
    k = 2 * np.pi * j / N
    return -2 * t_hop * np.cos(k)


def find_chemical_potential(eps, target_occupation, kT, lo=-10.0, hi=10.0, iters=60):
    """占有数の合計がtarget_occupationになるようmuを二分探索で求める。"""
    for _ in range(iters):
        mid = (lo + hi) / 2
        occ = np.sum(fermi_occupation(eps, mid, kT))
        if occ < target_occupation:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tight_binding_dos_continuum(eps, t_hop=1.0):
    """1次元タイトバインディングの厳密な連続体状態密度: g(eps)=1/(pi*sqrt(4t^2-eps^2))"""
    return 1.0 / (np.pi * np.sqrt(4 * t_hop ** 2 - eps ** 2 + 1e-300))


def continuum_filling(mu, xi, t_hop=1.0):
    """連続体極限でのフィリング(占有数の積分)。scipy.integrate.quadで高精度計算。"""
    from scipy import integrate
    f = lambda eps: tight_binding_dos_continuum(eps, t_hop) * fermi_occupation(eps, mu, xi / 2)
    val, _ = integrate.quad(f, -2 * t_hop + 1e-9, 2 * t_hop - 1e-9, limit=200)
    return val


def continuum_energy(mu, xi, t_hop=1.0):
    """連続体極限での全電子エネルギー(1電子あたり)。有限系サイズ効果を排除した検証用。"""
    from scipy import integrate
    f = lambda eps: eps * tight_binding_dos_continuum(eps, t_hop) * fermi_occupation(eps, mu, xi / 2)
    val, _ = integrate.quad(f, -2 * t_hop + 1e-9, 2 * t_hop - 1e-9, limit=200)
    return val


def continuum_find_mu(target, xi, t_hop=1.0):
    """連続体極限での化学ポテンシャルをbrentqで高精度に求める(有限系の二分探索精度限界を回避)。"""
    from scipy import optimize
    g = lambda mu: continuum_filling(mu, xi, t_hop) - target
    return optimize.brentq(g, -2 * t_hop, 2 * t_hop, xtol=1e-12)


def total_energy_smeared(eps, target_occupation, kT):
    """スメアリング(有限kT)を使った全電子エネルギーの計算(有限鎖版、DFTでの標準的な手法)。"""
    mu = find_chemical_potential(eps, target_occupation, kT)
    occ = fermi_occupation(eps, mu, kT)
    return np.sum(eps * occ)
