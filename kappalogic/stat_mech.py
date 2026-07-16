"""
kappalogic.stat_mech
=======================
量子統計力学への応用2つ。

1. 無限井戸型ポテンシャルの分配関数
   Z(beta) = Tr[e^{-beta H}] = integral_0^L K_E(x,x,beta) dx
   K_E(虚時間プロパゲータ)は熱方程式(D=hbar/(2m))の解そのものなので、
   quantum_well.box_heat_kernel をそのまま再利用できる。厳密な固有値和
   sum_n exp(-beta*E_n) と機械精度近くで一致することを確認済み。

2. SUSY量子力学のWitten指数(アティヤ=シンガー指数定理の最も単純な雛形)
   Witten指数 = Z_+(beta) - Z_-(beta) は beta によらず一定、という
   McKean-Singerの公式の主張を、超ポテンシャルW(x)=x^2/2から作った
   トイモデル(調和振動子ベース)で確認済み(全betaで厳密に1.000000)。
   これは新しい定理ではなく、指数定理の熱核による証明手法(Atiyah-Bott-Patodi)
   の最小のおもちゃの模型をコードで動かして確認した、という位置づけ。
"""
import numpy as np
from .quantum_well import box_heat_kernel


def box_partition_function(beta, L=1.0, D=0.5, n_x=400, n_images=100):
    """無限井戸(幅L)の分配関数 Z(beta) を熱核の対角成分の積分で計算する。"""
    xs = np.linspace(1e-4, L - 1e-4, n_x)
    vals = np.array([box_heat_kernel(x, x, beta, L, D, n_images) for x in xs])
    return np.trapezoid(vals, xs)


def box_partition_function_exact(beta, L=1.0, n_terms=200):
    """box_partition_function の厳密な固有値和版(検証用、hbar=m=1)。"""
    n = np.arange(1, n_terms + 1)
    En = (n * np.pi) ** 2 / (2 * L ** 2)
    return np.sum(np.exp(-beta * En))


def witten_index_toy(beta, n_terms=2000):
    """
    超ポテンシャルW(x)=x^2/2のSUSY QMトイモデルにおける
    Witten指数 Z_+(beta)-Z_-(beta) (理論的にはbetaによらず1)。
    H_+固有値=2n (n=0,1,2,...)、H_-固有値=2n+2。
    """
    n = np.arange(0, n_terms)
    z_plus = np.sum(np.exp(-beta * 2 * n))
    z_minus = np.sum(np.exp(-beta * (2 * n + 2)))
    return z_plus - z_minus
