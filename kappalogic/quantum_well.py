"""
kappalogic.quantum_well
=========================
応用21で触れられていた「無限井戸型ポテンシャル」との接続。

発想の経緯:
    heat.py の熱核(ガウス関数)は無限に広い空間の話だった。
    「両端で0になる有限区間[0,L]」の熱核は、鏡像法(電磁気学で
    接地導体板に電荷を置いたときの鏡像と同じ発想)で作れる:
    無限直線のガウス核を、境界の外側に符号反転させながら
    無限に並べて足し合わせると、ちょうどx=0,Lで0になる。

    これを固有関数展開 sum_n sin(nπx/L)sin(nπx'/L)exp(-Dn²π²t/L²)
    と比較して、両者が機械精度で一致することを確認した(diff ~ 1e-16)。

    さらに「ウィック回転」(D -> iħ/(2m), t -> T)を施すと、熱核は
    自由粒子の量子力学的プロパゲータに変わる。同じ鏡像法をこちらに
    適用すれば「無限井戸型ポテンシャル」の厳密なプロパゲータになるはず、
    と考えて実験した。

    最初は鏡像の和が発散気味で固有関数展開と全く合わなかった
    (熱核と違って量子プロパゲータは位相が振動するだけで振幅が
    減衰しないため、鏡像の和が絶対収束しない)。
    標準的な「εプリスクリプション」(T -> T - iε という微小な収束因子を
    加える、経路積分の教科書でおなじみの正則化)を導入したところ、
    固有関数展開と機械精度(diff ~ 1e-13)で一致した。

    結論: この理論の「箱の指示関数」(choc_bar_sourceで使ったAND/gt/lt)と
    「熱核」(heat.py)を鏡像法で組み合わせれば、無限井戸型ポテンシャルの
    量子プロパゲータを、既知の固有関数展開を一切使わずに構成できる。
"""
import numpy as np


def _gaussian(x, xp, t, D):
    return np.exp(-(x - xp) ** 2 / (4 * D * t)) / np.sqrt(4 * np.pi * D * t)


def box_heat_kernel(x, xp, t, L, D=1.0, n_images=200):
    """
    [0,L]、両端ディリクレ境界(u=0)の熱核。鏡像法による構成。
    固有関数展開 sum_n (2/L)sin(nπx/L)sin(nπx'/L)exp(-D(nπ/L)^2 t) と
    機械精度で一致することを確認済み(tests参照)。
    """
    s = 0.0
    for n in range(-n_images, n_images + 1):
        s += _gaussian(x, 2 * n * L + xp, t, D)
        s -= _gaussian(x, 2 * n * L - xp, t, D)
    return s


def box_heat_kernel_eigen(x, xp, t, L, D=1.0, n_terms=500):
    """box_heat_kernel の固有関数展開版(検証・比較用)。"""
    n = np.arange(1, n_terms + 1)
    return np.sum((2 / L) * np.sin(n * np.pi * x / L) * np.sin(n * np.pi * xp / L)
                  * np.exp(-D * (n * np.pi / L) ** 2 * t))


def _free_propagator(x, xp, T, hbar=1.0, m=1.0):
    return np.sqrt(m / (2j * np.pi * hbar * T)) * np.exp(1j * m * (x - xp) ** 2 / (2 * hbar * T))


def infinite_well_propagator(x, xp, T, L, hbar=1.0, m=1.0, eps=1e-3, n_images=2000):
    """
    無限井戸型ポテンシャル(幅L、両端で波動関数0)の量子プロパゲータ。
    box_heat_kernelと同じ鏡像法を、ウィック回転
    (D -> i*hbar/(2m), t -> T) した自由粒子プロパゲータに適用したもの。

    eps: 収束因子(T -> T - i*eps)。経路積分で標準的に使われる
    "iεプリスクリプション"。これがないと鏡像の和が収束しない
    (振幅が減衰せず位相だけ振動するため)。
    """
    Tc = T - 1j * eps
    s = 0j
    for n in range(-n_images, n_images + 1):
        s += _free_propagator(x, 2 * n * L + xp, Tc, hbar, m)
        s -= _free_propagator(x, 2 * n * L - xp, Tc, hbar, m)
    return s


def infinite_well_propagator_eigen(x, xp, T, L, hbar=1.0, m=1.0, eps=1e-3, n_terms=2000):
    """infinite_well_propagator の固有関数展開版(検証・比較用)。
    psi_n(x) = sqrt(2/L) sin(n*pi*x/L), E_n = n^2*pi^2*hbar^2/(2*m*L^2)
    image法と同じeps減衰(T -> T-i*eps)を適用しないと、位相が振動するだけで
    振幅が減衰せず、有限項での打ち切り誤差が大きくなる(数値実験で確認済み)。
    """
    Tc = T - 1j * eps
    n = np.arange(1, n_terms + 1)
    psin_x = np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
    psin_xp = np.sqrt(2 / L) * np.sin(n * np.pi * xp / L)
    En = (n * np.pi) ** 2 * hbar ** 2 / (2 * m * L ** 2)
    return np.sum(psin_x * psin_xp * np.exp(-1j * En * Tc / hbar))
