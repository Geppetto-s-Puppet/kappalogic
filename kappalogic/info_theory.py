"""
kappalogic.info_theory
=========================
de Bruijnの恒等式: 熱拡散(heat.py)でぼやけていく確率密度の
微分エントロピーS(t)の変化率は、フィッシャー情報量J(t)を使って
dS/dt = D*J(t) と書ける。ガウス分布(熱方程式の厳密解)で
数値的に確認済み(誤差~1e-10)。

これはheat.py(拡散・熱核)とAND/OR/reg系(論理・indicator関数)を
繋ぐ第三の軸として「情報理論的な量(エントロピー、フィッシャー情報量)」を
この枠組みの中に置く、という位置づけ。

v0.36追記(TODO.md G項「情報幾何への拡張」+ C項「ξの意味論統一」):
ガウス分布族N(mu,sigma^2)を(mu,sigma)というパラメータで見た
統計多様体に、フィッシャー情報計量(Fisher-Rao計量)を入れると

    ds^2 = dmu^2/sigma^2 + 2*dsigma^2/sigma^2

という計量になる(数値積分で確認済み、`fisher_metric_gaussian`)。
これは mu' := mu/sqrt(2) という単純な再スケールで、
**gauge.py(v0.14、命題7)で見つけた(x,xi)半平面の双曲計量
ds^2=(dx^2+dxi^2)/xi^2 と、定数倍(2倍)を除いて厳密に一致する**
(sympyで確認済み、差は厳密に0)。ガウス曲率もsympyで直接計算し、
K=-1/2(定数負曲率、標準双曲平面のK=-1のちょうど半分)であることを
確認した——古典的によく知られた事実(Fisher-Rao幾何のガウス族は
定数曲率-1/2の双曲平面)だが、kappalogic自身のgauge.pyの発見と
厳密に同じ計量が出てくることを確認したのは、このライブラリの
文脈では新しい統合。

これは「ξの意味論を統一する」(TODO C項)の候補をもう一つ増やす:
**ξを、ある検出対象の分布の"標準偏差"(Fisher-Rao幾何での尺度座標)
とみなせる**、という解釈。gauge.pyのアフィン群(並進+ディレーション)
が、まさにこの統計多様体の等長変換群(の一部)と同一視できる
ことになる——「xの並進+ξのディレーション」という対称性が、
「位置母数の並進+尺度母数のスケール変換」という統計学の自然な
対称性と、数式のレベルで同じものだったことになる。
"""
import numpy as np


def gaussian_variance(t, sigma2_0, D=1.0):
    """熱拡散で時間発展するガウス分布の分散: sigma^2(t) = sigma2_0 + 2*D*t"""
    return sigma2_0 + 2 * D * t


def differential_entropy_gaussian(t, sigma2_0, D=1.0):
    """ガウス分布の微分エントロピー S(t) = 0.5*log(2*pi*e*sigma^2(t))"""
    var = gaussian_variance(t, sigma2_0, D)
    return 0.5 * np.log(2 * np.pi * np.e * var)


def fisher_information_gaussian(t, sigma2_0, D=1.0):
    """ガウス分布のフィッシャー情報量 J(t) = 1/sigma^2(t)"""
    return 1.0 / gaussian_variance(t, sigma2_0, D)


def de_bruijn_check(t, sigma2_0, D=1.0, h=1e-6):
    """
    dS/dt (数値微分) と D*J(t) (de Bruijnの恒等式の予測値) を両方返す。
    理論上この2つは一致するはず。
    """
    dS_dt = (differential_entropy_gaussian(t + h, sigma2_0, D)
              - differential_entropy_gaussian(t - h, sigma2_0, D)) / (2 * h)
    predicted = D * fisher_information_gaussian(t, sigma2_0, D)
    return dS_dt, predicted


def fisher_metric_gaussian(mu, sigma, h=1e-5):
    """
    N(mu,sigma^2)のフィッシャー情報計量(mu,sigma)をパラメータとして
    数値的に計算する(スコア関数の期待値、scipy.integrate.quadで
    直接積分)。理論上 I_mumu=1/sigma^2, I_sigmasigma=2/sigma^2,
    I_musigma=0 になるはず(数値検証済み)。

    戻り値: (I_mumu, I_sigmasigma, I_musigma) のタプル。
    """
    from scipy import integrate

    def log_p(x, m, s):
        return -0.5 * np.log(2 * np.pi * s ** 2) - (x - m) ** 2 / (2 * s ** 2)

    def p(x):
        return np.exp(log_p(x, mu, sigma))

    def dlogp_dmu(x):
        return (log_p(x, mu + h, sigma) - log_p(x, mu - h, sigma)) / (2 * h)

    def dlogp_dsigma(x):
        return (log_p(x, mu, sigma + h) - log_p(x, mu, sigma - h)) / (2 * h)

    bound = mu - 20 * sigma, mu + 20 * sigma
    I_mumu, _ = integrate.quad(lambda x: dlogp_dmu(x) ** 2 * p(x), *bound)
    I_sigsig, _ = integrate.quad(lambda x: dlogp_dsigma(x) ** 2 * p(x), *bound)
    I_musig, _ = integrate.quad(lambda x: dlogp_dmu(x) * dlogp_dsigma(x) * p(x), *bound)
    return I_mumu, I_sigsig, I_musig


def fisher_metric_matches_gauge_hyperbolic_metric():
    """
    フィッシャー計量 ds^2 = dmu^2/sigma^2 + 2*dsigma^2/sigma^2 が、
    mu' := mu/sqrt(2) という再スケールのもとで、gauge.pyの
    双曲計量 ds^2=(dx^2+dxi^2)/xi^2 (命題7)の厳密に2倍になる
    ことをsympyで確認する。

    (x,xi) <-> (mu',sigma) という対応で、gauge.pyのアフィン群
    (x-並進+xiディレーション)が、フィッシャー情報幾何での
    (位置母数の並進+尺度母数のスケール変換)という統計学の
    自然な対称性と一致することになる。

    戻り値: 差(0になるはず、sympy式)。
    """
    import sympy as sp

    mu, sigma, dmu, dsigma = sp.symbols('mu sigma dmu dsigma', positive=True)
    ds2_fisher = dmu ** 2 / sigma ** 2 + 2 * dsigma ** 2 / sigma ** 2
    dmu_p = dmu / sp.sqrt(2)  # mu = sqrt(2)*mu', dmu = sqrt(2)*dmu_p
    standard_hyperbolic_times_2 = 2 * (dmu_p ** 2 + dsigma ** 2) / sigma ** 2
    return sp.simplify(ds2_fisher - standard_hyperbolic_times_2)


def fisher_gaussian_curvature():
    """
    フィッシャー情報計量(ガウス分布族)のガウス曲率を、Riemannテンソルを
    sympyで直接計算して求める。定数負曲率 K=-1/2 になるはず
    (標準双曲平面のK=-1のちょうど半分——gauge.pyの双曲計量に対して
    フィッシャー計量が定数2倍になっていることの帰結)。

    戻り値: ガウス曲率(sympy式、-1/2になるはず)。
    """
    import sympy as sp

    mu, sigma = sp.symbols('mu sigma', positive=True)
    coords = [mu, sigma]
    g = sp.Matrix([[1 / sigma ** 2, 0], [0, 2 / sigma ** 2]])
    ginv = g.inv()
    n = 2

    def christoffel(i, j, k):
        s = 0
        for l in range(n):
            s += ginv[i, l] * (sp.diff(g[l, j], coords[k]) + sp.diff(g[l, k], coords[j]) - sp.diff(g[j, k], coords[l]))
        return sp.simplify(s / 2)

    Gamma = [[[christoffel(i, j, k) for k in range(n)] for j in range(n)] for i in range(n)]

    def riemann(i, j, k, l):
        term1 = sp.diff(Gamma[i][j][l], coords[k])
        term2 = sp.diff(Gamma[i][j][k], coords[l])
        term3 = sum(Gamma[i][k][m] * Gamma[m][j][l] for m in range(n))
        term4 = sum(Gamma[i][l][m] * Gamma[m][j][k] for m in range(n))
        return sp.simplify(term1 - term2 + term3 - term4)

    Ricci = sp.zeros(n, n)
    for j in range(n):
        for l in range(n):
            Ricci[j, l] = sp.simplify(sum(riemann(i, j, i, l) for i in range(n)))

    Rscalar = sp.simplify(sum(ginv[j, l] * Ricci[j, l] for j in range(n) for l in range(n)))
    return sp.simplify(Rscalar / 2)
