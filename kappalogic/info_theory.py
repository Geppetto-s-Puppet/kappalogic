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
