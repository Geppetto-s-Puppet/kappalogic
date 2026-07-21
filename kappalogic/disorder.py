"""
kappalogic.disorder
=====================
v0.83: 乱れ(ランダムネス)と Anderson 局在——「普遍性クラス」を睨む足がかり。

これまで kappalogic で見つけた相転移(命題37 の Curie-Weiss、BCS)は**全て
平均場**であり、臨界指数は β=1/2, γ=1 という平均場の値だった(命題38 の分岐分類)。
「本物の普遍性クラス」に踏み込むには、**揺らぎ/乱れが本質的に効く現象**が要る。
その最小の入口が **1次元 Anderson 局在**である。

Anderson 局在は平均場では捉えられない、乱れが主役の量子現象:
- 1次元では**任意の乱れ W>0 で全ての固有状態が指数局在**する(拡張状態が無い)。
  つまり 1D には金属-絶縁体転移が**無い**——これは命題33 と同じく「臨界点が
  無い」という否定的な事実で、正直に記録する。
- 弱乱れのバンド中心では局在長が **ζ ∝ W^{-2}** という普遍スケーリングに従う。
  これが本モジュールの検証アンカー(BCS の 1.7639、ローレンツ数 2.44e-8、
  臨界指数 β=1/2 と同じ役割)。

実測(下記関数で再現可能、一様鎖 N=2000、バンド中心 E≈0 の30状態の中央値):
    W    : 0.5     0.7     1.0     1.4     2.0
    ζ    : 353.3   239.0   122.0    57.3    31.7
    ζ·W² :  88.3   117.1   122.0   112.3   126.7
W≥0.7 に絞った両対数フィットで **ζ ∝ W^{-1.92}**(理論の弱乱れ極限 -2 と整合)。
最小の W=0.5 だけ下振れするのは ζ=353 が系のサイズ N=2000 に近づく**有限サイズ
効果**であり、乱れが弱いほど長い鎖が要る、という素直な理由。

kappalogic との関係(正直に): 局在そのものは kappalogic のカーネル固有の現象では
なく、標準的なタイトバインディング+乱れの物理である。本モジュールの役割は
(1) 乱れた鎖を optics.py / transport.py / matrix_backend.py(SP2)へ供給する土台、
(2) 平均場を超えた臨界現象への足がかり、(3) 「1D には転移が無い」という否定的
事実を命題33 と同じ誠実さで記録すること。
"""
import numpy as np


def disordered_chain(n_sites, disorder_W, rng, t=1.0, t_alt=None):
    """
    箱型乱れ(on-site energy が [-W/2, W/2] の一様分布)を持つ 1D タイトバインディング鎖。

    t: ホッピング。t_alt を与えると交替ホッピング(SSH、ギャップあり)になる。
    rng: numpy Generator(再現性のため呼び出し側が渡す)。
    戻り値: 実対称ハミルトニアン (n_sites, n_sites)。
    """
    H = np.zeros((n_sites, n_sites), dtype=float)
    for i in range(n_sites - 1):
        hop = t if (t_alt is None or i % 2 == 0) else t_alt
        H[i, i + 1] = H[i + 1, i] = -hop
    H[np.arange(n_sites), np.arange(n_sites)] = disorder_W * (rng.random(n_sites) - 0.5)
    return H


def inverse_participation_ratio(psi):
    """
    逆参加比 IPR = Σ|ψ_i|⁴ / (Σ|ψ_i|²)²。
    拡張状態なら ~1/N(N は系のサイズ)、局在状態なら ~1/ζ(サイズに依らない)。
    1/IPR が実効的な「広がり(局在長)」の目安になる。
    """
    p = np.abs(np.asarray(psi)) ** 2
    return float(np.sum(p ** 2) / np.sum(p) ** 2)


def localization_length(n_sites, disorder_W, rng, n_samples=6, n_states=30, t=1.0):
    """
    バンド中心(E≈0)近傍の固有状態の 1/IPR の**中央値**を局在長 ζ の推定値として返す。
    n_samples 個の乱れ配位について、各配位で |E| が小さい n_states 個を使う。

    注意: ζ が系のサイズ n_sites に近づくと有限サイズ効果で過小評価になる
    (弱い乱れほど長い鎖が必要)。
    """
    zs = []
    for _ in range(n_samples):
        H = disordered_chain(n_sites, disorder_W, rng, t=t)
        evals, evecs = np.linalg.eigh(H)
        mid = np.argsort(np.abs(evals))[:n_states]
        zs.extend(1.0 / inverse_participation_ratio(evecs[:, m]) for m in mid)
    return float(np.median(zs))


def localization_exponent(n_sites, disorder_values, rng, **kwargs):
    """
    局在長の乱れ依存 ζ ∝ W^{-p} の指数 p を両対数フィットで測る。
    弱乱れ・バンド中心の理論値は **p=2**(1D Anderson の普遍スケーリング)。

    disorder_values は有限サイズ効果を避けるため ζ << n_sites となる範囲を選ぶこと
    (例: n_sites=2000 なら W>=0.7)。戻り値: (p, [(W, ζ), ...])。
    """
    pairs = [(W, localization_length(n_sites, W, rng, **kwargs)) for W in disorder_values]
    Ws = np.array([w for w, _ in pairs], dtype=float)
    zs = np.array([z for _, z in pairs], dtype=float)
    p = -np.polyfit(np.log(Ws), np.log(zs), 1)[0]
    return float(p), pairs


def one_dimension_has_no_mobility_edge():
    """
    正直な否定的事実の記録: 1次元では任意の乱れ W>0 で**全ての**固有状態が局在し、
    拡張状態と局在状態を分ける移動度端(mobility edge)も金属-絶縁体転移も存在しない
    (Mott-Twose、スケーリング理論)。したがって 1D の乱れ系には臨界点が無い
    ——命題33(OR_N に臨界性なし)と同じ種類の、誠実に記録すべき否定的結果。
    真の Anderson 転移(臨界指数 ν≈1.57 の非平均場普遍性クラス)は 3D で現れる。
    """
    return False
