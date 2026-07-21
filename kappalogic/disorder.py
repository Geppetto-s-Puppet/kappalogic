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


# ---------------------------------------------------------------------------
# 命題39: 「横断」基準は平均場から RG へ持ち上がる —— 3D Anderson 転移
# ---------------------------------------------------------------------------
# 文献値(検証のアンカー): 箱型乱れ・バンド中心 E=0 の 3D Anderson 転移
ANDERSON_WC = 16.5        # 臨界乱れ
ANDERSON_LAMBDA_C = 0.576  # 臨界点での規格化局在長 Λ=λ/L
ANDERSON_NU = 1.57         # 臨界指数(非平均場!)


def _cross_section_hopping(L):
    """準1次元バーの L×L 断面のホッピング行列(周期境界)。"""
    n = L * L
    H = np.zeros((n, n))

    def idx(x, y):
        return (x % L) * L + (y % L)

    for x in range(L):
        for y in range(L):
            i = idx(x, y)
            for dx, dy in ((1, 0), (0, 1)):
                j = idx(x + dx, y + dy)
                H[i, j] -= 1.0
                H[j, i] -= 1.0
    return H


def transfer_matrix_lambda(L, disorder_W, n_slices, rng, energy=0.0, qr_every=8):
    """
    【命題39 の中心量】準1次元バー(断面 L×L、長さ n_slices)の**規格化局在長**
    Λ = λ/L を転送行列法(MacKinnon-Kramer)で計算する。

    ψ_{k+1} = (E - H_⊥ - ε_k)ψ_k - ψ_{k-1} を転送行列で反復し、QR 分解で
    Lyapunov 指数を得る。最小の正 Lyapunov 指数 γ から λ=1/γ、Λ=λ/L。

    **判定**: W<W_c なら L を増やすと Λ は**増加**(金属=拡張状態)、
    W>W_c なら**減少**(絶縁体=局在)。両者が入れ替わる点が W_c。
    厳密対角化(L≤12)では有限サイズ補正が強すぎて分離できなかったが、
    転送行列なら長さ方向に 10^4 スライス伸ばせるので明瞭に分離する。
    """
    n = L * L
    Hp = _cross_section_hopping(L)
    Q = np.eye(2 * n, n)
    gsum = np.zeros(n)
    eye = np.eye(n)
    for step in range(n_slices):
        eps = disorder_W * (rng.random(n) - 0.5)
        A = energy * eye - Hp - np.diag(eps)
        top, bot = Q[:n], Q[n:]
        Q = np.vstack([A @ top - bot, top])
        if (step + 1) % qr_every == 0:
            Q, R = np.linalg.qr(Q)
            d = np.abs(np.diag(R))
            d[d < 1e-300] = 1e-300
            gsum += np.log(d)
    gammas = np.sort(gsum / n_slices)
    positive = gammas[gammas > 0]
    gamma_min = positive[0] if len(positive) else gammas[-1]
    return float((1.0 / gamma_min) / L)


def _cross_section_chain(L):
    """2D ストリップ(幅 L)の断面 = L サイトの 1D 鎖(硬い境界)。"""
    H = np.zeros((L, L))
    for i in range(L - 1):
        H[i, i + 1] = H[i + 1, i] = -1.0
    return H


def transfer_matrix_lambda_2d(L, disorder_W, n_slices, rng, energy=0.0, qr_every=8):
    """
    **2次元**版の規格化局在長 Λ=λ/L(幅 L のストリップを転送行列で伸ばす)。
    3D 版(transfer_matrix_lambda)と同じ道具を、断面を 1D 鎖に替えて使う。

    2次元は Anderson 局在の**下部臨界次元(周辺次元)**であり、スケーリング理論
    では β(g)→0⁻ ——全状態が局在するが「際どい」(弱局在)。実測では W をどう
    振っても β<0 のままで、**0 を横切らない**(下記 dimensional_beta_survey)。
    """
    Hp = _cross_section_chain(L)
    n = L
    Q = np.eye(2 * n, n)
    gsum = np.zeros(n)
    eye = np.eye(n)
    for step in range(n_slices):
        eps = disorder_W * (rng.random(n) - 0.5)
        A = energy * eye - Hp - np.diag(eps)
        top, bot = Q[:n], Q[n:]
        Q = np.vstack([A @ top - bot, top])
        if (step + 1) % qr_every == 0:
            Q, R = np.linalg.qr(Q)
            d = np.abs(np.diag(R))
            d[d < 1e-300] = 1e-300
            gsum += np.log(d)
    gammas = np.sort(gsum / n_slices)
    positive = gammas[gammas > 0]
    gamma_min = positive[0] if len(positive) else gammas[-1]
    return float((1.0 / gamma_min) / L)


def beta_function_2d(disorder_W, L1, L2, n_slices, rng, energy=0.0):
    """
    2次元の β = d lnΛ/d lnL。どの W でも **負**(0 を横切らない=転移なし)。

    **収束についての注意(実測)**: 弱乱れほど 2D の局在長が指数的に大きくなるため、
    転送行列を長く取らないと値が安定しない。同一シードで n_slices を振ると:
        W=2 : -0.03 / -0.32 / -0.15  (M=6000/15000/40000、**未収束・大きく揺らぐ**)
        W=4 : -0.18 / -0.22 / -0.19  (概ね安定)
        W=6 : -0.39 / -0.46 / -0.47  (収束)
        W=8 : -0.62 / -0.61 / -0.61  (収束)
        W=12: -0.71 / -0.75 / -0.71  (収束)
    したがって W>=4 程度で使うこと。W<=2 の弱乱れ域を定量的に見るには本実装より
    遥かに長い転送行列が要る(これは数値の都合であると同時に、2D の弱局在が
    "極めて長い距離でしか現れない"という物理そのもの)。
    """
    a = transfer_matrix_lambda_2d(L1, disorder_W, n_slices, rng, energy)
    b = transfer_matrix_lambda_2d(L2, disorder_W, n_slices, rng, energy)
    return float(np.log(b / a) / np.log(L2 / L1))


def dimensional_beta_survey(rng, n_slices_2d=12000, n_slices_3d=5000):
    """
    【命題39 の完成形】次元 d=1,2,3 で「β が 0 を横切るか」を並べる。

    | d | β の振る舞い                   | 結論                       |
    |---|--------------------------------|----------------------------|
    | 1 | 常に強く負(ζ∝W^{-2} で局在)   | 転移なし                   |
    | 2 | 常に負だが 0 に最も近づく      | **周辺次元**、転移なし     |
    | 3 | W_c≈16.5 で **0 を横切る**     | **転移あり**(ν=1.602)     |

    実測(本関数で再現可能、収束する W>=4 の範囲): 2D は W=4/8/12 で
    β ≈ -0.19 / -0.61 / -0.71 と**すべて負**——弱乱れ側ほど 0 に近い(周辺的)。
    3D は W=8 で +0.71、W=16.5 で -0.016、W=24 で -0.44 と**符号が変わる**。

    正直な限界: 2次元でスケーリング理論が予言する β→0⁻ の漸近(弱乱れで
    |β|~1/(π²g) と 0 に貼り付く)は、ここでは定量的に再現できていない。W<=2 では
    転送行列長を変えると値が -0.03〜-0.32 と大きく揺れて収束しないため
    (beta_function_2d の docstring 参照)、弱乱れ域は扱わない。示せたのは
    **「2D では β が負のまま 0 を横切らない=転移が無い」という定性的な事実**まで。

    戻り値: dict(d2=[(W, beta), ...], d3=[(W, beta), ...])
    """
    d2 = [(W, beta_function_2d(W, 8, 16, n_slices_2d, rng))
          for W in (4.0, 8.0, 12.0)]
    d3 = [(W, scaling_beta_function(W, 4, 8, n_slices_3d, rng))
          for W in (8.0, 16.5, 24.0)]
    return {"d2": d2, "d3": d3}


def scaling_beta_function(disorder_W, L1, L2, n_slices, rng, energy=0.0):
    """
    【命題39】スケーリング理論の**β関数** β = d ln Λ / d ln L を、2つの断面
    サイズ L1<L2 の差分で数値評価する:

        β ≈ ln(Λ(L2)/Λ(L1)) / ln(L2/L1)

    **これは命題38 の「傾きが1を横切るか」という判定基準の、揺らぎが支配する
    系への持ち上げである**:

        平均場(命題38): 自己無撞着写像の不動点の傾きが **1** を横切る → 転移
        RG(本命題)   : スケーリング β関数が **0** を横切る          → 転移

    そして両者は同じ結論の型を与える:
      - 1次元の乱れ系: β<0 が常に成り立ち 0 を横切らない → **転移なし**
        (全状態が局在、one_dimension_has_no_mobility_edge)。
        これは OR_N の G'(t*)>1 が常に成り立ち 1 を横切らない(命題33)のと同型。
      - 3次元: β が W_c≈16.5 で 0 を横切る → **転移あり**。しかもその臨界指数は
        ν≈1.57 で、**平均場の β=1/2, γ=1 とは異なる本物の非平均場普遍性クラス**。

    戻り値: β の推定値(>0 で金属的、<0 で絶縁体的、≈0 で臨界)。
    """
    a = transfer_matrix_lambda(L1, disorder_W, n_slices, rng, energy)
    b = transfer_matrix_lambda(L2, disorder_W, n_slices, rng, energy)
    return float(np.log(b / a) / np.log(L2 / L1))


def anderson3d_critical_exponent(sizes, disorder_values, n_slices_map, rng,
                                 energy=0.0):
    """
    有限サイズスケーリングで 3D Anderson 転移の臨界指数 ν を測る。

    Λ(W,L) = f((W-W_c)·L^{1/ν}) なので、W_c 近傍では dΛ/dW ∝ L^{1/ν}。
    各 L について W_c 近傍の傾きを最小二乗で求め、log|傾き| vs log L の勾配が
    1/ν を与える。

    **実測値(再現可能)**: sizes=(4,6,8,10)、W=15.5..17.5、
    n_slices=40000/25000/15000/9000 で
        傾き = 0.0588, 0.0771, 0.0924, 0.1038  (L とともに単調増加)
        → **ν = 1.602**(文献値 1.57、誤差約2%)
    同条件で W=16.5 での Λ の平均は **0.5727**(臨界値 Λ_c≈0.576、誤差0.5%)。
    これは本ライブラリで初めての**非平均場**の臨界指数である(命題37・BCS・
    命題38 で得た転移はすべて平均場 β=1/2, γ=1 だった)。

    戻り値: (nu, {L: slope}, {(L,W): Lambda})
    """
    lambdas = {}
    slopes = {}
    Ws = np.asarray(disorder_values, dtype=float)
    for L in sizes:
        vals = []
        for W in Ws:
            lam = transfer_matrix_lambda(L, W, n_slices_map[L], rng, energy)
            lambdas[(L, float(W))] = lam
            vals.append(lam)
        slopes[L] = abs(float(np.polyfit(Ws, vals, 1)[0]))
    Ls = np.asarray(sizes, dtype=float)
    inv_nu = np.polyfit(np.log(Ls), np.log([slopes[L] for L in sizes]), 1)[0]
    return float(1.0 / inv_nu), slopes, lambdas


def one_dimension_has_no_mobility_edge():
    """
    正直な否定的事実の記録: 1次元では任意の乱れ W>0 で**全ての**固有状態が局在し、
    拡張状態と局在状態を分ける移動度端(mobility edge)も金属-絶縁体転移も存在しない
    (Mott-Twose、スケーリング理論)。したがって 1D の乱れ系には臨界点が無い
    ——命題33(OR_N に臨界性なし)と同じ種類の、誠実に記録すべき否定的結果。
    真の Anderson 転移(臨界指数 ν≈1.57 の非平均場普遍性クラス)は 3D で現れる。
    """
    return False
