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


def anderson3d_hamiltonian(L, disorder_W, rng, t=1.0):
    """
    3次元立方格子(周期境界)の Anderson モデル。箱型乱れ。
    戻り値: (H, x_coords) —— x_coords は各サイトの x 座標(輸送計算で位置演算子
    として使う。格子の平坦インデックスをそのまま座標にするのは誤りなので、
    transport.kubo_dc_conductivity には必ずこれを渡すこと)。
    """
    N = L ** 3
    H = np.zeros((N, N))

    def idx(x, y, z):
        return (x % L) * L * L + (y % L) * L + (z % L)

    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx(x, y, z)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = idx(x + dx, y + dy, z + dz)
                    H[i, j] -= t
                    H[j, i] -= t
    H[np.arange(N), np.arange(N)] = disorder_W * (rng.random(N) - 0.5)
    x_coords = np.array([i // (L * L) for i in range(N)], dtype=float)
    return H, x_coords


def ipr_times_n_3d(L, disorder_W, rng, n_samples=3, n_states=20, mu=0.0):
    """
    【v0.87: 3D Anderson 転移の"効く"プローブ】バンド中心近傍の固有状態について
    **IPR × N**(N=L³)を返す。

    - **拡張状態**: IPR ~ 1/N なので IPR×N ≈ 一定(ランダム行列的な揺らぎで ~3)。
      → **L を変えても平坦**。
    - **局在状態**: IPR ~ 1/ζ³(サイズに依らない)なので IPR×N ∝ N。
      → **L とともに増大**。

    したがって「L を変えたとき平坦か増大か」で拡張/局在を判定できる。

    実測(L=8 vs L=10、L³=512 vs 1000、バンド中心20状態):
        W    :  6.0    12.0   16.5   22.0    30.0
        L=8  :  3.80   11.69  34.42  65.29  130.57
        L=10 :  3.88   12.53  35.43  99.72  359.52
        比   :  1.02   1.07   1.03   1.53    2.75
    W=6 では比 1.02 と**平坦=拡張**、W>=22 で 1.5〜2.8 と**明確に増大=局在**。
    有限サイズゆえ見かけの境界は文献の W_c=16.5 より高めに出る(小さい系では
    局在長が系サイズを超えると拡張に見えるため)——転移の**存在**は捉えられるが、
    W_c の精密決定には転送行列(命題39)の方が適する。

    正直な注記: 同じ系で久保伝導度 σ を有限サイズ signature に使おうとしたが
    **失敗した**(transport.kubo_dc_conductivity の docstring 参照)。効くプローブと
    効かないプローブを見極めるのも結果のうち。
    """
    vals = []
    for _ in range(n_samples):
        H, _ = anderson3d_hamiltonian(L, disorder_W, rng)
        E, V = np.linalg.eigh(H)
        mid = np.argsort(np.abs(E - mu))[:n_states]
        p = np.abs(V[:, mid]) ** 2
        ipr = np.sum(p ** 2, axis=0) / np.sum(p, axis=0) ** 2
        vals.append(float(np.mean(ipr)) * (L ** 3))
    return float(np.mean(vals))


def mean_level_spacing(eigenvalues, half_width=0.5):
    """バンド中心 |E|<half_width の平均準位間隔 Δ。"""
    e = np.sort(np.asarray(eigenvalues, dtype=float))
    mid = e[np.abs(e) < half_width]
    if len(mid) < 3:
        return float(np.mean(np.diff(e)))
    return float(np.mean(np.diff(mid)))


def smeared_dos_at_zero(eigenvalues, xi):
    """kappalogic のデルタ検出器で平滑化した E=0 の状態密度 (1/N)Σ δ_ξ(E_i)。"""
    from .funcs import delta_approx
    e = np.asarray(eigenvalues, dtype=float)
    return float(np.sum(delta_approx(e, xi)) / len(e))


# 命題41 の較正値(純アンサンブルで実測、N=400・300配位、ξ/Δ=4..64)
PICKET_EXPONENT = -1.501    # 等間隔(超剛的)スペクトル
GOE_EXPONENT = -0.968       # GOE(剛的、準位反発)  理論 -1
POISSON_EXPONENT = -0.594   # 独立準位(局在)        理論 -0.5


def resolution_exponent(spectra, xi_over_delta=(1, 2, 4, 8)):
    """
    【命題41(v0.90): ξ分解能則の指数は、スペクトル統計=局在の診断になる】

    ξ 幅の検出器で離散スペクトルから量を測るとき、窓は N_eff ≈ ξ/Δ 個の準位を
    平均する(Δ=平均準位間隔)。その**配位間の相対ゆらぎ**が ξ にどう依存するかを
    両対数の傾き p として返す:

        相対ゆらぎ(ρ_ξ(0))  ∝  ξ^p

    **p はスペクトル統計で決まる**:
      - **剛的スペクトル(準位反発あり=拡張状態)**: 窓内の準位数のゆらぎは
        対数的にしか増えない(Var(n)~ln n)ので相対ゆらぎ ~ 1/n → **p ≈ -1 以下**
      - **Poisson スペクトル(準位が独立=局在状態)**: Var(n)=n なので
        相対ゆらぎ ~ 1/√n → **p → -0.5**

    == 純アンサンブルによる較正(v0.91)==
    人工的に統計だけを作って測ると、理論値によく一致する:
        picket(等間隔・超剛的) <r>=0.961   p = -1.501
        **GOE(剛的)**          <r>=0.529   p = **-0.968**  (理論 -1)
        **Poisson(独立)**      <r>=0.387   p = **-0.594**  (理論 -0.5)
    これが p の**物差し**になる(PICKET/GOE/POISSON_EXPONENT 定数)。

    == 【重要】ξ には**両側**の境界がある(v0.91 で上限を発見)==
    既定の ξ/Δ=(1,2,4,8) は「局所窓」であり、これには理由がある:
      - **下限 ξ ≳ Δ**: 小さすぎると離散化のアーティファクトを解像する
        (v0.87 の久保σの η、v0.89 の逆設計の ξ)。
      - **上限 ξ ≲ 状態密度が変化するスケール**: 大きすぎると窓が**バンド構造を
        跨いで**しまい、測っているゆらぎが準位統計でなく DOS の形に汚染される。
    3D Anderson(L=8、W=6、拡張相=純GOEのはず)で ξ/Δ の範囲を変えると:
        ξ/Δ=4..64 → p=-0.527   2..16 → -0.797   **1..8 → -0.971**
    広い窓では GOE 値(-0.968)から大きくずれ、**局所窓にして初めて一致する**。
    したがって **Δ ≪ ξ ≪ E_DOS** という窓の中で使うこと。

    実測(1次元乱れ鎖 N=400、300配位、既定の局所窓):
        W     <r>       局在長ζ     p
        0.5   0.5914    153.6     -1.307   (超剛的・清浄極限寄り)
        1.0   0.4662     92.2     -1.131
        2.0   0.4076     32.1     -0.905   (≈GOE)
        6.0   0.3830      4.5     -0.623
        12.0  0.3879      2.0     -0.493   (≈Poisson の理論値 -0.5)
    3次元 Anderson(L=8、80配位、既定の局所窓):
        W=6.0  (拡張) <r>=0.532  **p=-0.971**  ← 純GOE と一致
        W=16.5        <r>=0.517    p=-0.577
        W=30.0 (局在) <r>=0.434  **p=-0.565**  ← 純Poisson と一致
    ⟨r⟩ の低下と同期して p が動き、**1次元でも3次元でも純アンサンブルの値を
    再現する**。

    == なぜ重要か ==
    (1) **ξ の選び方の処方箋が系依存だと分かる**: 相対精度 ε が欲しいとき、
        拡張系(p≈-1)なら ξ ≳ Δ/ε で足りるが、**局在系(p≈-0.5)では ξ ≳ Δ/ε²**
        と桁違いに大きく取る必要がある。v0.87(久保σのηを小さくしすぎて破綻)と
        v0.89(逆設計のξ)の失敗が、この1つの定量則に統一される。
    (2) **固有値だけで測れる局在診断**: IPR は固有ベクトルを、転送行列は別の
        計算を要するが、本量は**固有値のアンサンブルだけ**から得られる。

    引数 spectra: 固有値配列のリスト(同じ系の異なる乱れ配位)。
    戻り値: (p, [(xi, 相対ゆらぎ), ...])

    正直な限界:
    - **相を同定できるが、転移点 W_c は定まらない**。3D の L=8 では W=16.5(=W_c)
      が既に Poisson 寄り(-0.577)に読める——v0.87 で厳密対角化が W_c を分離
      できなかったのと同じ有限サイズの制約。W_c の精密決定は転送行列(命題39)へ。
    - 窓の選び方に敏感(上記の両側境界)。DOS が急峻な系ではさらに狭い窓が要る。
    - 配位数は 1次元では 30 でも W=0.5 と W=12 を分離できたが、ゆらぎ(2次
      モーメント)の推定なので平均量より多くの標本を要する。
    """
    spectra = [np.asarray(s, dtype=float) for s in spectra]
    delta = float(np.mean([mean_level_spacing(s) for s in spectra[:20]]))
    xis, rels = [], []
    for factor in xi_over_delta:
        xi = delta * factor
        vals = np.array([smeared_dos_at_zero(s, xi) for s in spectra])
        xis.append(xi)
        rels.append(float(np.std(vals) / np.mean(vals)))
    p = float(np.polyfit(np.log(xis), np.log(rels), 1)[0])
    return p, list(zip(xis, rels))


def required_xi_for_accuracy(delta, target_relative_error, exponent):
    """
    命題41 の**処方箋**: 平均準位間隔 Δ の系で相対精度 `target_relative_error` を
    得るのに必要な検出器幅 ξ の目安。相対ゆらぎ ≈ (ξ/Δ)^p を反転する:

        ξ ≈ Δ · (target_relative_error)^(1/p)

    拡張系(p≈-1)なら ξ ≈ Δ/ε、**局在系(p≈-0.5)なら ξ ≈ Δ/ε²** と桁違いに
    大きくなる。exponent は resolution_exponent で測った値を渡す(負の数)。
    """
    if exponent >= 0:
        raise ValueError("exponent must be negative (fluctuation decreases with xi)")
    return float(delta * target_relative_error ** (1.0 / exponent))


def one_dimension_has_no_mobility_edge():
    """
    正直な否定的事実の記録: 1次元では任意の乱れ W>0 で**全ての**固有状態が局在し、
    拡張状態と局在状態を分ける移動度端(mobility edge)も金属-絶縁体転移も存在しない
    (Mott-Twose、スケーリング理論)。したがって 1D の乱れ系には臨界点が無い
    ——命題33(OR_N に臨界性なし)と同じ種類の、誠実に記録すべき否定的結果。
    真の Anderson 転移(臨界指数 ν≈1.57 の非平均場普遍性クラス)は 3D で現れる。
    """
    return False
