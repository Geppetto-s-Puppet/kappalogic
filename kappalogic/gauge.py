"""
kappalogic.gauge
==================
v0.14: TODO.md A項「ゲージ理論的発想」への着手。
「大域的なξ変換」(ξ->定数倍)と「局所的なξ変換」(ξ(x)->位置依存の
場)を定式化し、交換子・リー群構造・保存量・キリングベクトル場を
実際に計算した。

**誠実な位置づけ(最初に断っておく)**:
ここで見つかった構造(アフィン群 "ax+b"、その2次元非可換リー環、
双曲平面の等長変換群としてのキリングベクトル場、局所ゲージ変換の
接続場)は、数学的にはどれも極めて古典的である:
    - アフィン群Aff(1,R)="ax+b group"は連続ウェーブレット変換の
      対称性としても知られる標準的な2次元可解リー群。
    - 上半平面のPSL(2,R)不変双曲計量とそのキリングベクトル場は
      双曲幾何の教科書的な内容。
    - 「大域対称性を局所化すると接続場が必要になる」という発想は
      ゲージ理論の基本(ここではWeyl変換/dilatonに近い、可換な
      "scale gauge"の初等版)。
新しいのは「kappalogicのk(x;ξ)=tanh(x/ξ)という(位置x、スケールξ)の
パラメータ化が、まさにこれらの古典的構造の実例になっている」ことを
実際に手を動かして(sympyで交換子・キリング方程式を、scipyで
測地線の保存量を)確認し、ξの意味論(TODO.md C項)の統一候補として
「ξ=RG(繰り込み群)スケール、x=境界座標」という解釈を提示したこと。
すべて`verify_gauge_structure()`で自己検証できる。
"""
import numpy as np
import sympy as sp


# ------------------------------------------------------------------
# 1. 大域的ξ変換 = xの拡大縮小(ディレーション)、と生成子
# ------------------------------------------------------------------

def xi_dilation_equals_x_dilation(x, xi, lam):
    """
    k(x; lam*xi) = k(x/lam; xi) の数値確認(kernel="tanh"を想定)。
    「ξをlam倍する」ことと「xを1/lam倍する」ことが厳密に同じ操作
    であることを示す(グローバルなξ変換の正体)。
    """
    x = np.asarray(x, dtype=float)
    lhs = np.tanh(x / (lam * xi))
    rhs = np.tanh((x / lam) / xi)
    return lhs, rhs


def dilation_generator_matches_xi_generator(x, xi, h=1e-6):
    """
    xi->exp(theta)*xiの無限小生成子(d/dtheta at theta=0)が、
    xの拡大縮小の生成子 X=x*d/dx にマイナス符号で一致することを
    数値的に確認する。戻り値: (d/dtheta k(x;e^theta*xi)|_0, -x*dk/dx)
    のペア(両方ほぼ同じ値になるはず)。
    """
    x = np.asarray(x, dtype=float)

    def k_of_theta(theta):
        return np.tanh(x / (np.exp(theta) * xi))

    d_dtheta = (k_of_theta(h) - k_of_theta(-h)) / (2 * h)

    def kappa(xx):
        return np.tanh(xx / xi)

    dk_dx = (kappa(x + h) - kappa(x - h)) / (2 * h)
    minus_X = -x * dk_dx
    return d_dtheta, minus_X


# ------------------------------------------------------------------
# 2. リー環: 並進T=d/dx と ディレーションD=x*d/dx の交換子
# ------------------------------------------------------------------

def bracket_1d(a1, a2, x_sym):
    """
    ベクトル場 V1=a1(x)d/dx, V2=a2(x)d/dx の交換子 [V1,V2] = c(x)*d/dx
    の係数c(x)をsympyで計算する。 c = a1*a2' - a2*a1'。
    """
    c = a1 * sp.diff(a2, x_sym) - a2 * sp.diff(a1, x_sym)
    return sp.simplify(c)


def translation_dilation_algebra():
    """
    T = d/dx (並進), D = x*d/dx (ディレーション) の交換子を計算する。
    [D,T] = -T となり、これは「アフィン群 Aff(1,R) = {x -> a*x+b, a>0}」
    のリー環(aff(1)、2次元・非可換・可解)の定義関係そのものである。

    戻り値: [D,T]の係数(sympy式。-1になるはず)
    """
    x_sym = sp.Symbol('x')
    a_T = sp.Integer(1)
    a_D = x_sym
    return bracket_1d(a_D, a_T, x_sym)


# ------------------------------------------------------------------
# 3. (x,xi)半平面上の双曲計量とキリングベクトル場
# ------------------------------------------------------------------

def hyperbolic_metric_is_scale_invariant():
    """
    上半平面 {(x,xi): xi>0} 上の双曲計量 ds^2=(dx^2+dxi^2)/xi^2 が、
    同時拡大縮小 (x,xi) -> (lam*x, lam*xi) の下で不変であることを
    sympyで確認する(直接代入して差が0になることを見るだけで十分:
    dx,dxi はlam倍、xi^2もlam^2倍になるので比は不変)。
    戻り値: 0になるはずのsympy式。
    """
    dx, dxi, xi_s, lam = sp.symbols('dx dxi xi lam', positive=True)
    metric = (dx ** 2 + dxi ** 2) / xi_s ** 2
    metric_scaled = (lam ** 2 * dx ** 2 + lam ** 2 * dxi ** 2) / (lam * xi_s) ** 2
    return sp.simplify(metric - metric_scaled)


def _lie_derivative_of_hyperbolic_metric(Vx, Vxi, x_sym, xi_sym):
    """内部ヘルパー: 計量g=diag(1/xi^2,1/xi^2)のリー微分 L_V g を計算する。"""
    coords = [x_sym, xi_sym]
    V = [Vx, Vxi]
    g = sp.Matrix([[1 / xi_sym ** 2, 0], [0, 1 / xi_sym ** 2]])
    n = 2
    Lg = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            term = 0
            for k in range(n):
                term += V[k] * sp.diff(g[i, j], coords[k])
                term += g[k, j] * sp.diff(V[k], coords[i])
                term += g[i, k] * sp.diff(V[k], coords[j])
            Lg[i, j] = sp.simplify(term)
    return Lg


def killing_vectors_of_xi_halfplane():
    """
    【命題(v0.14): 並進とディレーションはxi半平面の双曲計量の
    キリングベクトル場である】

    V1 = d/dx (並進), V2 = x*d/dx + xi*d/dxi (同時ディレーション) が、
    ds^2=(dx^2+dxi^2)/xi^2 (上半平面の双曲計量、xiを"高さ"とみなす)
    に対して L_V g = 0 (等長変換の生成子=キリングベクトル場) を
    満たすことをsympyで確認する。

    戻り値: {"V1": リー微分行列(零行列のはず), "V2": 同左,
             "bracket_V2_V1": [V2,V1]の係数ベクトル(-V1になるはず)}
    """
    x_sym, xi_sym = sp.symbols('x xi', positive=True)
    Lg_V1 = _lie_derivative_of_hyperbolic_metric(sp.Integer(1), sp.Integer(0), x_sym, xi_sym)
    Lg_V2 = _lie_derivative_of_hyperbolic_metric(x_sym, xi_sym, x_sym, xi_sym)

    def bracket2d(V, W):
        Vx, Vxi = V
        Wx, Wxi = W
        new_x = sp.simplify(
            Vx * sp.diff(Wx, x_sym) + Vxi * sp.diff(Wx, xi_sym)
            - (Wx * sp.diff(Vx, x_sym) + Wxi * sp.diff(Vx, xi_sym))
        )
        new_xi = sp.simplify(
            Vx * sp.diff(Wxi, x_sym) + Vxi * sp.diff(Wxi, xi_sym)
            - (Wx * sp.diff(Vxi, x_sym) + Wxi * sp.diff(Vxi, xi_sym))
        )
        return new_x, new_xi

    bracket = bracket2d((x_sym, xi_sym), (sp.Integer(1), sp.Integer(0)))
    return {"V1": Lg_V1, "V2": Lg_V2, "bracket_V2_V1": bracket}


# ------------------------------------------------------------------
# 4. 測地線の保存量(ネーターの定理の初等版)
# ------------------------------------------------------------------

def _geodesic_rhs(t, state):
    x, xi, vx, vxi = state
    ax = (2 / xi) * vx * vxi
    axi = -(1 / xi) * (vx ** 2 - vxi ** 2)
    return [vx, vxi, ax, axi]


def geodesic_conserved_quantities(x0=0.3, xi0=1.0, vx0=0.6, vxi0=0.8, t_max=3.0, n_points=300):
    """
    (x,xi)半平面(双曲計量ds^2=(dx^2+dxi^2)/xi^2)上の測地線を数値積分し、
    キリングベクトル場V1(並進),V2(ディレーション)に対応する
    「運動量」p1=g(V1,gamma')、p2=g(V2,gamma') が、測地線に沿って
    保存することを確認する(標準的なNoetherの定理: キリングベクトル場
    に沿った計量内積は測地線に沿って保存する)。副産物として速さ
    (ハミルトニアン)も保存するはず。

    戻り値: {"p_trans": 配列, "p_dilation": 配列, "speed_sq": 配列, "t": 配列}
    (理想的にはどの配列も定数に近い値になる)
    """
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        _geodesic_rhs, [0, t_max], [x0, xi0, vx0, vxi0],
        max_step=t_max / 3000, dense_output=True,
    )
    ts = np.linspace(0, t_max, n_points)
    xs, xis, vxs, vxis = sol.sol(ts)

    p_trans = vxs / xis ** 2
    p_dilation = (xs * vxs + xis * vxis) / xis ** 2
    speed_sq = (vxs ** 2 + vxis ** 2) / xis ** 2

    return {"p_trans": p_trans, "p_dilation": p_dilation, "speed_sq": speed_sq, "t": ts}


# ------------------------------------------------------------------
# 5. 局所ξ変換 = ゲージ接続 A(x) = (log xi(x))'
# ------------------------------------------------------------------

def local_xi_connection_symbolic():
    """
    ξを場ξ(x)にした(局所化した)とき、phi(x)=x/xi(x)の微分は
        phi'(x) = (1/xi(x)) * (1 - x*A(x)),   A(x) := (log xi(x))'(x)
    と書ける(A(x)がゲージ場ならぬ"scale connection"の役割を果たす)。
    さらにゲージ変換 xi(x) -> lam(x)*xi(x) の下で
        A(x) -> A(x) + (log lam(x))'(x)
    と、通常のU(1)ゲージ場と同じ「完全微分だけずれる」変換則に従う
    ことをsympyで確認する。

    戻り値: {"phi_prime_identity": 0になるはずの式,
             "gauge_shift_identity": 0になるはずの式}
    """
    x = sp.Symbol('x')
    xi_fn = sp.Function('xi')(x)
    lam_fn = sp.Function('lam')(x)

    phi = x / xi_fn
    dphi_dx = sp.diff(phi, x)
    A = sp.diff(sp.log(xi_fn), x)
    phi_prime_identity = sp.simplify(dphi_dx - (1 / xi_fn) * (1 - x * A))

    xi_new = lam_fn * xi_fn
    A_new = sp.diff(sp.log(xi_new), x)
    gauge_shift_identity = sp.simplify(A_new - (A + sp.diff(sp.log(lam_fn), x)))

    return {"phi_prime_identity": phi_prime_identity, "gauge_shift_identity": gauge_shift_identity}


def local_xi_connection_numeric(x, xi_func, h=1e-6):
    """
    xi_func: xi(x)を返す関数(callable)。上のsymbolic identityを
    具体的な数値のxi(x)(例: xi_func = lambda x: 1+0.1*x)について
    数値微分で再確認するヘルパー。
    戻り値: (phi'(x)の数値微分, (1/xi)(1-x*A)の値) のペア。
    """
    x = np.asarray(x, dtype=float)

    def phi(xx):
        return xx / xi_func(xx)

    phi_prime = (phi(x + h) - phi(x - h)) / (2 * h)

    def log_xi(xx):
        return np.log(xi_func(xx))

    A = (log_xi(x + h) - log_xi(x - h)) / (2 * h)
    xi_val = xi_func(x)
    closed = (1 / xi_val) * (1 - x * A)
    return phi_prime, closed


# ------------------------------------------------------------------
# 一括自己検証
# ------------------------------------------------------------------

def verify_gauge_structure():
    """
    このモジュールの主張を一括で数値的に検証する(dev_notes.md v0.14
    参照)。戻り値: {識別子: 最大絶対誤差 or 判定結果} の辞書。
    """
    out = {}

    lhs, rhs = xi_dilation_equals_x_dilation(np.linspace(-5, 5, 200), 0.37, 2.3)
    out["xi_dilation_eq_x_dilation"] = float(np.max(np.abs(lhs - rhs)))

    d1, d2 = dilation_generator_matches_xi_generator(np.linspace(-3, 3, 200) * 0.37, 0.37)
    out["dilation_generator_matches"] = float(np.max(np.abs(d1 - d2)))

    out["bracket_D_T"] = str(translation_dilation_algebra())  # 期待値: "-1"

    out["hyperbolic_metric_scale_invariance"] = str(hyperbolic_metric_is_scale_invariant())  # "0"

    kv = killing_vectors_of_xi_halfplane()
    out["killing_V1_is_zero_matrix"] = kv["V1"].is_zero_matrix
    out["killing_V2_is_zero_matrix"] = kv["V2"].is_zero_matrix
    out["bracket_V2_V1"] = kv["bracket_V2_V1"]  # 期待値: (-1, 0)

    geo = geodesic_conserved_quantities()
    out["p_trans_std"] = float(np.std(geo["p_trans"]))
    out["p_dilation_std"] = float(np.std(geo["p_dilation"]))
    out["speed_sq_std"] = float(np.std(geo["speed_sq"]))

    conn = local_xi_connection_symbolic()
    out["phi_prime_identity_is_zero"] = (sp.simplify(conn["phi_prime_identity"]) == 0)
    out["gauge_shift_identity_is_zero"] = (sp.simplify(conn["gauge_shift_identity"]) == 0)

    xs = np.linspace(-2, 2, 100)
    xi_func = lambda xx: 1.0 + 0.1 * xx + 0.02 * xx ** 2
    pp, closed = local_xi_connection_numeric(xs, xi_func)
    out["local_connection_numeric_max_err"] = float(np.max(np.abs(pp - closed)))

    return out
