"""
kappalogic.field_theory
==========================
発見(2026-07): この理論の基本ブロック sgn(x)=tanh(x/xi) は、
実は場の理論のφ^4理論(二重井戸型ポテンシャル)の静的古典解
(キンク・ソリトン)そのものである。

【命題1(運動方程式)】phi(x)=tanh(x/xi)は
    phi'' = (2/xi^2)(phi^3-phi)
を厳密に満たす。

証明(sympyによる記号微分で確認済み、有限差分ではなく閉じた式):
    phi'(x) = (1/xi)*sech^2(x/xi) = (1/xi)*(1-phi^2)   ... (*)
    phi''(x) = d/dx[(1/xi)(1-phi^2)] = (-2*phi/xi)*phi'
             = (-2*phi/xi)*(1/xi)(1-phi^2) = (2/xi^2)(phi^3-phi)
sympy.diff(tanh(x/xi), x, 2) と (2/xi^2)(phi^3-phi) の差を
simplify すると厳密に0になることを確認した。

このEOMは、ポテンシャル V(phi)=(phi^2-1)^2/(2*xi^2) のオイラー・
ラグランジュ方程式 phi''=dV/dphi と一致する(dV/dphiを計算して
2*phi*(phi^2-1)/xi^2になることも確認済み)。

【命題2(厳密なエネルギー)】このキンクの古典エネルギー(質量)は
    E = integral_{-inf}^{inf} [ (1/2)*phi'(x)^2 + V(phi(x)) ] dx = 4/(3*xi)
と閉じた式で求まる(sympy.integrate、無限区間の積分を厳密に実行)。
幅xiに反比例する、というのは物理的に自然な結果(細いソリトンほど
勾配エネルギーが大きい)。

ソリトン数(トポロジカル電荷) Q = (phi(+inf)-phi(-inf))/2 = 1 は、
sgn(±inf)=±1という漸近値からtanhの定義より自明に従う。

【命題3(v0.50): キンクの線形安定性スペクトル——Poschl-Teller型
ポテンシャルとの一致】

静的キンクphi_0(x)=tanh(x/xi)まわりに微小揺らぎ
phi(x,t)=phi_0(x)+f(x)*exp(-i*omega*t)を考えると、線形化した
場の方程式は次のシュレーディンガー型固有値問題になる:

    -f'' + (2/xi^2)*(3*tanh(x/xi)^2 - 1)*f = omega^2 * f

z:=x/xiと置換しLambda:=(omega*xi)^2とすると、これは
l=2のPoschl-Teller型ポテンシャル -g'' - 6*sech(z)^2*g = (Lambda-4)*g
に厳密に帰着する(sympyで確認)。このポテンシャルは厳密に2つの
束縛状態を持つことが知られている(教科書的な結果):

    - ゼロモード(並進モード): Lambda=0 (omega=0)、
      固有関数 g_0(z)=sech(z)^2 (=xi*phi_0'(x)、並進不変性から
      当然出てくる零モード)
    - 内部モード("shape mode"): Lambda=3 (omega=sqrt(3)/xi)、
      固有関数 g_1(z)=tanh(z)*sech(z)
    - 連続スペクトルは omega>=2/xi から始まる

両方の固有関数をsympyの記号微分で厳密に確認した(残差0)。さらに
有限差分によるシュレーディンガー方程式の数値対角化(scipy.linalg.
eigh_tridiagonal)でも、omega=0(零モード)、omega=sqrt(3)/xi
(内部モード)、omega>=2/xi(連続スペクトルの始まり)を独立に確認した
(xi=1での数値固有値: ~0, 1.732, 2.001, 2.003, ... と理論値に
綺麗に一致)。

正直な評価: この「1つだけ内部モードを持つ」という結果自体は
phi^4理論のキンクの教科書的に知られた事実(Manton & Sutcliffeの
"Topological Solitons"等)であり、新しい物理ではない。新規性は、
これをkappalogicの`xi`パラメータ付きの記法で明示的に導出・検証し、
既存のkink_profile/kink_energy_exactと同じ枠組みに統合したことに
限られる。

【命題4(v0.51): キンク・反キンク衝突の動的シミュレーション】

非線形波動方程式 phi_tt - phi_xx + (2/xi^2)(phi^3-phi) = 0 を、
ローレンツブーストしたキンク・反キンクの重ね合わせ初期条件から
leapfrog(蛙跳び)法で数値的に時間発展させる
(`kink_antikink_collision`)。エネルギー保存(~1%程度の数値誤差)を
確認済み。

衝突後の挙動を、ゼロ点(kink/antikinkの近似位置)の間隔の時間変化
から分類したところ、教科書的に知られている速度依存性を再現できた:
    - 低速度(例: v=0.15): 間隔が単調に減少し続け、"捕獲"(capture、
      衝突後も分離せず束縛状態のように振る舞う)に近い挙動
    - 高速度(例: v=0.5): 間隔が最小値まで減少したあと単調に増加、
      明確な"脱出"(escape、衝突後に分離して離れていく)
    - 中間速度(例: v=0.25): 間隔がいったん最小になったあと部分的に
      増加する、"バウンス"的な挙動(共鳴ウィンドウ現象の入り口に
      あたる可能性があるが、今回は精密な特定はしていない)

正直な限界: これは1983年のCampbell, Schonfeld, Wingateの研究以来
有名な、"共鳴ウィンドウ"(特定の狭い速度帯でのみ複数回バウンスの
末に脱出する、フラクタル的な構造を持つ現象、命題3のshape modeとの
共鳴的なエネルギーのやり取りが原因とされる)という現象の、精密な
特定・再現までは行っていない(そのためには、より高い空間・時間
分解能と、体系的な速度スキャンが必要)。今回確認できたのは、
「低速で捕獲・高速で脱出」という大まかな速度依存性の定性的な
再現にとどまる。

【追記(v0.52): 共鳴ウィンドウらしき非単調構造を実際に確認】

v=0.15〜0.30の範囲を細かく走査したところ(`kink_antikink_velocity_scan`)、
単純に「速度が上がるほど脱出しやすくなる」という単調な構造では
なく、**捕獲領域の中に孤立した脱出("escape")の窓が現れる**、
共鳴ウィンドウに特徴的な非単調構造を確認できた(例:
v=0.19は捕獲、v=0.195〜0.20は脱出、v=0.205は再び捕獲、という
孤立した窓)。

正直な限界: ただし窓の**正確な境界**は空間分解能(格子点数N)や
シミュレーション時間Tに対して敏感で、解像度を上げると境界の位置が
シフトする(これは文献でもよく知られた、共鳴ウィンドウが非常に
狭く精密な数値計算を要求する性質そのもの)。今回確認できたのは
「非単調な構造が実在する」という定性的な事実であり、特定の窓の
正確な速度範囲を精密に特定した、という主張はしない。
"""
import numpy as np
from .core import sgn


def kink_profile(x, xi=1.0):
    """phi(x) = tanh(x/xi): phi^4理論の静的キンク解。V(phi)=(phi^2-1)^2/(2*xi^2)の古典解。"""
    return sgn(x, xi)


def kink_eom_residual(x, xi=1.0, h=1e-4):
    """
    運動方程式 phi'' - (2/xi^2)(phi^3-phi) = 0 の残差(数値検証用)。
    厳密解ならほぼ0になるはず(命題1は既にsympyで閉じた式で証明済み、
    これはその追加の数値的サニティチェック)。
    """
    phi = lambda xx: kink_profile(xx, xi)
    phi_pp = (phi(x + h) - 2 * phi(x) + phi(x - h)) / h ** 2
    rhs = (2 / xi ** 2) * (phi(x) ** 3 - phi(x))
    return phi_pp - rhs


def kink_energy_exact(xi=1.0):
    """
    キンクの厳密なエネルギー(質量) E = 4/(3*xi)。命題2参照
    (sympyによる無限区間積分で導出済み、閉じた式)。
    """
    return 4.0 / (3.0 * xi)


def topological_charge(xi=1.0, x_inf=1e4):
    """
    ソリトン数(トポロジカル電荷) Q = (phi(+inf)-phi(-inf))/2。
    このキンクでは常に1(sgnの漸近値+1,-1の半差)。
    """
    return (kink_profile(x_inf, xi) - kink_profile(-x_inf, xi)) / 2



def kink_shape_mode_frequency(xi=1.0):
    """キンクの内部モード("shape mode")の振動数 omega = sqrt(3)/xi。命題3参照。"""
    return np.sqrt(3.0) / xi


def kink_continuum_threshold(xi=1.0):
    """キンクまわりの連続スペクトルの下限 omega = 2/xi。命題3参照。"""
    return 2.0 / xi


def kink_zero_mode_profile(x, xi=1.0):
    """
    キンクの並進零モードの固有関数(規格化前) g_0(x) = sech(x/xi)^2。
    xi*d(phi_0)/dx に比例する(並進不変性から来る零モード)。
    """
    return 1.0 / np.cosh(np.asarray(x, dtype=float) / xi) ** 2


def kink_shape_mode_profile(x, xi=1.0):
    """
    キンクの内部モード("shape mode")の固有関数(規格化前)
    g_1(x) = tanh(x/xi) * sech(x/xi)。命題3参照(sympyで厳密解である
    ことを確認済み)。
    """
    z = np.asarray(x, dtype=float) / xi
    return np.tanh(z) / np.cosh(z)


def kink_stability_spectrum_numeric(xi=1.0, L=30.0, n_points=3000, n_lowest=6):
    """
    命題3の数値検証: キンクまわりの線形化シュレーディンガー型固有値問題
    -f'' + (2/xi^2)(3*tanh(x/xi)^2-1)f = omega^2 f を有限差分で
    離散化し、最低n_lowest個の固有値omega^2を返す(scipy.linalg.
    eigh_tridiagonal)。理論的には[0, 3, 4, 4+, 4+, ...]
    (零モード・内部モード・連続スペクトルの始まり)に近い値になるはず。
    """
    from scipy.linalg import eigh_tridiagonal

    x = np.linspace(-L, L, n_points)
    h = x[1] - x[0]
    U = (2 / xi ** 2) * (3 * np.tanh(x / xi) ** 2 - 1)
    diag = 2 / h ** 2 + U
    offdiag = -1 / h ** 2 * np.ones(n_points - 1)
    eigvals, _ = eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, n_lowest - 1))
    return eigvals


def kink_antikink_collision(xi=1.0, v=0.3, x0=6.0, L=50.0, N=2000, T=45.0, cfl=0.4, sample_every=200):
    """
    命題4: キンク・反キンク衝突を非線形波動方程式の直接的な数値時間発展
    (leapfrog法)でシミュレーションする。

    初期条件は、位置-x0からvで右向きに動くキンクと、位置+x0からvで
    左向きに動く反キンクの重ね合わせ(ローレンツブースト込み):

        phi(x,0) = tanh(gamma*(x+x0)/xi) - tanh(gamma*(x-x0)/xi) - 1
        (gamma := 1/sqrt(1-v^2)、真空(+-1)の二重カウントを補正するため-1)

    戻り値: {"t": 時刻の配列, "separation": ゼロ点間隔の配列
    (kink/antikinkのおおよその位置の差、NaNは衝突中でゼロ点が
    見つからない場合), "energy": 保存されるはずのエネルギーの配列}
    """
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    dt = cfl * dx
    gamma = 1.0 / np.sqrt(1 - v ** 2)

    phi0 = np.tanh(gamma * (x + x0) / xi) - np.tanh(gamma * (x - x0) / xi) - 1.0
    phidot0 = (-v * gamma / xi / np.cosh(gamma * (x + x0) / xi) ** 2
               - v * gamma / xi / np.cosh(gamma * (x - x0) / xi) ** 2)

    phi = phi0.copy()
    lap0 = (np.roll(phi0, -1) - 2 * phi0 + np.roll(phi0, 1)) / dx ** 2
    acc0 = lap0 - (2 / xi ** 2) * (phi0 ** 3 - phi0)
    phi_prev = phi0 - dt * phidot0 + 0.5 * dt ** 2 * acc0

    n_steps = int(T / dt)
    times, seps, energies = [], [], []
    for step in range(n_steps):
        lap = (np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)) / dx ** 2
        acc = lap - (2 / xi ** 2) * (phi ** 3 - phi)
        phi_next = 2 * phi - phi_prev + dt ** 2 * acc
        phi_prev, phi = phi, phi_next

        if step % sample_every == 0:
            signs = np.sign(phi)
            crossings = x[np.where(np.diff(signs) != 0)[0]]
            sep = (crossings[-1] - crossings[0]) if len(crossings) >= 2 else np.nan
            phidot = (phi - phi_prev) / dt
            V = (phi ** 2 - 1) ** 2 / (2 * xi ** 2)
            grad = np.gradient(phi, dx)
            E = np.sum(0.5 * phidot ** 2 + 0.5 * grad ** 2 + V) * dx
            times.append(step * dt)
            seps.append(sep)
            energies.append(E)

    return {"t": np.array(times), "separation": np.array(seps), "energy": np.array(energies)}


def kink_antikink_velocity_scan(velocities, xi=1.0, x0=6.0, L=50.0, N=2000, T=60.0,
                                 cfl=0.4, sample_every=250):
    """
    v0.52: 命題4の追加検証。複数の初速度でkink_antikink_collisionを
    走査し、各速度が"脱出"(escape、衝突後に分離し続ける)か
    "捕獲"(capture、分離せず束縛的に振る舞う)かを分類する。

    共鳴ウィンドウ現象(捕獲領域の中に孤立した脱出の窓が現れる、
    非単調な速度依存性)が実在するかどうかの定性的な確認に使う。
    ただし窓の正確な境界はN・Tに敏感なので、精密な境界の特定には
    向かない(field_theory.pyのモジュールdocstring参照)。

    戻り値: {v: "escape"または"capture"} という辞書。
    """
    outcomes = {}
    for v in velocities:
        result = kink_antikink_collision(
            xi=xi, v=v, x0=x0, L=L, N=N, T=T, cfl=cfl, sample_every=sample_every,
        )
        sep = result["separation"]
        valid = sep[~np.isnan(sep)]
        if len(valid) >= 3 and valid[-1] > valid[-3]:
            outcomes[v] = "escape"
        else:
            outcomes[v] = "capture"
    return outcomes
