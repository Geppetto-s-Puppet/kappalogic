"""
kappalogic.identities
======================
v0.13: k(x) = tanh(x/xi) が満たす「双曲線関数としての恒等式」を
まとめたモジュール。

**重要な注意(誠実な位置づけ)**:
    ここに載っている式は「新発見」ではない。tanhが三角関数(sin/cos)の
    双曲線版であることから、加法定理・倍角公式・微分公式・連分数展開
    ・部分分数展開(Mittag-Leffler)・無限積展開(Weierstrass)は
    いずれも古典的に知られている(Lambertの連分数は1768年、πの
    無理数性の証明に使われた)。ここでの作業は「それらをこのライブラリ
    のk(x)=tanh(x/xi)という書き方に翻訳し、全部を実際に数値検証し、
    kappalogic自身のどの部分と接続するかを整理したこと」である。

**もっと重要な注意(kernel依存性)**:
    このモジュールの式は**kernel="tanh"のときにしか成り立たない**。
    kernels.pyにあるerf/algebraicカーネルはtanhと別の関数なので、
    加法定理も連分数展開もWeierstrass積も一般には別の式になる
    (例えばerfの加法定理はtanhのものとは異なる)。したがって
    このモジュールの関数はすべて「k(x)=tanh(x/xi)を前提にした
    ヘルパー」であり、kernel引数を持たない(持たせると誤用を招く)。
    汎用kernel対応の加法定理を求める場合はTODO.mdのC項(フレームワーク
    対応)ではなくA項(数式いじり)の課題として別途扱うべきもの。

すべて `verify_identities` (このファイル末尾) で自己チェック済み。
"""
import numpy as np

_DEFAULT_XI = 1e-3


def rapidity(x, xi=_DEFAULT_XI):
    """
    E(x) := (1+k(x))/(1-k(x)) = exp(2x/xi)

    tanhの定義から2行で出る恒等式だが、これが加法定理・倍角公式・
    n倍角公式すべての"親"になっている: E(x)は「足し算を掛け算に
    変換する」指数写像そのもの(E(x+y)=E(x)E(y))。

    物理対応: 特殊相対論のラピディティ v/c = tanh(phi) と同じ構造。
    x/xi が「ラピディティ」の役割を担う。

    検証: 機械精度(相対誤差 ~1e-13、verify_identities参照)。
    """
    return np.exp(2 * np.asarray(x, dtype=float) / xi)


def addition(x, y, xi=_DEFAULT_XI):
    """
    k(x+y) = (k(x)+k(y)) / (1+k(x)k(y))

    sin/cosの加法定理の双曲線版。rapidity()のE(x+y)=E(x)E(y)を
    kについて解き直すと出てくる(独立した事実ではなく上位構造の影)。

    AND_n/OR_n(n項ゲートの融合)のような「複数の判定結果を1つに
    まとめる」処理と発想が似ているが、現状のAND_n/OR_nはこの式を
    直接使ってはいない(積ベースの融合であり、加法定理ベースの融合
    ではない)。両者の関係を詰めるのはTODO.mdのG項(勾配保存則)の
    候補になりうる。

    検証: 機械精度(絶対誤差 ~1e-13)。
    """
    kx = np.tanh(np.asarray(x, dtype=float) / xi)
    ky = np.tanh(np.asarray(y, dtype=float) / xi)
    return (kx + ky) / (1 + kx * ky)


def n_tuple_angle(x, n, xi=_DEFAULT_XI):
    """
    k(n*x) = ((1+k(x))^n - (1-k(x))^n) / ((1+k(x))^n + (1-k(x))^n)

    rapidity()のE(x)^n = E(nx)から即座に出る。倍角公式(n=2)は
    この特殊ケースに過ぎない。AND_n/OR_nの"n項融合"という発想との
    命名上の類似はあるが、上のaddition()と同様、現状の実装は
    これを直接使ってはいない。

    検証: n=2,3,5,7,11で機械精度(絶対誤差 ~1e-15)。
    """
    kx = np.tanh(np.asarray(x, dtype=float) / xi)
    p = (1 + kx) ** n
    q = (1 - kx) ** n
    return (p - q) / (p + q)


def lambert_continued_fraction(x, xi=_DEFAULT_XI, depth=10):
    """
    k(x) = u / (1 + u^2/(3 + u^2/(5 + u^2/(7 + ...))))  , u = x/xi

    Johann Lambert(1768年、tanの連分数展開でπの無理数性を証明した
    のと同じ形)をtan(iu)=i*tanh(u)経由でtanh版に変換したもの。

    e^xを一切使わず、四則演算だけでk(x)を近似できるのが特徴。
    kernels.pyの"algebraic"カーネル(超越関数を避ける発想)と
    設計思想が近く、超越関数(tanh自体)を使わずに済む近似計算路
    として使える(ただし"algebraic"カーネルとは別の関数へ収束する
    ので、あくまでtanhカーネルの"別の計算経路"として位置づける)。

    検証: depth=10で機械精度(絶対誤差 <2e-9)、depth=20でほぼ
    完全な機械精度(浮動小数点の限界)。depth=3では絶対誤差4e-2
    程度なので、実用にはdepth>=10を推奨。
    """
    u = np.asarray(x, dtype=float) / xi
    val = np.zeros_like(u)
    for kk in range(depth, 0, -1):
        odd = 2 * kk + 1
        val = (u ** 2) / (odd + val)
    return u / (1 + val)


def mittag_leffler_sum(x, xi=_DEFAULT_XI, n_terms=10000):
    """
    k(x) = sum_{n=1}^inf  8*xi*x / (4*x^2 + pi^2*xi^2*(2n-1)^2)

    tanh(z)の極(z = i*pi*xi*(n+1/2)、虚軸上に等間隔)を全部集めて
    足し合わせる部分分数展開(Mittag-Leffler展開)。この「極が等間隔
    に並ぶ」構造は、有限温度場の理論のフェルミオン松原振動数
    omega_n=(2n+1)*pi*T と数学的に同一であり、electronic_structure.py
    のfermi_occupation(x,2kT)=sgn(x,2kT)がこの構造の入り口に
    立っていることを示している。

    収束は連分数(lambert_continued_fraction)よりずっと遅い
    (極が無限個あるため、1個ずつの寄与が緩やかにしか効かない):
    誤差はおよそ O(xi/(n_terms*x)) のオーダーで、n_terms=100で
    絶対誤差~1e-2、n_terms=100000でようやく~1e-5。実用計算には
    向かないが、"なぜフェルミ分布と同じ構造が出るのか"を理解する
    ための恒等式として意味がある。
    """
    x = np.asarray(x, dtype=float)
    n = np.arange(1, n_terms + 1)
    # broadcasting: x can be scalar or array
    x_ = x[..., None] if x.ndim > 0 else x
    total = np.sum(8 * xi * x_ / (4 * x_ ** 2 + np.pi ** 2 * xi ** 2 * (2 * n - 1) ** 2), axis=-1)
    return total


def weierstrass_product(x, xi=_DEFAULT_XI, n_terms=1000):
    """
    k(x) = [ u * prod_n (1 + u^2/(n*pi)^2) ] / [ prod_n (1 + 4u^2/((2n-1)*pi)^2) ]
    , u = x/xi

    分子はsinh(零点がu=i*n*pi)、分母はcosh(零点=tanhの極、
    u=i*pi*(n+1/2))のWeierstrass無限積。mittag_leffler_sum()と
    「同じ零点・極の情報を、和ではなく積で再構成する」双対の関係
    にある(複素解析の定理: 有理型関数は零点・極の位置だけから
    和(部分分数分解)でも積(Weierstrass分解)でも再構成できる)。

    収束はmittag_leffler_sumよりやや速い: n_terms=1000で絶対誤差
    ~1e-6程度。
    """
    u = np.asarray(x, dtype=float) / xi
    num = u.copy() if hasattr(u, "copy") else u
    den = np.ones_like(u) if hasattr(u, "shape") else 1.0
    for n in range(1, n_terms + 1):
        num = num * (1 + (u ** 2) / (n ** 2 * np.pi ** 2))
        den = den * (1 + (4 * u ** 2) / ((2 * n - 1) ** 2 * np.pi ** 2))
    return num / den


def gudermannian(x, xi=_DEFAULT_XI):
    """
    k(x) = sin(gd(x/xi)) ,  gd(u) := 2*arctan(e^u) - pi/2

    tanhとsinの間の「翻訳」を文字通り実行する式。gd(グーデルマン
    函数)は円関数(sin,cos,tan)と双曲線関数(sinh,cosh,tanh)を結ぶ
    古典的な架け橋(メルカトル図法の緯度変換に使われる)。
    k(x)はsinに"似ている"のではなく、gd(x/xi)という角度を代入した
    瞬間のsinそのものである、という関係を明示する。

    検証: 機械精度(絶対誤差 ~1e-16)。
    """
    u = np.asarray(x, dtype=float) / xi
    gd = 2 * np.arctan(np.exp(u)) - np.pi / 2
    return np.sin(gd)


def verify_identities(xi=0.37, n_samples=500, seed=0):
    """
    このモジュールの恒等式を乱数サンプルで一括自己検証するヘルパー。
    (dev_notes.md v0.13の検証記録の再現用。テストからも呼ばれる。)

    戻り値: {識別子: 最大絶対誤差(または相対誤差)} の辞書。
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-5, 5, n_samples) * xi
    ys = rng.uniform(-5, 5, n_samples) * xi
    kappa = lambda z: np.tanh(z / xi)

    out = {}
    out["addition"] = float(np.max(np.abs(kappa(xs + ys) - addition(xs, ys, xi))))
    out["rapidity"] = float(np.max(np.abs(
        (1 + kappa(xs)) / (1 - kappa(xs)) - rapidity(xs, xi)
    ) / rapidity(xs, xi)))
    for n in (2, 3, 5, 7):
        out[f"n_tuple_angle_n{n}"] = float(np.max(np.abs(kappa(n * xs) - n_tuple_angle(xs, n, xi))))
    out["lambert_cf_depth10"] = float(np.max(np.abs(kappa(xs) - lambert_continued_fraction(xs, xi, depth=10))))
    out["mittag_leffler_n1e5"] = float(np.max(np.abs(kappa(xs) - mittag_leffler_sum(xs, xi, n_terms=100_000))))
    out["weierstrass_n1000"] = float(np.max(np.abs(kappa(xs) - weierstrass_product(xs, xi, n_terms=1000))))
    out["gudermannian"] = float(np.max(np.abs(kappa(xs) - gudermannian(xs, xi))))
    return out
