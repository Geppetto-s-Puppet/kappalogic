"""
kappalogic.theory
====================
reg/AND/OR系(6ゲート: AND, OR, NAND, NOR, XOR, XNOR)の勾配構造・
誤分類境界を、閉じた形の式として体系的に導出・検証した命題群(命題1〜21)。

全命題の一覧と数式はREADME.mdの「命題まとめ」表を参照。発見の経緯・
迷った点はdev_notes.mdを参照。ここでは各関数のdocstringに、対応する
命題の主張・導出の要点・検証結果を記載する。

要点だけ先に書いておくと:
- 命題1〜3: reg/ANDの勾配構造(命題3はv0.16で反例発見・修正)。
- 命題4〜6: ORの勾配の"二重共鳴"構造と対数スケールの第二共鳴。
- 命題5: ANDは`a*b/xi`という1変数に還元できる(部分ディレーション不変)が、
  ORはできない——以降の非対称性の根本原因。
- 命題8〜10: OR(_n)の誤分類境界(2変数・n変数)と、naive foldとの
  一致条件(命題10、TODOの本丸だった課題)。
- 命題11〜20: NAND/NOR/XOR/XNORへの拡張、6ゲート全部の分類と
  (a,b)平面の完全な閉形式。
- 命題16: 命題11・13・15を「NOTをn回重ねる」という1つの一般式に統合。
- 命題21: 命題6・8・9・12・13・15・19・20が使ってきた漸近近似が、
  実は`identities.py`の積分形と厳密に同じ恒等式`-ln(NOT(x;xi))=2ln(cosh(x/xi))`
  の近似に過ぎなかったと判明——OR系命題群を統一する厳密方程式。
"""

import numpy as np


def max_gradient_location(xi=1.0):
    """reg(x)の勾配|reg'(x)|が最大になるxの位置(x>0側)。命題1参照。"""
    u_star = np.arctanh(1 / np.sqrt(3))
    return xi * u_star


def max_gradient_value(xi=1.0):
    """reg(x)の勾配|reg'(x)|の最大値。命題1参照。"""
    u_star = np.arctanh(1 / np.sqrt(3))
    tanh_u = 1 / np.sqrt(3)
    sech2_u = 1 - tanh_u ** 2
    return (2 / xi) * tanh_u * sech2_u


def is_single_passing_at_zero(and_fn, b=0.0, a_values=(-5, -1, 0.5, 3, 100), h=1e-5):
    """
    AND(a,b)がb=0でsingle-passing(dAND/da=0 for all a)かどうかを
    数値的に検証するヘルパー。命題2参照。
    """
    for a in a_values:
        d = (and_fn(a + h, b) - and_fn(a - h, b)) / (2 * h)
        if abs(d) > 1e-6:
            return False
    return True


def fusion_is_safe(values, xi=1.0, C=3.0):
    """
    命題3(v0.16修正版): AND_n融合の安全条件。

    当初(v0.7)の主張「部分積 s_k=a1*...*ak が全部|s_k|>C*xiなら安全」は
    **誤りだった**。反例(xi=1,C=3): `[100,100,0.01,50]`は部分積が全部
    閾値を超えるが、naive_fold≈2.5e-5 vs fused=1.0とほぼ正反対になる
    (乱数2万試行で旧条件の約12%が不一致、Cを大きくしても直らない)。

    原因: naive foldは`acc_k=reg(acc_{k-1}*a_k)`という漸化式で計算され、
    acc_{k-1}が一度reg()で[0,1)に飽和すると部分積の大きさの情報を失う。
    次段は実質`reg(a_k)`程度で決まるため、効くのは「直近の個々の値」。

    **正しい条件**: 個々の値`a_1,...,an`が全部`|a_k|>C*xi`(部分積では
    なく個別の値)。乱数2万試行(n=3〜100)で不一致0件を確認。誤差上界

        |naive_fold - AND_n(融合版)| <= n * 4*exp(-2*C)

    も導出・検証済み(1-reg(x)は|x|>C*xiのとき4exp(-2C)以下という
    命題1と同根の指数減衰から、n段でn*4exp(-2C)。実測gapは常に
    この1/3以下)。

    戻り値: True なら個々の値が全部C*xiを超えており、n*4exp(-2C)
    程度の精度でnaive foldとAND_n融合版が一致するとみなせる。
    """
    values = np.abs(np.asarray(values, dtype=float))
    return bool(np.all(values > C * xi))


def fusion_error_bound(n, C=3.0):
    """
    命題3(v0.16修正版)の誤差上界 n*4*exp(-2C) を返すヘルパー。
    fusion_is_safe(values,xi,C)がTrueのとき、
    |naive_fold - AND_n(融合版)| はこの値以下に収まると期待できる
    (乱数検証込み、詳細はfusion_is_safeのdocstring・dev_notes.md参照)。
    """
    return n * 4 * np.exp(-2 * C)


def _reg(x, xi):
    return np.tanh(np.asarray(x, dtype=float) / xi) ** 2


def _reg_prime(x, xi):
    """reg'(x) = (2/xi)*tanh(x/xi)*sech^2(x/xi) の閉形式(命題1参照)。"""
    u = np.asarray(x, dtype=float) / xi
    tanh_u = np.tanh(u)
    sech2_u = 1 - tanh_u ** 2
    return (2 / xi) * tanh_u * sech2_u


def or_gradient_closed_form(a, b, xi=1.0):
    """
    d/da OR(a,b) の閉形式(命題4の式(*))。
        p = (1-reg(a))*(1-reg(b))
        d/da OR(a,b) = reg'(a) * (1-reg(b)) * reg'(p)
    数値微分との相対誤差~1e-6で一致することを確認済み。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = (1 - _reg(a, xi)) * (1 - _reg(b, xi))
    return _reg_prime(a, xi) * (1 - _reg(b, xi)) * _reg_prime(p, xi)


def gradient_landscape_stats(xi=1e-3, scale=8.0, n_samples=200_000, seed=0, tol=1e-3):
    """
    命題4の表(AND/ORの勾配地形の非対称性)を再現するベンチマーク。
    a, b を xi の +-scale 倍の範囲から一様サンプルし、数値微分で
    d/da AND(a,b), d/da OR(a,b) を評価して比較する。

    戻り値: {"and_flat_frac", "and_max_grad", "or_flat_frac", "or_max_grad"}
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(-scale, scale, n_samples) * xi
    b = rng.uniform(-scale, scale, n_samples) * xi
    h = 1e-6

    def NOT(x):
        return 1 - _reg(x, xi)

    def AND(x, y):
        return _reg(x * y, xi)

    def OR(x, y):
        # OR(a,b) = NOT(NOT(a)*NOT(b)) = 1 - reg(NOT(a)*NOT(b))
        return NOT(NOT(x) * NOT(y))

    d_and = (AND(a + h, b) - AND(a - h, b)) / (2 * h)
    d_or = (OR(a + h, b) - OR(a - h, b)) / (2 * h)

    return {
        "and_flat_frac": float(np.mean(np.abs(d_and) < tol)),
        "and_max_grad": float(np.max(np.abs(d_and))),
        "or_flat_frac": float(np.mean(np.abs(d_or) < tol)),
        "or_max_grad": float(np.max(np.abs(d_or))),
    }


def or_fusion_disagreement_rate(xi=1e-3, n_vars=7, scale=8.0, n_trials=20_000, seed=42, gap_threshold=0.5):
    """
    OR_nの融合版とnaive foldが「正反対の答え」(gapが大きい)に割れる頻度を
    乱数試行で推定する(命題4、AND_nとの対比)。
    比較用に同条件でのAND_nの不一致率も返す。
    """
    rng = np.random.default_rng(seed)

    def NOT(x):
        return 1 - _reg(x, xi)

    def OR(x, y):
        # OR(a,b) = NOT(NOT(a)*NOT(b)) = 1 - reg(NOT(a)*NOT(b))
        return NOT(NOT(x) * NOT(y))

    def AND(x, y):
        return _reg(x * y, xi)

    def naive_fold(vals, op):
        acc = vals[0]
        for v in vals[1:]:
            acc = op(acc, v)
        return acc

    def or_n_fused(vals):
        prod = 1.0
        for v in vals:
            prod = prod * NOT(v)
        return NOT(prod)

    def and_n_fused(vals):
        prod = 1.0
        for v in vals:
            prod = prod * v
        return _reg(prod, xi)

    mismatch_or = 0
    mismatch_and = 0
    for _ in range(n_trials):
        vals = rng.uniform(-scale, scale, n_vars) * xi
        if abs(float(or_n_fused(vals)) - float(naive_fold(vals, OR))) > gap_threshold:
            mismatch_or += 1
        if abs(float(and_n_fused(vals)) - float(naive_fold(vals, AND))) > gap_threshold:
            mismatch_and += 1
    return {
        "or_disagreement_rate": mismatch_or / n_trials,
        "and_disagreement_rate": mismatch_and / n_trials,
    }


def or_second_resonance_w0():
    """
    命題6で使う普遍定数w0: tanh(w) + w*(1-3*tanh(w)^2) = 0 の、
    u*=arctanh(1/sqrt(3))より大きい側の唯一解(w0~1.00955655)。
    scipy.optimize.brentqで数値的に解く(cに依存しない普遍定数)。
    """
    from scipy.optimize import brentq
    u_star = np.arctanh(1 / np.sqrt(3))

    def eq(w):
        t = np.tanh(w)
        return t + w * (1 - 3 * t ** 2)

    return brentq(eq, u_star + 1e-9, 10.0)


def or_second_resonance_location(xi, c):
    """
    命題6の閉形式(v0.17): a=c*xi(cは固定定数)としたとき、
    ORの"第二共鳴"点 b* = v* * xi の v* を返す(xi->0の漸近極限)。

        v* = -(1/2)*ln(xi) - (1/2)*ln(w0 / (4*(1-tanh(c)^2)))

    戻り値: 予測されるv*=b*/xi(実数)。
    """
    w0 = or_second_resonance_w0()
    A0 = 1 - np.tanh(c) ** 2   # = NOT(c*xi; xi) = sech^2(c)、xiに依存しない
    return -0.5 * np.log(xi) - 0.5 * np.log(w0 / (4 * A0))


def or_second_resonance_numeric_argmax(xi, c, v_max=None, n_coarse=400_000, n_fine=40_000):
    """
    命題6の閉形式を検証するためのヘルパー: a=c*xi固定のもとで、
    h(v) := NOT(v*xi;xi) * reg'(p(v);xi), p(v)=(1-reg(c*xi;xi))*(1-reg(v*xi;xi))
    を実際にv>0についてグリッド探索で最大化し、argmax(=v*)を返す。
    or_second_resonance_locationの予測値と比較するために使う。
    """
    if v_max is None:
        v_max = max(30.0, -3.0 * np.log(xi))

    a_fixed = c * xi

    def h_of_v(v):
        b = v * xi
        p = (1 - _reg(a_fixed, xi)) * (1 - _reg(b, xi))
        return (1 - _reg(b, xi)) * _reg_prime(p, xi)

    vs = np.linspace(1e-6, v_max, n_coarse)
    hs = h_of_v(vs)
    imax = np.argmax(hs)
    lo = max(vs[imax] - v_max / n_coarse * 5, 1e-8)
    hi = vs[imax] + v_max / n_coarse * 5
    vs2 = np.linspace(lo, hi, n_fine)
    hs2 = h_of_v(vs2)
    return float(vs2[np.argmax(hs2)])


def or_full_gradient_magnitude_argmax(xi, c, v_bounds=None):
    """
    v0.18で残っていた"(a,b)平面の危険地図の真の尾根"の解決策
    (v0.26)。命題6は当初「d/da OR(a,b)の一部h(v)=NOT(b)*reg'(p)」
    だけを最大化していたが、実は**|grad OR|=sqrt((dOR/da)^2+(dOR/db)^2)
    という結合された勾配の大きさそのものを最大化しても、a=c*xiを
    固定したときの最適なvは(数値精度10^-6〜10^-10のオーダーで)
    or_second_resonance_locationと厳密に一致する**ことが分かった。

    つまり命題6の閉形式は、"部分的な勾配"の共鳴点であるだけでなく、
    **(a,b)平面上の|grad OR|の"真の尾根"そのもの**でもあった
    (v0.18で「近似曲線は数%〜十数%の誤差で尾根に近い」と書いていたが、
    命題6の閉形式を(離散的な数点ではなく)cの連続関数として使えば、
    誤差は数値精度の範囲に収まる)。

    このヘルパーは、|grad OR|^2をscipyのminimize_scalarで直接
    最大化し、or_second_resonance_locationの予測と比較するために使う。
    v_boundsを省略すると、命題6の予測値を中心にした狭い探索範囲を
    自動で使う(scipyのbounded探索が広すぎる範囲で境界に張り付く
    誤りを避けるため)。
    """
    from scipy.optimize import minimize_scalar

    a = c * xi

    if v_bounds is None:
        v_guess = or_second_resonance_location(xi, c)
        v_bounds = (max(v_guess - 5.0, 0.05), v_guess + 5.0)

    def neg_grad_mag_sq(v):
        b = v * xi
        p = (1 - _reg(a, xi)) * (1 - _reg(b, xi))
        ga = _reg_prime(a, xi) * (1 - _reg(b, xi)) * _reg_prime(p, xi)
        gb = _reg_prime(b, xi) * (1 - _reg(a, xi)) * _reg_prime(p, xi)
        return -(ga ** 2 + gb ** 2)

    res = minimize_scalar(neg_grad_mag_sq, bounds=v_bounds, method="bounded",
                           options={"xatol": 1e-13})
    return float(res.x)


def or_misclassification_boundary_sum(xi):
    """
    命題8の閉形式(v0.19): OR(a,b;xi)=0.5となる境界での u+v=a/xi+b/xi
    の値(xi->0の漸近極限)。

        u + v = (1/2)*ln(1/xi) + K,   K = (1/2)*ln(16/arctanh(1/sqrt(2)))

    この直線より内側(u+vが小さい)だと、a,bが両方とも非ゼロなのに
    OR(a,b;xi)は誤って0(偽)近くの値を返してしまう。
    """
    A = np.arctanh(1 / np.sqrt(2))
    K = 0.5 * np.log(16 / A)
    return 0.5 * np.log(1 / xi) + K


def or_misclassification_boundary_numeric(xi, ratio=1.0):
    """
    命題8を数値的に検証するヘルパー: u=ratio*v として、
    OR(ratio*v*xi, v*xi; xi) = 0.5 となるvをbrentqで求め、
    u+v(=v*(1+ratio))を返す。or_misclassification_boundary_sumの
    予測値(ratioに依らないはず)と比較するために使う。
    """
    from scipy.optimize import brentq

    def f(v):
        u = ratio * v
        a, b = u * xi, v * xi
        return _or_value(a, b, xi) - 0.5

    v_cross = brentq(f, 1e-8, 500.0)
    u_cross = ratio * v_cross
    return u_cross + v_cross


def _or_value(a, b, xi):
    NOT_a = 1 - _reg(a, xi)
    NOT_b = 1 - _reg(b, xi)
    return 1 - _reg(NOT_a * NOT_b, xi)


def or_value(a, b, xi=1.0):
    """OR(a,b;xi)の値そのもの(命題8の検証・可視化用の薄いラッパー)。"""
    return _or_value(a, b, xi)


def or_n_misclassification_K(n):
    """
    命題9のn項版の定数K(n) = (1/2)*ln(4^n / A), A=arctanh(1/sqrt(2))。
    n=2のときK(2)は命題8のKと一致する。
    """
    A = np.arctanh(1 / np.sqrt(2))
    return 0.5 * np.log(4 ** n / A)


def or_n_misclassification_boundary_sum(xi, n):
    """
    命題9の閉形式(v0.20): 融合版OR_n(a_1,...,a_n;xi)=0.5となる境界での
    sum_k(a_k/xi) の値(xi->0の漸近極限、全項が同程度の大きさの場合)。

        sum_k u_k = (1/2)*ln(1/xi) + K(n)
    """
    return 0.5 * np.log(1 / xi) + or_n_misclassification_K(n)


def or_n_value(*vals, xi=1.0):
    """融合版OR_n(a_1,...,a_n;xi)の値そのもの(命題9の検証用ラッパー)。"""
    prod = 1.0
    for v in vals:
        prod = prod * (1 - _reg(v, xi))
    return 1 - _reg(prod, xi)


def or_n_misclassification_boundary_numeric(xi, weights):
    """
    命題9を数値的に検証するヘルパー。weights(正の配列、和は問わない)に
    比例する形でu_kを配分し、sum(weights正規化後)*T が
    融合版OR_n(a_1,...,a_n;xi)=0.5 となる T(=sum u_kの実測値)を
    brentqで求める。or_n_misclassification_boundary_sumの予測値と
    比較するために使う。
    """
    from scipy.optimize import brentq

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    def f(T):
        vals = weights * T * xi
        return or_n_value(*vals, xi=xi) - 0.5

    return brentq(f, 1e-6, 2000.0)


def or_n_threshold_Cstar(xi):
    """命題10の閾値定数 C*(xi) = (1/2)*ln(4/xi)。命題6/8/9と同型のlog(1/xi)閾値。"""
    return 0.5 * np.log(4 / xi)


def or_n_fold_error_bound(margin):
    """
    命題10の誤差上界 exp(-4*margin)。少なくとも1つの値がC*(xi)+marginを
    超えていれば、naive foldとOR_n(融合版)の差はこの値程度に収まる
    (数値検証: xi=1e-2〜1e-8, n=2〜50, margin=0.5〜4で、実測誤差との
    比が margin>=2 でほぼ1.000に収束することを確認済み)。
    """
    return np.exp(-4 * margin)


def or_n_fusion_is_safe(values, xi, margin=3.0):
    """
    【命題10の判定版】naive foldとOR_n(融合版)が
    (or_n_fold_error_bound(margin)程度の精度で)一致するとみなせるか
    どうかを判定する。AND側のfusion_is_safe(命題3)のOR版に相当。

    条件: 値のうち少なくとも1つが |a_k| > xi*(C*(xi)+margin) を満たす
    (AND側が「全部」大きい必要があったのに対し、OR側は「どれか1つ」
    で足りる——ORの"どれか一つ真なら真"という性質に対応する)。

    戻り値: True なら naive foldと融合版がor_n_fold_error_bound(margin)
    程度の精度で一致するとみなせる。
    """
    values = np.abs(np.asarray(values, dtype=float))
    threshold = xi * (or_n_threshold_Cstar(xi) + margin)
    return bool(np.any(values > threshold))


def or_n_naive_fold(*vals, xi=1.0):
    """OR_nのnaive fold版(逐次的にOR(acc,次の値)を計算する)。命題10の検証用。"""
    def NOT(x):
        return 1 - _reg(x, xi)

    def OR2(x, y):
        return NOT(NOT(x) * NOT(y))

    acc = vals[0]
    for v in vals[1:]:
        acc = OR2(acc, v)
    return acc


def and_partial_dilation_invariance(a, b, xi, lam):
    """
    命題5の式(**): AND(lam*a, b; lam*xi) == AND(a, b; xi) を数値確認する。
    AND(a,b;xi)=reg(a*b;xi)がa*b/xiという単一の比だけの関数である
    ことの帰結。戻り値: (AND(lam*a,b;lam*xi), AND(a,b;xi)) のペア
    (理想的には一致する)。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lhs = _reg(lam * a * b, lam * xi)
    rhs = _reg(a * b, xi)
    return lhs, rhs


def or_breaks_partial_dilation_invariance(a, b, xi, lam):
    """
    命題5の式(***): OR(lam*a,b;lam*xi) と OR(a,b;xi) は一般には
    一致しないことを数値確認する。ORがa*bのような単一の比に
    還元できない(a/xiとb/xiを別々に必要とする)ことの帰結。
    戻り値: (OR(lam*a,b;lam*xi), OR(a,b;xi)) のペア(一般には不一致)。
    """
    def NOT(x, xi_):
        return 1 - _reg(x, xi_)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lhs = NOT(NOT(lam * a, lam * xi) * NOT(b, lam * xi), lam * xi)
    rhs = NOT(NOT(a, xi) * NOT(b, xi), xi)
    return lhs, rhs


def _core_gate_values(a, b, xi):
    """内部ヘルパー: core.pyと同じ定義でAND/OR/NAND/NOR/XOR/XNORを計算する。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def NOT_(x):
        return 1 - _reg(x, xi)

    AND_ = _reg(a * b, xi)
    OR_ = NOT_(NOT_(a) * NOT_(b))
    NAND_ = NOT_(AND_)
    NOR_ = NOT_(OR_)
    XOR_ = NOT_(NOT_(_reg(a * NOT_(b), xi)) * NOT_(_reg(NOT_(a) * b, xi)))
    XNOR_ = NOT_(XOR_)
    return {"AND": AND_, "OR": OR_, "NAND": NAND_, "NOR": NOR_, "XOR": XOR_, "XNOR": XNOR_}


def nand_threshold_ab(xi):
    """
    【命題11(v0.24)の一部】NAND(a,b;xi)=0.5 となる境界(a=bの対角線上)
    の閉形式(xi->0の漸近極限):

        a*b ~= sqrt(arctanh(1/sqrt(2))) * xi^(3/2)

    NAND=NOT(AND(a,b;xi);xi)のように、すでに[0,1)に収まっている
    確率的な値(ここではAND(a,b;xi))にもう一度NOT(=reg)を適用すると、
    AND自身の共鳴点(命題1のxi*u*、O(xi))とは異なる**xi^(3/2)という
    新しいスケール**が閾値として現れる。AND側(命題1、O(xi))・
    OR側(命題6/8/9、O(xi*log(1/xi)))に続く、**3つ目の異なる
    スケーリング則**。

    導出: NAND=0.5 <=> reg(AND;xi)=0.5 <=> AND=xi*A (A:=arctanh(1/sqrt2))。
    AND=tanh(ab/xi)^2=xi*A(微小量)なので、tanh(ab/xi)~=sqrt(xi*A)
    (小さい引数なのでtanh(w)~=w)、よってab~=xi*sqrt(xi*A)=sqrt(A)*xi^1.5。

    数値検証: xi=1e-2〜1e-10で、実際の境界とこの閉形式の比が
    1.00000に収束することを確認済み(`nand_threshold_numeric`参照)。
    """
    A = np.arctanh(1 / np.sqrt(2))
    return np.sqrt(A) * xi ** 1.5


def nand_threshold_numeric(xi):
    """
    命題11のnand_threshold_abを数値的に検証するヘルパー。a=bの対角線上で
    NAND(a,a;xi)=0.5となるa^2(=ab)の値をbrentqで求める。
    """
    from scipy.optimize import brentq

    def NOT_(x):
        return 1 - _reg(x, xi)

    def f(s):
        AND_ = _reg(s * s, xi)
        NAND_ = NOT_(AND_)
        return NAND_ - 0.5

    s_cross = brentq(f, 1e-12, 10.0)
    return s_cross ** 2


def gate_dilation_type(gate_name, a=1.1, b=-0.6, xi=0.5, lam=1.8, tol=1e-9):
    """
    【命題11: ゲートの分類】core.pyの6つのゲート(AND,OR,NAND,NOR,XOR,XNOR)
    それぞれについて、命題5の部分ディレーション不変性
    (gate(lam*a,b;lam*xi) == gate(a,b;xi)) を満たすかどうかを判定する。

    結果(sympyでの記号的確認・数値確認の両方で一致):
        AND  : 満たす(a*b/xiという単一の比だけの関数、命題5)
        NAND, OR, NOR, XOR, XNOR : いずれも満たさない

    満たさない5つのうち、NANDだけは「単一の鋭い閾値がxi^(3/2)という
    別のスケールにシフトするだけ」で、OR・NOR・XOR・XNORのような
    "log(1/xi)"型の広い誤分類領域(命題8)は持たない、という違いが
    ある(数値実験で確認。詳細はdev_notes.md v0.24参照)。

    戻り値: True なら"AND型"(部分ディレーション不変)、
    False なら"OR型"(不変性が崩れる)。
    """
    vals = _core_gate_values(a, b, xi)
    vals_scaled = _core_gate_values(lam * a, b, lam * xi)
    return bool(abs(vals[gate_name] - vals_scaled[gate_name]) < tol)


def nor_misclassification_boundary_sum(xi):
    """
    【命題12(v0.25)】命題8(ORの誤分類境界)と同じ手法をNORに適用した
    もの。NOR(a,b;xi)=0.5 となる境界を u=a/xi, v=b/xi の言葉で調べると、
    命題8と同じ「u+v=一定」という直線に(xi->0の漸近極限で)収束するが、
    定数項に**もう一段のlog(log)補正**が付く:

        u + v = (1/2)*ln(1/xi) + K_NOR(xi)
        K_NOR(xi) := (1/2)*ln(32 / ln(4/(xi*A))),   A := arctanh(1/sqrt(2))

    導出: NOR=0.5 <=> OR=xi*A(命題8のOR=0.5条件と同型の式をもう一段
    適用)。OR=1-reg(p;xi)=xi*Aを解くと、p/xi=arctanh(1-xi*A/2)、
    小さいepsilonについてarctanh(1-eps)~=(1/2)ln(2/eps)を使うと
    p/xi~=(1/2)ln(4/(xi*A))となり、これをp~=16*exp(-2(u+v))に代入して
    u+vについて解くと上式が出る。

    数値検証: xi=1e-2〜1e-12で、実際の境界と閉形式の差が着実に縮む
    ことを確認(xi=1e-12で差~1.8e-6)。u/v比を0.2〜5で振っても
    u+vはほぼ一定(9.40〜9.45、xi=1e-8)であることも確認済み
    (`nor_misclassification_boundary_numeric`参照)。

    NOR=NOT(OR)なので、これは命題8のORの誤分類領域をそのまま
    "裏返した"ものに過ぎない——a,bが両方とも非ゼロなのにNORが
    誤って1(both false)に近い値を返してしまう領域が、この直線の
    内側(u+vが小さい側)に広がっている。
    """
    A = np.arctanh(1 / np.sqrt(2))
    K_NOR = 0.5 * np.log(32 / np.log(4 / (xi * A)))
    return 0.5 * np.log(1 / xi) + K_NOR


def nor_misclassification_boundary_numeric(xi, ratio=1.0):
    """
    命題12を数値的に検証するヘルパー。u=ratio*v として、
    NOR(ratio*v*xi, v*xi; xi) = 0.5 となるvをbrentqで求め、u+vを返す。
    """
    from scipy.optimize import brentq

    def NOT_(x):
        return 1 - _reg(x, xi)

    def OR_(x, y):
        return NOT_(NOT_(x) * NOT_(y))

    def f(v):
        u = ratio * v
        a, b = u * xi, v * xi
        return NOT_(OR_(a, b)) - 0.5

    v_cross = brentq(f, 1e-8, 500.0)
    u_cross = ratio * v_cross
    return u_cross + v_cross


def xor_zero_cross_section_threshold(xi):
    """
    【命題13(v0.27): XORの断面(b=0)における新しいスケーリング則】

    XOR(a,0;xi) = NOT(NOT(reg(a;xi);xi);xi) という"二重NOT"の構造
    (b=0のとき、AND(a,NOT(0))=reg(a;xi)、AND(NOT(a),0)=0なので、
    XOR(a,0)=OR(reg(a;xi),0)=NOT(NOT(reg(a;xi))*1;xi)=NOT(NOT(reg(a;xi));xi)
    となる)を持つ。NAND(=NOT(AND)の"一重NOT")と同じ発想だが、
    ここではreg(a;xi)にさらにもう一段NOTが掛かる。

    XOR(a,0;xi)=0.5 となる境界を u=a/xi の言葉で解くと、xi->0の
    漸近極限で

        u ~= sqrt( (1/2)*xi*ln(4/(xi*A) ) ),   A := arctanh(1/sqrt(2))

    という、AND(O(xi))・OR(O(xi*log(1/xi)))・NAND(O(xi^1.5))に続く
    **4つ目の独立したスケーリング則**(xi^1.5に対数の平方根補正が
    掛かった形、a自体で見るとa~xi^1.5*sqrt(log(1/xi))相当)になる。

    導出: XOR(a,0;xi)=0.5 <=> NOT(reg(a;xi);xi)=xi*A(命題11のNAND
    と同じ式)。1-reg(x;xi)=xi*A(x:=reg(a;xi))を解くとx~=xi*(1/2)*
    ln(4/(xi*A))(NORの導出と同じepsilon近似)。x=tanh(a/xi)^2が
    この微小値に等しいので、tanh(a/xi)~=sqrt(x)(小さい引数なので
    tanh(w)~=wも使う)、よってa/xi~=sqrt(x)=sqrt((1/2)*xi*ln(4/(xi*A)))。

    数値検証: xi=1e-2〜1e-10で、実測境界との比が1.00000に収束する
    ことを確認済み(`xor_zero_cross_section_numeric`参照)。

    正直な限界: これはb=0に固定した特別な断面での結果であり、
    a,b両方が非ゼロな一般の場合のXORの誤分類領域(命題8のような
    閉じた2変数の式)はまだ導出できていない。
    """
    A = np.arctanh(1 / np.sqrt(2))
    target = xi * 0.5 * np.log(4 / (xi * A))
    return np.sqrt(target)


def xor_zero_cross_section_numeric(xi):
    """
    命題13を数値的に検証するヘルパー。XOR(u*xi, 0; xi) = 0.5 となる
    uをbrentqで求める。xor_zero_cross_section_thresholdの予測値と
    比較するために使う。
    """
    from scipy.optimize import brentq

    def NOT_(x):
        return 1 - _reg(x, xi)

    def AND_(x, y):
        return _reg(x * y, xi)

    def OR_(x, y):
        return NOT_(NOT_(x) * NOT_(y))

    def XOR_(x, y):
        return OR_(AND_(x, NOT_(y)), AND_(NOT_(x), y))

    def f(u):
        return XOR_(u * xi, 0.0) - 0.5

    return brentq(f, 1e-9, 10.0)


def xor_diagonal_lower_threshold(xi):
    """
    【命題14(v0.28)の一部: XOR(a,a;xi)の誤分類"帯"、下側の境界】

    XOR(a,a;xi)は定義上つねに0(偽)であるべき対称な量だが、実際には
    u=a/xiがある範囲(下側閾値<u<上側閾値)にあるとき、誤って1
    (真)に張り付いてしまう。この下側の境界(xi->0の漸近極限):

        u_lower ~= sqrt( xi * (1/2) * ln(4/sqrt(xi*A)) ),  A:=arctanh(1/sqrt(2))

    導出: XOR(a,a;xi)=OR(x,x;xi)、x:=AND(a,NOT(a);xi)=reg(a*NOT(a);xi)。
    OR(x,x;xi)=0.5を解くとNOT(x;xi)=sqrt(xi*A)になり(命題8のOR=0.5条件
    のx=y版)、これをxについて解くとx~=xi*(1/2)*ln(4/sqrt(xi*A))
    (NOT(x;xi)=1-tanh(x/xi)^2=sqrt(xi*A)という小さい目標値を、
    命題12と同様のarctanh(1-epsilon)近似で解く)。さらにx=reg(a*NOT(a);xi)
    をa*NOT(a)について解くと、小さいu(NOT(a)~=1相当)の極限で
    a*NOT(a)~=a=u*xiとなるので、u~=sqrt(x/xi)という上の式になる。

    数値検証: xi=1e-3〜1e-10で、実測境界との比が1.00000に収束する
    ことを確認済み(`xor_diagonal_threshold_numeric`参照)。
    """
    A = np.arctanh(1 / np.sqrt(2))
    x_target = xi * 0.5 * np.log(4 / np.sqrt(xi * A))
    return np.sqrt(x_target)


def xor_diagonal_upper_threshold(xi):
    """
    【命題14(v0.28)の一部: XOR(a,a;xi)の誤分類"帯"、上側の境界】

    下側閾値(xor_diagonal_lower_threshold)を超えると、XOR(a,a;xi)は
    誤って1に張り付く"帯"に入るが、uがさらに大きくなると、この帯を
    抜けて正しく0に戻る。この上側の境界は、**Lambert W函数**を使う
    閉形式になる(xi->0の漸近極限):

        R(xi) := (sqrt(xi)/4) * sqrt( (1/2)*ln(4/sqrt(xi*A)) )
        u_upper ~= -W_{-1}(-2*R(xi)) / 2

    (W_{-1}はLambert W函数の下側分岐。scipy.special.lambertw(z,k=-1)。)

    導出: xor_diagonal_lower_thresholdと同じくOR(x,x;xi)=0.5から
    x~=xi*(1/2)*ln(4/sqrt(xi*A))(下側閾値と同じ目標値)までは共通。
    ここで、a*NOT(a;xi)=reg^{-1}(x)を解く際に、**uが大きい**極限
    (NOT(a;xi)~=4*exp(-2u)、下側閾値の"uが小さい"極限とは逆側)を
    使うと、方程式が u*exp(-2u) = R(xi) という超越方程式になる
    (u*NOT(a)~=4*u*xi*exp(-2u)を、x~=xi*sqrt(x/xi)...ではなく
    tanh(y/xi)~=sqrt(x/xi)から出るy~=xi*sqrt(x/xi)に代入して整理)。
    w:=-2uと置換するとw*exp(w)=-2*R(xi)というLambert W函数の定義式
    そのものになり、大きい|w|側の解(W_{-1}分岐)がu_upperを与える。

    数値検証: xi=1e-3〜1e-10で、実測境界との差が着実に縮小する
    ことを確認済み(xi=1e-10で差~1.2e-6)。
    """
    from scipy.special import lambertw

    A = np.arctanh(1 / np.sqrt(2))
    R = (np.sqrt(xi) / 4) * np.sqrt(0.5 * np.log(4 / np.sqrt(xi * A)))
    w = lambertw(-2 * R, k=-1)
    return float((-w / 2).real)


def xor_diagonal_threshold_numeric(xi, which="lower"):
    """
    命題14を数値的に検証するヘルパー。XOR(u*xi,u*xi;xi)=0.5となる
    uをbrentqで求める。which="lower"なら下側の交差(u<1付近を探索)、
    which="upper"なら上側の交差(u>1付近を探索)を返す。
    """
    from scipy.optimize import brentq

    def NOT_(x):
        return 1 - _reg(x, xi)

    def AND_(x, y):
        return _reg(x * y, xi)

    def OR_(x, y):
        return NOT_(NOT_(x) * NOT_(y))

    def XOR_(x, y):
        return OR_(AND_(x, NOT_(y)), AND_(NOT_(x), y))

    def f(u):
        return XOR_(u * xi, u * xi) - 0.5

    if which == "lower":
        return brentq(f, 1e-9, 1.0)
    return brentq(f, 1.0, 40.0)


def xnor_zero_cross_section_threshold(xi):
    """
    【命題15(v0.29): XNORの断面(b=0)、二重対数のネスト構造】

    XNOR(a,0;xi) = NOT(XOR(a,0;xi);xi) の"=0.5"境界を、命題13
    (XORのb=0断面)と同じ手法をもう一段適用して求める。
    b=0のときXOR(a,0;xi)=NOT(NOT(reg(a;xi);xi);xi)(命題13)なので、
    XNOR(a,0;xi)=NOT(NOT(NOT(reg(a;xi);xi);xi);xi)という**三重NOT**
    の構造になる。

    導出(NOT(z;xi)=xi*A(A:=arctanh(1/sqrt(2)))という小さい目標値を
    逆算する、という同じ手順を3回繰り返すだけ):

        m := NOT(reg(a;xi);xi) の目標値 y ~= xi*(1/2)*ln(4/(xi*A))
        (これは命題13の"内側"の式そのもの)
        次に reg(a;xi) 自体の目標値 z ~= xi*(1/2)*ln(4/y)
        (NOT(reg(a;xi);xi)=yを逆算、命題12・NORの式と同型)
        最後に u=a/xi ~= sqrt(z) (reg(a;xi)=tanh(u)^2=zをuについて
        小さい引数の近似tanh(w)~=wで解く、命題13の最後の式と同型)

    3段のネストになるが、どの段も命題8以降で繰り返し使ってきた
    「NOT(z;xi)=小さい目標値、を逆算する」という同じ操作の反復に
    すぎない。

    数値検証: xi=1e-2〜1e-12で、実測境界との比が1.000000に収束する
    ことを確認済み(`xnor_zero_cross_section_numeric`参照)。
    """
    A = np.arctanh(1 / np.sqrt(2))
    y = xi * 0.5 * np.log(4 / (xi * A))
    z = xi * 0.5 * np.log(4 / y)
    return np.sqrt(z)


def xnor_zero_cross_section_numeric(xi):
    """
    命題15を数値的に検証するヘルパー。XNOR(u*xi,0;xi)=0.5となるuを
    brentqで求める。xnor_zero_cross_section_thresholdの予測値と
    比較するために使う。
    """
    from scipy.optimize import brentq

    def NOT_(x):
        return 1 - _reg(x, xi)

    def AND_(x, y):
        return _reg(x * y, xi)

    def OR_(x, y):
        return NOT_(NOT_(x) * NOT_(y))

    def XOR_(x, y):
        return OR_(AND_(x, NOT_(y)), AND_(NOT_(x), y))

    def XNOR_(x, y):
        return NOT_(XOR_(x, y))

    def f(u):
        return XNOR_(u * xi, 0.0) - 0.5

    return brentq(f, 1e-9, 10.0)


def not_composition_tower(x, n, xi):
    """
    命題16(v0.30)の基本演算: Psi_n(x;xi) := reg(x;xi)にNOTをn回
    繰り返し適用したもの。n=0ならAND型(reg自体)、n=1ならNAND型、
    n=2ならXOR(a,0)型、n=3ならXNOR(a,0)型に対応する
    (単一変数xの断面としての比較。詳細はnot_tower_thresholdの
    docstring参照)。
    """
    val = _reg(x, xi)
    for _ in range(n):
        val = 1 - _reg(val, xi)
    return val


def not_tower_threshold(n, xi):
    """
    命題16(NOT合成塔の統一理論): 命題11(NAND)・13(XOR b=0)・15(XNOR b=0)は
    「reg(x;xi)にNOTをn回繰り返した合成Psi_n(x;xi)が0.5になる閾値」という
    1つの一般式の特殊ケース(n=1,2,3)だった。逆算すると次の対数の塔になる:

        T_0 := xi*A                      (A := arctanh(1/sqrt(2)))
        T_k := xi*(1/2)*ln(4/T_{k-1})    for k=1,...,n-1
        x*_n := xi*sqrt(T_{n-1})           (n>=1)、x*_0 := xi*A

    導出: Psi_n(x)=0.5には Psi_{n-1}(x)=xi*A が必要(reg(z)=0.5<=>z=xi*A)。
    以降も同じ「NOT(z)=小さい目標値を逆算する」操作(epsilon近似)をn-1回
    繰り返し、最後にreg(x)=T_{n-1}を小さい引数の近似tanh(w)~=wで解く。

    n=1,2,3で命題11・13・15を厳密に再現し、n=4,5という名前のない
    "より深いNOTの入れ子"でも数値検証で成立を確認済み(xi=1e-2〜1e-40、
    mpmath多倍長精度)。float64実装なのでxi<1e-6あたりで丸め誤差が
    蓄積することがある(`not_tower_threshold_numeric`参照)。
    """
    if n == 0:
        A = np.arctanh(1 / np.sqrt(2))
        return xi * A

    A = np.arctanh(1 / np.sqrt(2))
    T = xi * A
    for _ in range(1, n):
        T = xi * 0.5 * np.log(4 / T)
    return xi * np.sqrt(T)


def not_tower_threshold_numeric(n, xi, bracket=None):
    """
    命題16を数値的に検証するヘルパー。not_composition_tower(x,n,xi)=0.5
    となるxを二分法(bisection)で求める。scipyのbrentqではなく
    自前の二分法を使うのは、xiが非常に小さいとき(や、nが大きい
    とき)にnot_composition_towerの値がfloat64の精度限界付近まで
    圧縮されてしまい、brentqの収束判定が不安定になることがある
    ため(開発時にmpmathの多倍長精度でこの問題を確認・回避した)。

    戻り値: 二分法で求めた閾値x*(float)。not_tower_thresholdの
    予測値と比較するために使う。
    """
    pred = not_tower_threshold(n, xi)
    lo, hi = pred * 1e-3, pred * 1e3

    def f(x):
        return not_composition_tower(x, n, xi) - 0.5

    flo = f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        fmid = f(mid)
        if (fmid > 0) == (flo > 0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return (lo + hi) / 2


def or_n_trigger_condition_is_not_necessary(xi, n_trials=20000, seed=0, margin=3.0):
    """
    【命題10の必要性についての追加調査(v0.34)】命題10の条件
    (少なくとも1つの値がxi*(C*(xi)+margin)を超える)は**十分条件**
    であって**必要条件ではない**ことを数値で確認するヘルパー。

    全部の値がこの閾値を下回るように乱数生成しても(=命題10の条件が
    成立しない状況を作っても)、naive foldとOR_n融合版が実際には
    高い確率で一致する(gapが小さい)ことを確認できる。逆に、
    Prop9(命題9)の"総和"の閾値との相関を見ても、良い一致と悪い
    不一致のグループでmargin(総和が閾値をどれだけ超えているか)の
    分布が大きく重なっており、単純な閾値では両者をきれいに
    分離できない——つまり「naive foldとOR_n融合版が一致するための
    必要十分条件」は、個々の値の最大値や総和のような単純な集計量
    だけでは決まらず、畳み込みの順序や個々の値の並び方に依存する、
    より複雑な条件である可能性が高い。

    戻り値: {"good_match_rate": 全部の値が閾値未満のときにgap<0.01と
    なる割合, "severe_mismatch_rate": 同条件でgap>0.5となる割合}
    (両方とも0でも1でもない、中間的な値になるはず——これが
    「命題10の条件が必要条件ではない」ことの数値的な確認)。
    """
    rng = np.random.default_rng(seed)
    C = or_n_threshold_Cstar(xi) + margin
    threshold = xi * C

    def NOT(x):
        return 1 - _reg(x, xi)

    def OR2(x, y):
        return NOT(NOT(x) * NOT(y))

    def naive_fold(vals):
        acc = vals[0]
        for v in vals[1:]:
            acc = OR2(acc, v)
        return acc

    def fused(vals):
        prod = 1.0
        for v in vals:
            prod = prod * NOT(v)
        return 1 - _reg(prod, xi)

    good, severe = 0, 0
    for _ in range(n_trials):
        n = rng.integers(3, 15)
        vals = rng.uniform(-threshold * 0.99, threshold * 0.99, n)
        gap = abs(naive_fold(vals) - fused(vals))
        if gap < 0.01:
            good += 1
        elif gap > 0.5:
            severe += 1

    return {"good_match_rate": good / n_trials, "severe_mismatch_rate": severe / n_trials}


def xor_2d_boundary_v_given_large_u(u, xi):
    """
    命題19の閉形式: XOR(a,b;xi)=0.5 の境界を、u=a/xi が大きいという
    前提のもとで v=b/xi について解いたもの。

        v ~= (1/2)*ln(4*u / u2(xi))

    u2(xi)はxor_zero_cross_section_threshold(xi)(命題13の閾値)。
    uが小さい(命題13・14の前提を外れる)場合はこの式は使えない
    ——uが十分大きい(目安として u2(xi)よりずっと大きい)ときの
    漸近的な閉形式であることに注意。
    """
    u2 = xor_zero_cross_section_threshold(xi)
    return 0.5 * np.log(4 * u / u2)


def xor_2d_boundary_numeric(u, xi):
    """
    命題19を数値的に検証するヘルパー。固定したu(=a/xi)について、
    XOR(u*xi, v*xi; xi) = 0.5 となるvをbrentqで求める。
    xor_2d_boundary_v_given_large_uの予測値と比較するために使う。
    """
    from scipy.optimize import brentq

    def NOT_(x):
        return 1 - _reg(x, xi)

    def AND_(x, y):
        return _reg(x * y, xi)

    def OR_(x, y):
        return NOT_(NOT_(x) * NOT_(y))

    def XOR_(x, y):
        return OR_(AND_(x, NOT_(y)), AND_(NOT_(x), y))

    def f(v):
        return XOR_(u * xi, v * xi) - 0.5

    return brentq(f, 1e-6, 50.0)


def xnor_2d_boundary_v_given_large_u(u, xi):
    """
    命題20の閉形式: XNOR(a,b;xi)=0.5 の境界を、u=a/xi が大きいという
    前提のもとで v=b/xi について解いたもの(命題19と同じ導出、
    定数がu2(命題13)からu3(命題15)に変わるだけ)。

        v ~= (1/2)*ln(4*u / u3(xi))

    u3(xi)はxnor_zero_cross_section_threshold(xi)(命題15の閾値)。
    """
    u3 = xnor_zero_cross_section_threshold(xi)
    return 0.5 * np.log(4 * u / u3)


def xnor_2d_boundary_numeric(u, xi):
    """
    命題20を数値的に検証するヘルパー。固定したu(=a/xi)について、
    XNOR(u*xi, v*xi; xi) = 0.5 となるvをbrentqで求める。
    xnor_2d_boundary_v_given_large_uの予測値と比較するために使う。
    """
    from scipy.optimize import brentq

    def NOT_(x):
        return 1 - _reg(x, xi)

    def AND_(x, y):
        return _reg(x * y, xi)

    def OR_(x, y):
        return NOT_(NOT_(x) * NOT_(y))

    def XOR_(x, y):
        return OR_(AND_(x, NOT_(y)), AND_(NOT_(x), y))

    def XNOR_(x, y):
        return NOT_(XOR_(x, y))

    def f(v):
        return XNOR_(u * xi, v * xi) - 0.5

    return brentq(f, 1e-6, 50.0)


def exact_not_log_identity(x, xi):
    """
    命題21の基本恒等式: -ln(NOT(x;xi)) = 2*ln(cosh(x/xi))
    (NOT(x;xi)=sech^2(x/xi)の定義から直接出る、厳密な恒等式)。

    identities.pyのintegral_of_k(x,xi)=xi*ln(cosh(x/xi))と厳密に
    一致する: -ln(NOT(x;xi)) = (2/xi)*integral_of_k(x,xi)。

    戻り値: -ln(NOT(x;xi))の値(2*ln(cosh(x/xi))と厳密に一致するはず)。
    """
    return 2 * np.log(np.cosh(np.asarray(x, dtype=float) / xi))


def or_n_exact_master_equation_lhs(values, xi):
    """
    命題21の厳密マスター方程式の左辺: sum_k ln(cosh(a_k/xi))。
    OR_n(values;xi)=0.5 <=> この値が (1/2)*ln(1/(xi*A)) に等しい
    (A:=arctanh(1/sqrt(2)))、という厳密な(近似ではない)関係になる。
    """
    values = np.asarray(values, dtype=float)
    return np.sum(np.log(np.cosh(values / xi)))


def or_n_exact_master_equation_rhs(xi):
    """命題21の厳密マスター方程式の右辺: (1/2)*ln(1/(xi*A))。"""
    A = np.arctanh(1 / np.sqrt(2))
    return 0.5 * np.log(1 / (xi * A))


def or_n_exact_threshold_last_value(fixed_values, xi, bracket=(1e-9, 1e4)):
    """
    命題21の厳密マスター方程式を使って、fixed_valuesに1つの値
    (u_lastに相当するa_n)を足したときにOR_n(...)=0.5となる
    その最後の値を、厳密に(近似なしで)brentqで解く。

    命題6・8・9・12・13・15・19・20のような漸近近似ではなく、
    xiやa_kの大きさに関係なく厳密に成り立つ値を返す
    (ただし解が正の実数の範囲に存在しない場合はValueErrorになりうる
    ——fixed_valuesだけで既に左辺が右辺を超えている場合など)。
    """
    from scipy.optimize import brentq

    rhs = or_n_exact_master_equation_rhs(xi)
    partial = or_n_exact_master_equation_lhs(fixed_values, xi) if len(fixed_values) > 0 else 0.0

    def f(u_last):
        return partial + np.log(np.cosh(u_last)) - rhs

    u_last = brentq(f, *bracket)
    return u_last * xi
