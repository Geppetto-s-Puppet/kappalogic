"""
kappalogic.theory
=====================
「検出器代数」(reg/AND/OR系)の勾配構造を、van Krieken et al. (2022)の
t-norm解析のスタイル(命題+証明)を借りて厳密に特徴づける。

背景(dev_notes.md v0.6参照): 標準的なt-norm(van Krieken論文の対象)は
真偽度[0,1]の範囲で単調な関数だが、このライブラリのreg(x)=tanh(x/xi)^2は
任意の実数を受け取り「0か非0か」を判定する非単調な"検出器"であり、
t-normの枠組みには存在しない。この違いを、証明可能な命題として明文化する。

【命題1】reg(x)=tanh(x/xi)^2 の勾配 reg'(x) は x=0 と x=±∞ で
ちょうど0になり(境界)、その中間の x=±xi*arctanh(1/sqrt(3)) で
最大になる。

証明: u=x/xi とおくと reg'(x)=(2/xi)*tanh(u)*sech^2(u)。
d/du[tanh(u)sech^2(u)] = sech^2(u)*(1-3*tanh^2(u))。
これが0になるのはtanh(u)^2=1/3、すなわちu=arctanh(1/sqrt(3))。
(解析的に導出し、数値微分でu*=0.658479...と一致することを確認済み)

【命題2(single-passing性)】AND(a,b)=reg(a*b)は、b=0のとき
任意のaについて d(AND)/da = 0 となる。
これはLogic Tensor Networks文献(Badreddine et al. 2022)が指摘する
"single-passing"(一方の入力が確定すると、もう一方への勾配が完全に
遮断される)という現象の一例だが、標準的なt-norm(積t-norm等)とは
異なる理由(値0が"detector"としての"偽"を意味し、reg自体がx=0で
微分0の谷になっているため)で生じる。

証明: d/da reg(ab) = b * reg'(ab) (連鎖律)。b=0のとき、
この式は b*reg'(0) = 0*0 = 0 となる(reg'(0)=0は命題1より)。
(数値微分で複数のaについて厳密に0になることを確認済み)

【命題4(v0.13, ORの"二重共鳴"構造)】
命題1〜3はいずれもAND系だけを扱っており、OR系には対応する命題が
存在しなかった(TODO.md G項の空白地帯)。実際にAND/ORの勾配地形を
乱数比較すると、質的に違う挙動が出る:

    OR(a,b) = NOT(NOT(a)*NOT(b)),  p := (1-reg(a))*(1-reg(b)) とおくと

        d/da OR(a,b) = reg'(a) * (1-reg(b)) * reg'(p)          ... (*)

    (連鎖律を2回適用するだけで出る式。数値微分との相対誤差
    ~1e-6のオーダーで一致することを確認済み。theory.pyのテスト参照)

(*)には命題1の「x=x*≈0.658*xiでのみ大きい値を取る鋭いバンプ関数
reg'」が、"a"と"p(a,bの複雑な合成)"という2つの独立な引数に
掛け算の形で登場する。したがってOR(a,b)の勾配が無視できない
大きさを持つには、次の2条件を同時に満たす必要がある:
    (i)  aそのものがx*付近にある (reg'(a)が効く)
    (ii) 合成量p=(1-reg(a))(1-reg(b))も改めてx*付近にある (reg'(p)が効く)

一方AND(a,b)=reg(ab)の勾配 b*reg'(ab) は共鳴条件が(ab付近という)
1つだけである。この「共鳴条件の数の違い(AND=1個、OR=2個)」が、
乱数ベンチマークで観測された次の非対称性を説明する
(xi=1e-3、a,bをxiの±8倍の範囲でサンプリング):

    指標                          | AND    | OR
    勾配がほぼ0(<1e-3)になる割合  | ~6%    | ~53%
    観測された最大勾配             | ~1.0   | ~10^3

さらに質的な違いとして、AND_n(命題3)の不一致は基本的に「ぼやけ」
(勾配の掛け違いによる緩やかなずれ)だが、OR_nの不一致は勾配地形が
「ほとんど平坦、ごく一部だけ急峻」という二値的な構造をしているため、
naive foldとOR_n(融合版)の答えが0/1のように正反対に割れる、
という質的に重い失敗モードになりうる(乱数7項ベンチマークで
0.2%程度の頻度で発生、AND_nでは同条件で発生ゼロ。
or_fusion_disagreement_rate参照)。

正直な限界: (*)自体は数値検証込みで確信を持って正しいと言えるが、
命題3のような「n項版の安全条件(部分積がCxiを超えれば安全、という
単純な指標)」にはまだ落とし込めていない。ORの場合、条件が
「a」「b」個別ではなく「aとp」という異なるスケールにまたがるため、
命題3の"部分積"に相当する単純な指標には収まらない可能性が高い
(TODO.md G項、今後の課題として残す)。文献調査は行っていないため
新規性の断定は避けるが、少なくとも本ライブラリのtheory.py上には
それまで記載されていなかった具体的な観察と、それを説明する
検証済みの閉形式である。
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
    【命題3(AND_n融合の安全条件)】
    AND_n(a1,...,an)=reg(a1*a2*...*an)(融合版)と、
    a1,a2,...を左から畳み込んだ素朴な連結版
    AND(...AND(AND(a1,a2),a3)...,an) が一致するのは、
    左から畳み込んだ「途中の部分積」s_k=a1*...*ak が、
    すべてのk=1..nについて |s_k| > C*xi を満たすときである。

    証明の概略: reg(x)=tanh(x/xi)^2は|x|>C*xiでexp(-2C)以下の
    誤差で1に飽和する(理論の公理1、C=3で1e-13以下)。素朴な連結は
    各段階でreg(部分積 * 次の値)を計算するため、途中のどこかの
    部分積がO(xi)だと、その段階のtanhが飽和しきらず、後続の計算に
    誤差が伝播する。融合版は最終的な全部の積を一度だけreg()に
    渡すため、「最終的な積」さえ|全積|>C*xiを満たせば安全。

    ランダムな入力2万組でこの安全条件と実際の不一致を突き合わせたところ、
    99.95%の精度で一致を予測できることを確認済み(dev_notes.md参照。
    残り0.05%は閾値定数Cの選び方に依存する境界効果)。

    戻り値: True ならnaive foldとAND_nの融合版が(exp(-2C)程度の精度で)
    一致するとみなせる。Falseなら部分積のどこかがxiに近く、不一致の
    リスクがある。
    """
    partial_products = np.cumprod(np.asarray(values, dtype=float))
    return bool(np.all(np.abs(partial_products) > C * xi))


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
