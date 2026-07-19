"""
kappalogic.dynamics
=====================
応用21の最後に触れられていた「強引に定点をねじ込んだアベル関数」を
実際に構成する。

アベルの関数方程式 alpha(f(x)) = alpha(x) + 1 は、写像fの不動点まわりで
定義される(Koenigsの定理: 不動点x0での乗数lambda=f'(x0)が0,1でなければ、
線形化座標 beta(x) = lim_{n->inf} (f^n(x)-x0)/lambda^n が存在し、
beta(f(x))=lambda*beta(x) を満たす。alpha(x)=log|beta(x)|/log|lambda|
とすればアベルの方程式を満たす)。

このモジュールでは、元の写像fが「自然には不動点を持たない場所」x0に、
このライブラリのgaussian_match(ガウス核)を使って局所的なパッチを当て、
g(x) = f(x) + (x0-f(x0))*gaussian_match(x,x0,t) として不動点を強引に
埋め込む。パッチが対称なガウス核であれば、g'(x0)=f'(x0)となり
(gaussian_match'(x0)=0のため)、乗数を元の写像の情報からそのまま
受け継げる。

実例(f(x)=0.5x+x^2、自然な不動点は0と0.5、x0=-0.1に不動点を強引に埋め込む)
で、Koenigs方程式・アベル方程式の両方が数値的に検証済み(相対誤差1e-6〜1e-11、
n=18回反復、乗数lambda=0.3)。

注意点(実験で分かったこと):
- パッチの基底(その不動点に収束する初期値の範囲)は、パッチの幅tと同程度の
  狭い領域に限られる。元の写像がその近くに別の(より強い)吸引的不動点を
  持つ場合、基底はさらに狭くなる。
- 乗数lambdaが1に近いと反復の収束が遅く、多くの反復回数が必要になる。
  逆にlambdaが0に近すぎると、少ない反復回数で浮動小数点のアンダーフローが
  起きる。実用上は反復回数nを、lambda^nが機械精度(~1e-16)より
  大きく1e-6程度より小さい範囲に収まるよう選ぶ必要がある。

v0.31追記: theory.pyの命題16(NOT合成塔)のnを連続化できないか
(force_fixed_pointのような人工パッチなしで、fractional_iterateが
そのまま使えるか)調べていて、NOT(z;xi):=1-reg(z;xi) という写像
自体が**自然な不動点**を持つことに気づいた。ただしその不動点での
乗数は常に負(period-2的)であり、素朴な連続反復(lambda^tで実数tに
ついて補間する)はlambdaが負だと複素数が必要になり実現できない
(この点は本モジュールの既存の連続反復コードでは正の乗数を前提と
しているため、そのままでは使えない)。その代わり、乗数がちょうど
-1になる"自然な放物型不動点"を発見した(`not_map_critical_xi`)。
これはdev_notes.md v0.12で行き詰まった「人工的に埋め込んだ不動点の
放物型収束」とは別に見つかった、NOT写像そのものが固有に持つ構造。

v0.32追記: 上の"放物型不動点"(乗数-1)を、標準的な放物型繰り込み
理論(乗数+1を前提とすることが多い)に繋げるため、F:=NOT∘NOT
(NOTを2回)という合成写像を考えた。F'(z0)=(-1)^2=+1で標準的な
意味での放物型になるが、実際に調べたところ**F''(z0)が恒等的に0**
になり、通常の(2次で接する)放物型よりも縮退した"3次で接する"
放物型不動点になっていた。これは「乗数-1を持つ任意の写像を自分
自身と合成すると、必ず3次接触の放物型になる」という**一般的な
事実**であることをsympyで確認した(NOT写像固有の偶然ではない)。
力学的な帰結として、標準の放物型不動点の1/n減衰ではなく
**1/sqrt(n)減衰**という、質的に遅い収束をすることも数値的に
確認した(`squared_map_cubic_coefficient`, 
`squared_map_convergence_exponent`)。
"""
import numpy as np
from .heat import gaussian_match


def force_fixed_point(f, x0, patch_width=0.01):
    """
    f(x)に、x0という(自然にはf(x0)=x0を満たさない)点で
    強引に不動点を作った新しい写像 g(x) を返す。
    g(x) = f(x) + (x0-f(x0)) * gaussian_match(x,x0,patch_width)
    """
    correction = x0 - f(x0)

    def g(x):
        return f(x) + correction * gaussian_match(x, x0, patch_width)

    return g


def multiplier_at(g, x0, h=1e-6):
    """不動点x0でのg'(x0)(乗数lambda)を数値微分で求める。"""
    return (g(x0 + h) - g(x0 - h)) / (2 * h)


def koenigs_coordinate(g, x, x0, lam, n=20):
    """
    Koenigsの線形化座標 beta(x) = (g^n(x)-x0)/lambda^n の有限n近似。
    beta(g(x)) = lambda*beta(x) を満たすはず(|lambda|<1で前進反復、
    xがx0の吸引的基底内にあることが必要)。
    """
    xn = x
    for _ in range(n):
        xn = g(xn)
    return (xn - x0) / (lam ** n)


def abel_function(g, x, x0, lam, n=20):
    """
    アベル関数 alpha(x) = log|beta(x)| / log|lambda|。
    alpha(g(x)) = alpha(x) + 1 を満たすはず(force_fixed_pointで
    強引に埋め込んだ不動点まわりで、実際に数値検証済み)。
    """
    beta = koenigs_coordinate(g, x, x0, lam, n)
    return np.log(np.abs(beta)) / np.log(np.abs(lam))


def fractional_iterate(g, x, t, x0, lam, n=20, x_guess=None):
    """
    連続反復(分数反復) g^t(x)。アベル関数(Koenigs座標)を使い、
    g^t(x) = beta^{-1}(lambda^t * beta(x)) として、betaの根探しで逆算する。

    t=0でx自身、t=1でg(x)、t=2でg(g(x))と一致することを確認済み。
    さらにg^0.5を2回合成するとg(x)に一致し(半分だけ反復する、という
    直感的な意味が数値的に成立)、g^aをg^bに続けて適用するとg^(a+b)に
    一致する(分数反復の加法性)ことも確認済み(相対誤差~1e-9〜1e-14)。

    x_guess: 根探しの初期探索中心。省略時はxを使う(t=0付近で妥当)。
    """
    from scipy.optimize import brentq

    def beta(y):
        return koenigs_coordinate(g, y, x0, lam, n)

    target = (lam ** t) * beta(x)
    f = lambda y: beta(y) - target
    center = x_guess if x_guess is not None else x
    span = 0.02
    for _ in range(40):
        lo, hi = center - span, center + span
        if f(lo) * f(hi) < 0:
            return brentq(f, lo, hi, xtol=1e-13)
        span *= 1.5
    raise RuntimeError("fractional_iterate: bracket not found; try a different x_guess")


# ----------------------------------------------------------------------
# v0.31: NOT(z;xi)自身の"自然な"不動点とその放物型分岐
# ----------------------------------------------------------------------
# theory.py の命題16(NOT合成塔)で見つかったPsi_n(x;xi)(reg(x;xi)に
# NOTをn回繰り返す)を、nについて連続にできないか(force_fixed_pointの
# ような人工的なパッチなしで、fractional_iterateがそのまま使えるか)
# 調べていて、NOT(z;xi):=1-reg(z;xi) という写像自体が、実は
# **人工的なパッチを当てなくても自然な不動点を持つ**ことに気づいた。
# ただしその不動点での乗数(乗法)は常に負(period-2的な性質)であり、
# しかもある特別なxiでちょうど-1(放物型)になる、という発見があった。

def not_map_fixed_point(xi):
    """
    NOT(z;xi):=1-tanh(z/xi)^2 という写像の自然な不動点z0
    (NOT(z0;xi)=z0を満たす、(0,1)内の唯一の点)を求める。

    z=0でNOT(0;xi)=1>0、z->inftyでNOT(z;xi)->0<zなので、中間値の
    定理により(0,1)の中に必ず不動点がある(force_fixed_pointのような
    人工的なパッチは不要)。
    """
    from scipy.optimize import brentq

    def f(z):
        return (1 - np.tanh(z / xi) ** 2) - z

    return brentq(f, 1e-9, 10.0)


def not_map_multiplier(xi):
    """
    not_map_fixed_pointで求めた不動点z0での、NOT(z;xi)の乗数
    (微分係数)lambda = NOT'(z0;xi)を返す。

    どのxiについてもlambdaは**常に負**になることを確認済み
    (xi=0.1〜2.0で検証、period-2的な"往復"の性質を持つ不動点で
    あることを示している)。|lambda|はxiが大きくなるにつれて
    単調に減少し、xi_c~=0.7518(not_map_critical_xi参照)で
    ちょうど-1(放物型不動点)を通過し、それより大きいxiでは
    吸引的(|lambda|<1)、小さいxiでは反発的(|lambda|>1)になる。
    """
    z0 = not_map_fixed_point(xi)
    h = 1e-6

    def phi(z):
        return 1 - np.tanh(z / xi) ** 2

    return (phi(z0 + h) - phi(z0 - h)) / (2 * h)


def not_map_critical_xi():
    """
    【命題17(v0.31): NOT写像の自然な放物型不動点】

    not_map_fixed_pointの不動点での乗数が、ちょうど-1になる
    (放物型不動点、dynamics.pyの元々のテーマである"放物型"の
    自然な実例)臨界のxi_cを、閉形式で求める。

    導出: z0=NOT(z0;xi)=1-tanh(z0/xi)^2 と NOT'(z0;xi)=-1 の
    連立方程式を解く。1本目からtanh(z0/xi)^2=1-z0、すなわち
    tanh(z0/xi)=sqrt(1-z0)。NOT'(z)=-(2/xi)*tanh(z/xi)*sech^2(z/xi)
    に代入し(sech^2(z0/xi)=1-tanh(z0/xi)^2=z0を使う)、
    NOT'(z0;xi)=-(2/xi)*sqrt(1-z0)*z0=-1を得る。これは
    xi=2*z0*sqrt(1-z0)、かつ z0/xi=arctanh(sqrt(1-z0)) という
    2つの関係になる。s:=sqrt(1-z0)と置くと、この2式から

        2*s*arctanh(s) = 1                    ... (*)

    という、sだけの1変数の超越方程式に帰着する。(*)を解いて、

        z0_c = 1 - s^2,   xi_c = 2*s*(1-s^2) = 2*s*z0_c

    数値検証: s~=0.647918229、z0_c~=0.580201968、
    xi_c~=0.751846864。not_map_multiplier(xi_c)が厳密に-1
    (誤差<1e-10)になることを直接確認済み。

    この"自然な放物型不動点"は、dev_notes.md v0.12で行き詰まった
    「人工的に埋め込んだ不動点の放物型収束」の問題とは別に、
    NOT写像そのものが持つ固有の構造として見つかったもの。
    外部アドバイザーが指摘していた「放物型繰り込み」(Inou-Shishikura
    理論)との関係を調べる際の、具体的な足がかりになりうる。

    戻り値: (xi_c, z0_c, s) のタプル。
    """
    from scipy.optimize import brentq

    def g(s):
        return 2 * s * np.arctanh(s) - 1

    s = brentq(g, 1e-6, 1 - 1e-9)
    z0_c = 1 - s ** 2
    xi_c = 2 * s * z0_c
    return xi_c, z0_c, s


def squared_map_cubic_coefficient(xi):
    """
    【命題18(v0.32)の一部】F(z):=NOT(NOT(z;xi);xi)(NOTを2回、
    命題17の不動点z0での乗数がF'(z0)=NOT'(z0)^2になる合成写像)の、
    z0まわりのテイラー展開の**3次係数**b := F'''(z0)/6 を返す
    (sympyで厳密な導関数を計算してから数値評価する)。

    一般に、NOT'(z0;xi)=-1(命題17の臨界xi_c)のとき、F=NOT∘NOTは
    F'(z0)=(-1)^2=1(標準的な意味での放物型不動点)になるが、
    実は**F''(z0)は恒等的に0になる**(下記の一般補題を参照)。
    つまりF(z0+eps)=z0+eps+b*eps^3+O(eps^4)という、通常の(2次で
    接する)放物型不動点よりも縮退した"3次で接する"放物型不動点に
    なる。

    【一般補題(任意の写像に成り立つ、NOT写像に限らない事実)】
    phi'(z0)=-1を満たす任意の写像phi(z0+eps)=z0-eps+c*eps^2+d*eps^3+...
    に対し、F:=phi∘phiは F(z0+eps)-z0 = eps - 2*(c^2+d)*eps^3 + O(eps^4)
    となり、**eps^2の係数は常に厳密に0になる**(sympyで一般の
    c,dを使って記号的に確認済み)。つまり「乗数-1を持つ写像を
    自分自身と合成すると、必ず(少なくとも)3次接触の放物型不動点に
    なる」という一般的な事実であり、NOT写像固有の偶然ではない。
    """
    import sympy as sp

    z_sym, xi_sym = sp.symbols('z xi', positive=True)
    NOT_sym = 1 - sp.tanh(z_sym / xi_sym) ** 2
    F_sym = NOT_sym.subs(z_sym, NOT_sym)
    Fppp = sp.diff(F_sym, z_sym, 3)

    z0 = not_map_fixed_point(xi)
    val = float(Fppp.subs({z_sym: z0, xi_sym: xi}))
    return val / 6


def squared_map_quartic_coefficient(xi):
    """
    【命題19(v0.58): 命題18の続き——3次接触放物型不動点の対数/平方根
    補正項の正体】

    F(z):=NOT(NOT(z;xi);xi)のz0まわりのテイラー展開の**4次係数**
    c := F''''(z0)/24 を返す(squared_map_cubic_coefficientの
    3次係数bとペアで使う: F(z0+eps)=z0+eps+b*eps^3+c*eps^4+O(eps^5))。

    このcが、v0.33で「導出できなかった」としていたFatou座標の
    大域的な補正項の係数を決める(fatou_coordinate_correction_
    coefficient参照)。
    """
    import sympy as sp

    z_sym, xi_sym = sp.symbols('z xi', positive=True)
    NOT_sym = 1 - sp.tanh(z_sym / xi_sym) ** 2
    F_sym = NOT_sym.subs(z_sym, NOT_sym)
    Fp4 = sp.diff(F_sym, z_sym, 4)

    z0 = not_map_fixed_point(xi)
    val = float(Fp4.subs({z_sym: z0, xi_sym: xi}))
    return val / 24


def fatou_coordinate_correction_coefficient(xi):
    """
    命題19: u_n:=1/(F^n(z)-z0)^2 の厳密な漸化式を、3次接触の
    テイラー展開 F(z0+eps)=z0+eps+b*eps^3+c*eps^4+O(eps^5) から
    導出すると(b,cはsquared_map_cubic/quartic_coefficient):

        u_{n+1} = u_n - 2b - 2c/sqrt(u_n) + O(1/u_n)

    という関係になる(1/(1+x)^2の展開とu_n*w_n^2=1、u_n*w_n^3=
    sign(w_n)/sqrt(u_n)という恒等式を使うだけの初等的な計算)。

    u_n ~= -2b*n(標準的な3次接触の主要項)を代入して和を取ると、
    Sum_{n<=N} 1/sqrt(u_n) ~= 2*sqrt(N)/sqrt(2|b|) (Sum 1/sqrt(n)~=2sqrt(N)
    という標準的な漸近和を使う)ので、

        u_N + 2*b*N ~= -(4c/sqrt(2|b|)) * sqrt(N) + O(log N)

    という**平方根補正項**(標準の2次接触の放物型不動点で知られる
    対数補正項とは異なり、3次接触ではsqrt(N)補正になる)が出る。
    この関数はその係数 -(4c/sqrt(2|b|)) を返す。

    検証: xi=xi_c(命題17の臨界値)で、1ステップだけの厳密な関係
    u_{n+1}-u_n-(-2b) と 予測値-2c/sqrt(u_n) を、u_n~=7.6e5という
    大きな値のところで比較したところ、比が0.9926(ほぼ1)で一致した
    (mpmath 60桁精度、n=200000反復後)。この"1ステップ"レベルでの
    検証は非常に良く一致するが、これを多数ステップ**累積**した
    ときのu_N+2bNの挙動を直接fitで確認しようとすると、対数項
    (O(1/u_n)を足し合わせるとlog Nが出る)との分離が数値的に
    不安定になり、大域的な閉形式Fatou座標の完全な確立にはまだ
    至っていない——「1ステップの漸化式の係数は特定できたが、
    それを多数回反復した後の大域的な挙動を安定して数値確認する」
    という最後の一歩が残っている。

    正直な評価: v0.33で「未解決」としていた補正項の**正体(平方根
    補正であること、その係数が4次のテイラー係数cで決まること)**は
    今回特定できた。ただし多数回反復後の大域的な収束を数値的に
    安定して再現するには、対数項の係数も合わせて導出し、より高精度
    な数値実験(桁数・反復回数を増やす)が必要——完全な決着は
    まだついていない。
    """
    b = squared_map_cubic_coefficient(xi)
    c = squared_map_quartic_coefficient(xi)
    return -(4 * c) / np.sqrt(2 * abs(b))


def fatou_coordinate_local_step_check(xi, n_iterations=200000, eps0=0.01):
    """
    fatou_coordinate_correction_coefficientの"1ステップ"レベルでの
    検証ヘルパー。z0+eps0からF(z)をn_iterations回反復した後、
    さらに1回反復して u_{n+1}-u_n-(-2b) と -2c/sqrt(u_n) を比較する
    (mpmath高精度演算を使用)。

    戻り値: (実測のlocal_diff, 予測値, 比率(1に近いほど良い))
    """
    import mpmath as mp

    mp.mp.dps = 60
    xi_mp = mp.mpf(xi)
    z0 = mp.mpf(not_map_fixed_point(xi))
    b = squared_map_cubic_coefficient(xi)
    c = squared_map_quartic_coefficient(xi)

    def F(z):
        def NOT(x):
            return 1 - mp.tanh(x / xi_mp) ** 2
        return NOT(NOT(z))

    z = z0 + mp.mpf(eps0)
    for _ in range(n_iterations):
        z = F(z)
    u = float(1 / (z - z0) ** 2)
    z_next = F(z)
    u_next = float(1 / (z_next - z0) ** 2)

    local_diff = u_next - u - (-2 * b)
    predicted = -2 * c / np.sqrt(u)
    ratio = local_diff / predicted
    return local_diff, predicted, ratio


def squared_map_convergence_exponent(xi, eps0=0.01, n_values=(1e5, 1e6)):
    """
    【命題18(v0.32)】F(z):=NOT(NOT(z;xi);xi)を臨界xi(命題17のxi_c)で
    反復したとき、不動点z0からのずれ|F^n(z)-z0|が、標準的な
    (2次で接する)放物型不動点の場合の 1/n ではなく、**1/sqrt(n)**
    という、より遅い収束をすることを数値的に確認するヘルパー。

    これは、squared_map_cubic_coefficientで確認した「3次で接する
    (縮退した)放物型不動点」であることの力学的な帰結: 標準の
    2次接触ではF(z)=z+a*eps^2+...からeps_n~-1/(a*n)(1/n減衰)が
    出るが、3次接触ではF(z)=z+b*eps^3+...からeps_n~C/sqrt(n)
    (1/sqrt(n)減衰、bの符号が適切なら)という、質的に遅い収束になる。

    戻り値: n_valuesの最後の2点(デフォルトでn=1e5と1e6)における
    |F^n(z)-z0|の値から最小二乗的に求めた減衰指数p
    (|F^n(z)-z0| ~ n^p という近似で、標準放物型ならp~=-1、
    3次接触ならp~=-0.5になるはず)。
    """
    z0 = not_map_fixed_point(xi)

    def F(z):
        return (1 - np.tanh((1 - np.tanh(z / xi) ** 2) / xi) ** 2)

    n_values = sorted(int(v) for v in n_values)
    diffs = []
    z = z0 + eps0
    n_done = 0
    for target_n in n_values:
        for _ in range(target_n - n_done):
            z = F(z)
        n_done = target_n
        diffs.append(abs(z - z0))

    log_n = np.log(n_values)
    log_d = np.log(diffs)
    slope = (log_d[-1] - log_d[0]) / (log_n[-1] - log_n[0])
    return slope


def asymptotic_fatou_coordinate(z, xi):
    """
    命題18の続き(v0.33): 3次接触の放物型不動点(F:=NOT∘NOT、
    z0=not_map_fixed_point(xi)、b=squared_map_cubic_coefficient(xi))
    に対する、**局所的な**漸近アベル(Fatou)座標。

        phi(z) := -1 / (2*b*(z-z0)^2)

    標準的な放物型不動点の理論(F(z)=z+a*(z-z0)^2+...の場合は
    phi(z)~=-1/(a*(z-z0)))を、3次接触(a=0、b*(z-z0)^3が主要項)の
    場合に素朴に対応させたもの(w:=z-z0としたときdw/dn~=b*w^3という
    連続近似から、d(w^-2)/dn~=-2bを積分するだけで出る)。

    **局所的な**性質(phi(F(z))-phi(z) -> 1 as z->z0)は数値的に
    確認済み(eps0=1e-4でdiff~=1.00002、eps0を小さくするとfloat64の
    精度限界(NOTの二重合成による丸め誤差)にすぐ達してしまう)。

    **正直な限界**: 標準の放物型不動点の理論では、Fatou座標の
    正確な漸近形には対数補正項がつくことが知られている
    (phi(z) = -1/(a*(z-z0)) + beta*ln(...)/a + O(1))。今回の
    3次接触の場合も、同様の補正項があるはずだが、今回はまだ
    導出していない。実際、この閉形式だけを使ってF^n(z)を多数回
    (10万回以上)反復し phi(F^n(z))-n が一定値に収束するかを
    確認しようとしたところ、**収束しなかった**(nが大きくなるに
    つれて系統的にずれていく)。つまりこの閉形式は、F(z)とzが
    近いときの"局所的な"1ステップの振る舞いは正しく捉えているが、
    多数回反復した"大域的な"Fatou座標としてはまだ不完全であり、
    対数補正項(または、より高次の補正)を追加する必要がある。
    これは今後の課題として残す。
    """
    z0 = not_map_fixed_point(xi)
    b = squared_map_cubic_coefficient(xi)
    w = z - z0
    return -1 / (2 * b * w ** 2)


def asymptotic_fatou_coordinate_local_check(xi, eps0=1e-4):
    """
    asymptotic_fatou_coordinateの局所的な性質
    (phi(F(z))-phi(z) -> 1 as z->z0)を、z=z0+eps0という1点で
    数値的に確認するヘルパー。戻り値: phi(F(z))-phi(z)の値
    (1に近いほど良い近似。eps0を小さくしすぎるとNOTの二重合成の
    float64精度限界に達し、かえって精度が落ちることに注意
    ——実用上はeps0~1e-4程度が良い)。
    """
    z0 = not_map_fixed_point(xi)

    def F(z):
        return (1 - np.tanh((1 - np.tanh(z / xi) ** 2) / xi) ** 2)

    z = z0 + eps0
    return asymptotic_fatou_coordinate(F(z), xi) - asymptotic_fatou_coordinate(z, xi)
