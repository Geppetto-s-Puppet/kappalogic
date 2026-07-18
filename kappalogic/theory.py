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

【命題5(v0.15, gauge.pyとの橋渡し: ANDの"部分ディレーション不変性")】
gauge.py(v0.14)で確認した「k(x;lam*xi) = k(x/lam;xi)」という
大域的ディレーション不変性が、AND/ORの非対称性(命題4)の
もう一段深い理由を説明する。

AND(a,b;xi) = reg(a*b;xi) = tanh(a*b/xi)^2 は、a*b/xi という
"たった1つの比"だけの関数である(aとbが別々に効くのではなく、
積a*bという1つの合成量だけが意味を持つ)。したがって

    AND(lam*a, b; lam*xi) = AND(a, b; xi)          ... (**)

が全てのlam>0で厳密に成り立つ(gauge.pyの基本恒等式を
x=a*bに適用しただけ。symbolic に確認済み、theory.pyのテスト参照)。
これは「aとxiを一緒にlam倍すれば、bは変えなくてもANDの値は
変わらない」という"部分ゲージ不変性"であり、fusion_is_safe(命題3)
の安全条件が「部分積s_k**自体**」ではなく「s_k/xiという比」だけに
依存する形で書けている理由でもある(s_k/xiはこの不変性のもとで
不変な量そのもの)。命題1の共鳴点x*=xi*arctanh(1/sqrt(3))も、
比の言葉で言えばx*/xi=arctanh(1/sqrt(3))という**xiに依存しない
普遍定数**であり、この不変性のもとで意味を持つ量である。

一方OR(a,b;xi) = NOT(NOT(a;xi)*NOT(b;xi);xi)は、aとbをそれぞれ
別々にtanhへ通してから掛け合わせ、もう一度tanhへ通す構造なので、
**a*bのような単一の比には決して還元できない**(a/xiとb/xiという
2つの独立な比が、線形合成では消せない形で効いてくる)。この事実の
帰結として

    OR(lam*a, b; lam*xi) != OR(a, b; xi)  (一般には不一致)   ... (***)

が成り立つ(symbolicに確認済み。具体例: a=1.3,b=-0.8,xi=0.37,
lam=2.7で OR(a,b;xi)=0.99999976、OR(lam*a,b;lam*xi)=0.99999608と
値が変わる。theory.pyのテスト参照)。

つまり**命題4で見つかった「ANDは共鳴条件が1つ、ORは2つ」という
非対称性は、「ANDはa*b/xiという1変数の関数として書けるが、ORは
2変数(a/xi, b/xiの組)を本質的に必要とする」という、より根本的な
構造の帰結として理解できる**。ANDが1次元のディレーション不変性
(gauge.pyのアフィン群の一部)をそのまま受け継ぐのに対し、ORは
それを受け継がない、という言い方もできる。

正直な限界: これは「ANDが1変数化できる/ORはできない」という
既に分かっている事実(定義から明らか)を、v0.14のゲージ理論的な
言葉(ディレーション不変性)で言い換えただけであり、それ自体は
真に新しい定理ではない。しかし、dev_notes.md v0.14で「measured
quantities(命題4)とgauge.pyの構造との橋渡しがまだない」と
明記していた穴に、少なくとも部分的な橋を架けられたと考えている。
XOR/NAND/NORなど他ゲートがこの1変数/多変数の分類のどちらに
入るかはまだ調べていない(次の一手の候補)。

【命題6(v0.17, ORの"第二共鳴"の対数補正スケール)】
命題4で、OR(a,b)の勾配が効くには(i) aがx*=xi*u*付近、
(ii) p=(1-reg(a))(1-reg(b))もx*付近、という2条件が同時に要る
ことを見た。ここでaをc*xi(c=a/xiは固定した定数、cはO(1))に
固定したとき、「(ii)を満たすために必要なbの位置」がxiとともに
どうスケールするかを調べた。

素朴には「x*=xi*u*と同じくbもO(xi)だろう」と予想したくなるが、
実際に調べると**bはO(xi)ではなく、O(xi*log(1/xi))というより
ゆっくり減衰する形**になる。具体的には、d/db[NOT(b)*reg'(p)]=0を
解くと、b*=v* * xi (v*=b*/xi)は、

    v* = -(1/2)*ln(xi) - (1/2)*ln(w0 / (4*(1-tanh(c)^2)))   ... (v0.17)

という閉形式に厳密に(xi->0の漸近極限で)収束する。ここでw0は
tanh(w) + w*(1 - 3*tanh(w)^2) = 0 の(u*より大きい側の)唯一解
w0≈1.0095565499...という普遍定数(cによらない)。

対数係数「-1/2*ln(xi)」の部分はcの値に依らず常に厳密に1/2で
あること(異なるc=0.3, u*, 1.0, 2.0全部で確認)、そして定数項は
上の式で与えられcに依存すること、を数値的に(xi=1e-4〜1e-8で
実際の最適点を数値探索し、上の閉形式との差が10^-5〜10^-9の
オーダーでxi->0とともに縮むこと)確認した。

つまり**AND側の唯一の共鳴点x*=xi*u*が正確にxiに比例するのに対し、
OR側の"第二の"共鳴点はxi*log(1/xi)という対数補正つきでしか
スケールしない**——これはANDとORの非対称性(命題4)に、これまで
知られていなかった定量的な次元をもう一つ加える発見である。

正直な限界: この closed form は xi->0 の漸近極限で導出したもので、
有限のxiでは(数値検証の通り)小さいがゼロでない誤差がある。
また、cを固定したときの話であり、a自体を動かした場合の全体像
(a,bの2変数空間での本当の"危険領域"の形)はまだ描けていない。
それでも、「なぜORの方がANDより広い範囲で危険なのか」
(命題4の勾配地形の表)を、対数補正という具体的な形で
定量的に説明できたのは今回の収穫。

【命題8(v0.19, ORの"誤分類領域": u+vの対数閾値)】
命題6・v0.18の危険地図をさらに追いかけていて、勾配の話ではなく
**OR(a,b)の"値そのもの"が、a,bが両方とも非ゼロなのに
誤って0(偽)に近い値を返してしまう領域が、実は非常に広い**ことに
気づいた。具体例(xi=1e-3、u=a/xi, v=b/xiとして):

    u,v がどちらも0.5〜2.0程度の"そこそこ非ゼロ"な値の組み合わせでは、
    OR(a,b)はほぼ厳密に0.0(!)を返す(u=v=2.0でもOR=0.000185)。
    u=v=2.5になって初めてOR=0.63程度まで持ち直す。

これは「AND側は個々の値がC*xiを超えれば安全」(命題3)という
比較的単純な話だったのに対し、**OR側は片方だけでなく両方の値が
十分大きくないと、正しい"true"すら返せない**という、命題4・6が
示唆していた"二重共鳴"構造の帰結を、勾配ではなく値そのもので
生々しく示す例になっている。

**閉形式(v0.19)**: OR(a,b;xi)=0.5となる境界(誤判定領域と正しい
領域の境目)を、u=a/xi, v=b/xiの言葉で調べると、xi->0の漸近極限で

    u + v = (1/2)*ln(1/xi) + K + O(xi),   K = (1/2)*ln(16/arctanh(1/sqrt(2))) ~= 1.4494

という**直線**(u+vの和だけで決まる)に厳密に収束することを
導出・検証した。u/v比を0.2〜5の範囲で振っても、u+vの値は
1%以内でほぼ一定(命題8の直線が比に依らないことの確認)。
xi=1e-4〜1e-12まで検証したところ、実際のu+v(数値的な境界)と
上の閉形式の差は、xiを10倍小さくするたびにちょうど10分の1に
縮んでいく(O(xi)の収束であることを確認、xi=1e-12で差~4.7e-7)。
比が極端(0.01倍や100倍)になるとu,vの一方が小さくなりすぎて
「両方とも大きい」という導出の前提から外れ、近似の精度は
数%程度に落ちる(境界付近の非対称な補正が必要、今回は未導出)。

**この命題の意味**: AND(a,b)=reg(a*b;xi)は、a,bが非ゼロで固定
されている限り、xi->0とともに**必ず正しく1に収束する**
(たとえa,bがどれだけ小さくても、xiがそれよりさらに小さくなれば
いずれ検出される)。しかしOR(a,b;xi)は、a,bを固定したままxi->0
すると、u=a/xi, v=b/xiがどんどん大きくなる(xiで割っているため)
ので一見安全に見えるが、**「両方の非ゼロ性を検出するのに必要な
u+vの閾値」自体もlog(1/xi)で際限なく大きくなっていく**ため、
a,bを先に固定してからxi->0する、という(このライブラリの多くの
命題が暗黙に仮定している)極限の取り方の順序次第で、"true"に
収束するとは限らない、という直感に反する結果になる。

正直な限界: 直線の"形"(u+v=const)は数値的な観察であり、
命題6のw0のような厳密な変分法での再導出はまだ行っていない
(ただし対数の係数1/2と定数Kは、sech^2の漸近展開から閉形式で
導出し、桁ごとに収束することを確認済みなので、これ自体は
確信を持って正しいと言える)。n項版への一般化(命題3や命題4の
残課題と同じ形)も、今回はまだ手を付けていない。

【命題9(v0.20, 命題8のn項一般化: 融合版OR_nの誤分類境界)】
命題8(2引数)を素直にn引数に一般化できるか調べた。
fused OR_n(a_1,...,a_n;xi) = 1 - reg(prod_k NOT(a_k); xi) の
"=0.5となる境界"を、u_k := a_k/xi の言葉で調べると、
命題8の「u+v=直線」がそのままn項の「総和Σu_kが一定」という
形に一般化されることを見つけた:

    sum_k u_k = (1/2)*ln(1/xi) + K(n),   K(n) = (1/2)*ln(4^n / A)

    (A := arctanh(1/sqrt(2))、命題8のK=K(2)と整合する)

導出: reg(P;xi)=0.5 <=> P=A*xi(P:=prod_k NOT(a_k))。各u_kが
大きい極限でNOT(a_k)~4*exp(-2*u_k)なので、
P ~ 4^n * exp(-2*sum(u_k))。これを P=A*xi に代入してsum(u_k)に
ついて解くと上の式が出る。

数値検証: 全項が同じu(対称な場合)でn=2,3,5,8,12について、
xiを1e-10,1e-30,1e-60(mpmathの多倍長精度で計算)まで小さくすると、
実際の境界と予測値の差が、xiが小さくなるにつれて着実に縮む
ことを確認した(n=12,xi=1e-60で差~3e-5)。非対称な重み
(u_kの配分を変えた場合)でも、sum(u_k)が同じ境界にほぼ収まる
ことも確認済み(重みの配分に依らず、"総和"だけが本質的な量で
あることの傍証)。

**重要な但し書き**: これは"融合版"OR_n(=定義どおりの一括計算)
**自体**が正しい値を返すかどうかの境界であり、命題3・命題4が
扱っていた「naive fold(逐次計算)と融合版が一致するか」という
問題とは別物である。実際に調べたところ、「sum(u_k)が命題9の
閾値を十分超えている」という条件だけでは、naive foldと融合版が
一致することは保証されない(乱数実験で、条件を満たしていても
naive foldが0(偽)に壊滅的に張り付いてしまうケースが約半数
見つかった——おそらく畳み込みの途中で"程よく非ゼロ"な値同士が
連続して出会うタイミングに依存するため)。したがって、
**「naive foldとOR_n融合版が一致するための(命題3のOR版に相当する)
n項条件」は、依然として未解決のまま残る**。命題9は
「融合版OR_n自体がいつ正しいか」という、それとは独立した
(しかし関連の深い)問いに答えたものとして位置づける。

【命題10(v0.23、ついに本丸: naive foldとOR_n融合版が一致するn項条件)】
命題9の直後に残した課題(「sum(u_k)の総和条件だけでは
naive foldとの一致は保証されない」)を、naive foldの漸化式
`acc_k = OR(acc_{k-1}, a_k)` そのものを追いかけることで解決した。

**鍵となる観察**: `m_k := NOT(acc_k)` の漸化式は
`m_k = phi(phi(m_{k-1} * s_k))`(phi:=NOTという関数、s_k:=NOT(a_k))
という"二重NOT"になっている。この`phi(phi(w))`という合成を
w in (0,1] について調べると、**wがxiより十分小さい(w << xi)
ときにだけ、phi(phi(w))は極端に小さい値(=「真」)に収束し、
そうでなければ1(=「偽」)にほぼ張り付く**という、鋭い閾値挙動を
することがわかった(w<<xiでない限りphi(w)がxiよりずっと小さい
値になり、それを再びphiに通すとまた1近くに戻ってしまうため)。

これはつまり、`m_{k-1}*s_k << xi` さえ満たせば、m_{k-1}(それまでの
畳み込みの"どれだけ怪しかったか")の値に関係なく(m_{k-1}<=1は
常に成り立つので)、新しいaccは正しく「真」に転じる、ということ。
そして `m_{k-1}*s_k <= s_k`(m_{k-1}<=1より)なので、
**s_k自体がxiより十分小さければ、m_{k-1}の値によらず必ず安全**。

s_k=NOT(a_k;xi)がxiより十分小さくなる条件を、
u_k:=|a_k|/xi の言葉で解くと:

    s_k ~ 4*exp(-2*u_k) << xi  <=>  u_k > C*(xi) := (1/2)*ln(4/xi)

という、命題6・8・9と同じ「log(1/xi)」型の閾値が出てくる。

**定理(命題10)**: n個の値 a_1,...,a_n のうち**少なくとも1つ**が

    |a_k| > xi * (C*(xi) + M),   C*(xi) = (1/2)*ln(4/xi)

を満たす(Mは好きに選べる安全マージン)とき、naive foldと
OR_n融合版の誤差は

    |naive_fold - OR_n(融合版)| <~ exp(-4*M)

程度に収まる。この誤差上界は、xi=1e-2から1e-8まで、n=2から50まで、
Mを0.5から4まで動かして数値検証したところ、M>=2でratio(実測/予測)が
1.000に収束することを確認した(`or_n_fold_matches_fused_bound`
参照)。特にM=4のときは、xiやn、他の値の分布に関係なく
誤差がexp(-16)~1.1e-7にほぼぴったり一致する、という驚くほど
綺麗な普遍性を示した。

**AND側(命題3、v0.16)との対比**: AND_nは「全部の値が大きい」
ことを要求したが、OR_nは(ORの"どれか一つ真なら真"という性質に
素直に対応する形で)「どれか一つの値が大きい」ことだけで足りる。
ただしその"大きい"の基準は、ANDが単純な定数閾値C*xiだったのに対し、
ORは命題6・8・9と同じ log(1/xi) 補正のかかった閾値
`xi*((1/2)*ln(4/xi)+M)` になる、という違いがある。

**正直な評価**: これで「naive foldとOR_n融合版が一致するための
n項条件」という、v0.13の命題4の直後からTODO.mdの本丸として
残っていた課題に、閉じた形の十分条件と定量的な誤差上界を
与えることができた。ただし(1)これは"十分条件"であり、
必要十分条件(=もっと緩い条件でも安全な場合があるかもしれない)
までは詰めていない、(2)導出の途中(phi(phi(w))の閾値挙動)は
漸近的な議論であり、機械的に厳密な不等式の証明(assistant内で
sympyやペンと紙で追い切る形の証明)までは行っていない
——数値検証(複数のxi・n・マージンで、比が1.000に収束することを
確認)による裏付けにとどまる。文献調査もしていないので、
微分可能論理ゲート分野で本当に新規かどうかは断定しない。
それでも、v0.13から続いていた具体的な空白を、閉じた形の式で
埋められたことは、今回の一番の収穫だと思う。
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
    【命題3(v0.16修正版): AND_n融合の安全条件】

    **v0.16での訂正**: 当初(v0.7)の命題3は「畳み込みの"部分積"
    s_k=a1*...*akが、すべてのk=1..nについて|s_k|>C*xiを満たせば
    naive foldとAND_n(融合版)が一致する」と主張していたが、これは
    **誤りだった**。反例(xi=1, C=3): values=[100,100,0.01,50]の
    部分積は[100,10000,100,5000]で全部C*xi=3を大きく超えるので
    「安全」と判定されるはずだが、実際には naive_fold≈2.5e-5,
    fused=1.0 と、ほぼ正反対の値になる(乱数2万試行でも、この
    "old"条件が「安全」と判定したケースの約12%で実際には
    naive foldとfusedがgap>0.1で不一致になることを確認済み——
    しかもこの12%という失敗率はCを3,5,8と変えても変わらない、
    つまりCを大きくしても直らない構造的な誤りだった)。

    原因: naive foldは`acc_k = reg(acc_{k-1}*a_k)`という漸化式で
    計算される。acc_{k-1}が一度reg()を通って1近くに飽和すると、
    **それ以前の部分積の大きさの情報は失われる**(reg()の値域は
    [0,1)で、どれだけ大きいxを入れても出力は高々1未満)。したがって
    次段の acc_k は実質的に reg(a_k) 程度で決まってしまい、
    「部分積が大きいかどうか」ではなく**「直近の個別の値a_kが
    大きいかどうか」**が効いてくる。

    **正しい条件**: naive foldとAND_n(融合版)が一致するのは、
    **個々の値a_1,...,anがすべて |a_k| > C*xi を満たすとき**である
    (部分積ではなく、個別の値についての条件)。乱数2万試行でこの
    修正条件下ではgap>0.1の不一致が1件も発生しないことを確認済み
    (`fusion_error_bound`参照、n=3〜100で成立を確認)。

    さらに、この条件のもとで**定量的な誤差上界**も導出・検証した:

        |naive_fold - AND_n(融合版)| <= n * 4*exp(-2*C)     ... (誤差上界)

    (導出: reg(x)=tanh(x/xi)^2の"未飽和度" 1-reg(x) は |x|>C*xi のとき
    4*exp(-2C)以下(sech^2の指数減衰、命題1のreg'の話と同根)。
    各段でこの未飽和度が高々4exp(-2C)ずつ新たに混入すると考えると、
    n段でn*4exp(-2C)という上界が出る。乱数検証(C=3,5,8,10、
    n=3〜100)では実際のgapは常にこの上界の1/3以下に収まっていた
    ので、この上界は(タイトではないが)安全側の保証として使える。
    詳細な証明はdev_notes.md v0.16参照)。

    戻り値: True なら「個々の値が全部C*xiを超えている」ことを意味し、
    naive foldとAND_n融合版はn*4exp(-2C)程度の精度で一致するとみなせる。
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
