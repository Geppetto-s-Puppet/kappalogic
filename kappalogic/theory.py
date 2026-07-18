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

**(v0.26追記)** 当初は「cを固定したときの話であり、a自体を動かした
場合の全体像(a,bの2変数空間での本当の"危険領域"の形)はまだ描けて
いない」と書いていたが、これは誤解だった。実はor_second_resonance_
locationをcの**連続関数**として使えば、それがそのまま
(a,b)平面上の|grad OR|=sqrt((dOR/da)^2+(dOR/db)^2)の**真の尾根**に
なっている(数値精度10^-6〜10^-10のオーダーで一致することを確認、
`or_full_gradient_magnitude_argmax`参照)。v0.18で「近似曲線
(1-reg(a))(1-reg(b))=xi*u*は数%〜十数%の誤差でしか尾根に近似
できない」としていたが、命題6の閉形式を使えばこの誤差はほぼ解消
される。つまり命題6は、2変数空間の真の危険領域の形も(cを動かす
ことで)すでに与えていた、ということになる。

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

【命題11(v0.24): 6ゲートの分類——AND型かOR型か】
TODO.mdで積み残していた「XOR/NAND/NORなど他ゲートが、AND型
(a*b/xiという1変数に還元できる、命題5)かOR型(できない)か」を
調べた。core.pyの6ゲート全部について、命題5の部分ディレーション
不変性(gate(lam*a,b;lam*xi)==gate(a,b;xi))をsympy・数値の両方で
チェックしたところ:

    AND                        : 満たす(AND型)
    NAND, OR, NOR, XOR, XNOR   : いずれも満たさない(OR型寄り)

NAND=NOT(AND)のように、AND型のゲートに外側からNOTを1回被せる
だけで、この綺麗な不変性は壊れてしまう(NOT自身がxiに依存する
"もう一段のreg"だから)。

**ただし満たさない5つの中でもNANDだけは性質が違った**。NAND・OR・
NOR・XOR・XNORで実際にa,bを動かして誤分類領域を調べたところ:

    - OR, NOR, XOR, XNOR: 命題8と同じ「a,bが両方ともそこそこ
      非ゼロなのに広い範囲で誤判定する」領域を持つ(NORはORの
      誤分類領域をそのまま裏返しただけ、XORは内部でOR型の
      合成を使っているため同じ問題を引き継ぐ)。
    - NAND: 単一の鋭い閾値を持つ(AND型と同様、広い誤分類領域は
      **持たない**)。ただしその閾値はAND自身の共鳴点
      (命題1、O(xi))とは異なる、**新しいスケール**:

        a*b ~= sqrt(arctanh(1/sqrt(2))) * xi^(3/2)      (v0.24)

      という、xi^(3/2)というAND(O(xi))ともOR(O(xi*log(1/xi)))
      とも違う、**3つ目の独立したスケーリング則**であることを
      閉形式で導出・検証した(xi=1e-2〜1e-10で、実測境界と
      予測の比が1.00000に収束することを確認。`nand_threshold_ab`,
      `nand_threshold_numeric`参照)。

**正直な評価**: 「単一の比に還元できるか」という命題5の判定基準
だけでは、NANDのような"AND型の親戚だが閾値のスケールは違う"
ケースを見落としてしまうことがわかった。分類自体
(部分ディレーション不変性の有無)は厳密(sympyで確認)だが、
NORやXOR/XNORの誤分類領域の精密な閉形式(命題8のような)は
今回導出していない(定性的な確認にとどまる)。NANDのxi^(3/2)則は
導出・数値検証ともに確信を持てる。

【命題12(v0.25): NORの誤分類境界(命題8の直接の応用)】
命題11で「NORはOR型(命題8のような広い誤分類領域を持つ)」と
定性的に確認するにとどめていたが、命題8と同じ手法をそのまま
適用することで、NORについても閉じた形の境界を導出できた。

NOR(a,b;xi)=0.5 の境界は、u=a/xi, v=b/xiの言葉で

    u + v = (1/2)*ln(1/xi) + K_NOR(xi)
    K_NOR(xi) = (1/2)*ln(32/ln(4/(xi*A))),  A := arctanh(1/sqrt(2))

という直線に収束する(命題8のOR境界と同じ「u+v=一定」という形だが、
定数項K_NORにxiのlogのlog、という珍しい二重対数補正がつく)。

導出: NOR=0.5 <=> OR=xi*A(命題8のOR=0.5条件と同型の式を、
NOR=NOT(OR)という一段の入れ子越しに適用しただけ)。OR=1-reg(p;xi)=xi*A
を解くとp/xi=arctanh(1-xi*A/2)~=(1/2)ln(4/(xi*A))(小さいepsilonに
ついてarctanh(1-eps)~=(1/2)ln(2/eps)を使用)。p~=16*exp(-2(u+v))
に代入してu+vについて解くと上式になる。

数値検証: xi=1e-2〜1e-12で実測境界との差が着実に縮小(xi=1e-12で
差~1.8e-6)。u/v比を0.2〜5で振ってもu+vはほぼ一定(9.40〜9.45、
xi=1e-8)であることも確認した。

NOR=NOT(OR)なので、これは命題8のORの誤分類領域を単に"裏返した"
ものに過ぎず、数学的に新しい現象ではない。それでも命題11で
"定性的な確認にとどまる"と正直に書いていた課題に、閉じた形の
答えを出せたのは収穫。XOR/XNORについては、対角線a=b上で単純な
"0.5交差"という形にならない(対称性のため、XOR(a,a)は別の構造を
持つ)ことが分かり、今回は同じ手法をそのまま適用できなかった
——引き続き未解決のまま残す。

【命題13(v0.27): XORの断面(b=0)における4つ目のスケーリング則】
対角線a=bが使えなかったXORについて、別の断面(b=0固定)を試した。
b=0のとき、XOR(a,0;xi) = NOT(NOT(reg(a;xi);xi);xi) という
"二重NOT"の構造になる(NAND=NOT(AND)の"一重NOT"よりもう一段深い)。

XOR(a,0;xi)=0.5の境界をu=a/xiで解くと、xi->0の漸近極限で

    u ~= sqrt( (1/2)*xi*ln(4/(xi*A)) ),   A := arctanh(1/sqrt(2))

という、AND(O(xi))・OR(O(xi*log(1/xi)))・NAND(O(xi^1.5))に続く
**4つ目の独立したスケーリング則**が出てきた。xi=1e-2〜1e-10で、
実測境界との比が1.00000に収束することを確認した
(`xor_zero_cross_section_threshold`, `xor_zero_cross_section_numeric`参照)。

面白いのは、この閾値がAND自身の共鳴点(命題1、O(xi)、a=xi*u*)
よりもさらに**小さい**a(xi^1.5オーダーの補正込みなので、
xiが小さいほどAND自身の閾値よりずっと手前で反応する)で切り替わる
ことである。これは"二重NOT"がAND単体よりも感度を上げる方向に
働く(NAND・NORで見た"もう一段NOTを重ねると閾値がシフトする"
という現象の、また違う現れ方)ことを示している。

正直な限界: これはb=0に固定した特別な断面での結果であり、a,b
両方が非ゼロな一般の場合のXORの誤分類領域(命題8のような閉じた
2変数の式)はまだ導出できていない。XNORについても同様の手法が
使えるはずだが、今回は時間の都合で手を付けていない。

【命題14(v0.28): XOR(a,a)の誤分類"帯"とLambert W函数の登場】
命題13(b=0断面)の直後、対角線a=b自体も実は攻略できることに
気づいた。XOR(a,a;xi)は定義上つねに0(偽)であるべき対称な量
だが、実際にはuが**ある帯の中にあるとき**だけ誤って1(真)に
張り付くことが分かった:

    u_lower ~= sqrt( xi * (1/2) * ln(4/sqrt(xi*A)) )          (下側境界)
    u_upper ~= -W_{-1}(-2*R(xi)) / 2,
        R(xi) := (sqrt(xi)/4) * sqrt( (1/2)*ln(4/sqrt(xi*A)) )  (上側境界)

    (A := arctanh(1/sqrt(2)) 、W_{-1}はLambert W函数の下側分岐)

u<u_lowerとu>u_upperでは正しく0、u_lower<u<u_upperの帯の中でだけ
誤って1になる。導出はXOR(a,a;xi)=OR(x,x;xi)(x:=AND(a,NOT(a);xi))
という単純化から始まり、OR(x,x;xi)=0.5を解いてx自体の目標値を求め
(下側・上側で共通)、それをx=reg(a*NOT(a);xi)について解く際に
「uが小さい極限」(下側境界、NOT(a)~=1)と「uが大きい極限」
(上側境界、NOT(a)~=4*exp(-2u))のどちらを使うかで、それぞれ
違う閉形式が出る。上側境界の方程式 u*exp(-2u)=R(xi) は、
w:=-2uと置換するとLambert W函数の定義式 w*exp(w)=-2R(xi) その
ものになり、**このライブラリで初めてLambert W函数が登場した**。

数値検証: xi=1e-3〜1e-10で、両方の境界とも実測値との比が
1.00000に収束することを確認した(`xor_diagonal_lower_threshold`,
`xor_diagonal_upper_threshold`, `xor_diagonal_threshold_numeric`参照)。

正直な評価: これで対角線a=b上でのXORの挙動は(下側・上側とも)
きれいに閉じた形で説明できた。ただし一般の(a≠b、両方とも
自由に動く)場合の誤分類領域の全体像はまだ描けていない
——これは命題13・14を包含する、より大きな2変数の問題として
残る。

【命題15(v0.29): XNORの断面(b=0)、6ゲート全部の断面解析が完了】
XNOR(a,0;xi)=NOT(XOR(a,0;xi);xi)を、命題13(XORのb=0断面)に
もう一段NOTを重ねた**三重NOT**として解析した。命題8以降で
繰り返し使ってきた「NOT(z;xi)=xi*A(A:=arctanh(1/sqrt(2)))という
小さい目標値を逆算する」という同じ操作を3回入れ子にするだけで、

    y := NOT(reg(a;xi);xi)の目標値 ~= xi*(1/2)*ln(4/(xi*A))
    z := reg(a;xi)の目標値        ~= xi*(1/2)*ln(4/y)
    u = a/xi                      ~= sqrt(z)

という閉形式が出た。xi=1e-2〜1e-12で、実測境界との比が1.000000に
収束することを確認した(`xnor_zero_cross_section_threshold`,
`xnor_zero_cross_section_numeric`参照)。

これで、6ゲート(AND, OR, NAND, NOR, XOR, XNOR)すべてについて、
少なくとも1つの具体的な断面(対角線または b=0)で、誤分類境界の
閉形式を与えられたことになる:

    ゲート  | 断面        | スケーリング則
    AND     | (単一変数)  | O(xi)                          (命題1)
    OR      | u+v         | O(xi*log(1/xi))                (命題8)
    NAND    | a*b(対角線) | O(xi^1.5)                       (命題11)
    NOR     | u+v         | O(xi*log(1/xi))、二重対数補正   (命題12)
    XOR     | b=0         | O(sqrt(xi*log(1/xi)))           (命題13)
    XOR     | 対角線      | 帯構造、上側境界にLambert W函数 (命題14)
    XNOR    | b=0         | 三重入れ子のlog                 (命題15)

正直な評価: これらはすべて特定の断面(対角線かb=0)での結果であり、
各ゲートの(a,b)平面全体の誤分類領域の完全な閉形式は、OR側の
命題8(と、その2次元的な完全性がv0.26で確認された経緯)を除いて、
まだ描けていない。それでも「NOTを重ねるたびに閾値のスケールが
系統的にシフトしていく」という現象を、6ゲート全部について
具体的な数式で確認できたのは、今回のシリーズ(命題5、11〜15)の
一番の到達点だと思う。

【命題16(v0.30): NOT合成塔の統一理論——命題11・13・15を1つに統合】
命題11(NAND)・13(XORのb=0断面)・15(XNORのb=0断面)を並べて
見比べていて、これらが実は**たった1つの一般理論の特殊ケース**
だったことに気づいた。共通しているのは「reg(x;xi)にNOTをn回
繰り返し適用した合成Psi_n(x;xi)が0.5になる閾値」という構造
(n=1がNAND、n=2がXORのb=0断面、n=3がXNORのb=0断面に対応する)。

これを一般のnについて解くと、次の"対数の塔"(log tower)が
閉じた形で現れる:

    T_0(xi)  := xi*A                          (A := arctanh(1/sqrt(2)))
    T_k(xi)  := xi*(1/2)*ln(4/T_{k-1}(xi))    for k=1,...,n-1
    x*_n(xi) := xi*sqrt(T_{n-1}(xi))            (n>=1)

導出は、命題8以降で繰り返し使ってきた「reg(z;xi)=0.5<=>z=xi*A」
という関係と、「NOT(z;xi)=小さい目標値、を逆算する」という
epsilon近似を、nに応じてn回繰り返し適用するだけ。

**この一般公式は、n=1,2,3で命題11・13・15をそのまま(誤差ゼロで)
再現する**——実際、not_tower_threshold(1,xi)、(2,xi)、(3,xi)は
それぞれnand_threshold_ab、xor_zero_cross_section_threshold*xi、
xnor_zero_cross_section_threshold*xiとビット単位で一致する
(同じ計算を一般化しただけなので当然ではあるが、統一できたことに
意味がある)。さらにn=4,5という、名前の付いた標準ゲートには
対応しない"より深いNOTの入れ子"についても、この一般公式が
そのまま正しく機能することを数値検証で確認した——これは新しい
ゲート(将来もっと複雑な論理式を合成したとき)の閾値予測に、
そのまま使える一般理論になっている。

数値検証: n=1,3,5について、xi=1e-2〜1e-6(float64)で実測値との
比が1.000002〜1.01のオーダーで1に収束することを確認した。開発中は
mpmathの50〜60桁精度でxi=1e-40まで検証し、比が1.00000000
(小数点以下8桁まで完全に1)に収束することも確認済み——float64での
検証はxi<1e-6あたりでNOTの繰り返し適用による丸め誤差が蓄積し
精度が落ちる(`not_tower_threshold_numeric`のdocstring参照)。

正直な評価: これは「新しい現象の発見」というより「既に見つけていた
3つの結果(命題11・13・15)が、実は1つの一般理論の特殊ケースに
過ぎなかったと気づき、それを nについて閉じた形で解いた」という
**統合(unification)**である。とはいえ、ユーザーが当初から目指して
いた「統一理論」という言葉にもっとも近い成果になったと思う。
n>=4の場合の解釈(対応する具体的な論理ゲートがあるのか)は
考えていない——あくまで数式としての一般化にとどまる。

【命題10の必要性についての追加調査(v0.34)】命題10(v0.23)の条件
(少なくとも1つの値が閾値を超える)が**必要条件でもあるか**を
調べた。全部の値をこの閾値未満に強制して乱数生成しても、実際には
高い確率(実測で7〜9割程度)でnaive foldとOR_n融合版が一致して
しまう(gapが小さい)ことを確認した——つまり**命題10の条件は
十分条件だが必要条件ではない**ことがはっきりした。

命題9(融合版OR_nの正しさの"総和"条件)のmarginとの相関も調べたが、
「良い一致」グループと「悪い不一致」グループでmarginの分布が
大きく重なっており、単純な閾値では両者をきれいに分離できない
ことも確認した(`or_n_trigger_condition_is_not_necessary`参照)。

正直な評価: 「naive foldとOR_n融合版が一致するための必要十分条件」
は、個々の値の最大値や総和のような単純な集計量だけでは決まらず、
畳み込みの順序や個々の値の並び方に依存する、より複雑な条件である
可能性が高い、ということが今回の調査で明確になった。完全な解決に
は至っていないが、「どこまでが分かっていて、どこからが分かって
いないか」の境界をはっきりさせられたのは収穫。
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
    【命題16(v0.30): NOT合成塔の統一理論——命題11・13・15を包含する
    一般公式】

    命題11(NAND)・13(XORのb=0断面)・15(XNORのb=0断面)は、実は
    すべて「reg(x;xi)にNOTをn回繰り返し適用した合成
    Psi_n(x;xi):=not_composition_tower(x,n,xi)が0.5になる閾値
    x*_n(xi)」という、**たった1つの一般理論の特殊ケース**
    (n=1,2,3)だった。

    Psi_n(x*_n;xi)=0.5 を逆算で解くと、次の"対数の塔"
    (log tower)が現れる:

        T_0(xi)   := xi*A                      (A := arctanh(1/sqrt(2)))
        T_k(xi)   := xi*(1/2)*ln(4/T_{k-1}(xi))   for k=1,...,n-1
        x*_n(xi)  := xi*sqrt(T_{n-1}(xi))           (n>=1の場合)
        x*_0(xi)  := xi*A                           (n=0、reg自体の場合)

    導出: Psi_n(x)=NOT(Psi_{n-1}(x);xi)なので、Psi_n(x)=0.5を
    満たすには、命題8のOR=0.5条件と同型の式 NOT(z;xi)=1-2*reg(z;xi)
    より、Psi_{n-1}(x)=xi*Aが必要(reg(z;xi)=0.5<=>z=xi*Aという、
    このライブラリで繰り返し使ってきた基本の関係そのもの)。
    さらにPsi_{n-2}(x)を求めるには、NOT(Psi_{n-2};xi)=xi*A
    (小さい目標値)を逆算し、reg(Psi_{n-2};xi)=1-xi*A(1に近い)から
    Psi_{n-2}~=xi*(1/2)*ln(4/(xi*A))を得る——これが命題12(NOR)・
    13で使った"epsilon近似"と同じ操作である。以下同様に、小さい
    目標値T_{k-1}からNOT(z;xi)=T_{k-1}を逆算してT_k~=xi*(1/2)*
    ln(4/T_{k-1})を得る、という同じ操作をn-1回繰り返す。最後に
    Psi_0(x)=reg(x;xi)=T_{n-1}(小さい値)をxについて解くと、
    小さい引数の近似tanh(w)~=wにより x*_n~=xi*sqrt(T_{n-1})になる。

    **この一般公式は、n=1,2,3で命題11・13・15をそのまま再現する**
    (n=1: NANDの"a*b"、n=2: XORのb=0断面の"a"、n=3: XNORのb=0
    断面の"a"、として、それぞれ変数の意味づけだけが違う)。さらに
    n=4,5,...という、名前の付いた標準ゲートには対応しない
    "より深いNOTの入れ子"についても、この一般公式がそのまま
    使えることを数値検証で確認した(下記参照)——これは新しい
    ゲートを設計する際の閾値予測にそのまま使える。

    数値検証: n=1,3,5について、xi=1e-2〜1e-40(開発時にmpmath
    50〜60桁精度で検証)まで、実測値との比が1.00000000に収束する
    ことを確認した。本関数はfloat64で実装しているため、xi<1e-6
    あたりでnot_composition_towerのfloat64精度限界(NOTの繰り返しに
    よる丸め誤差の蓄積)に達することがある——大きなnやxi<<1での
    高精度な検証にはmpmathなどの多倍長精度演算を推奨する
    (`not_tower_threshold_numeric`のdocstring参照)。
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
