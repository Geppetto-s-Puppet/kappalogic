"""
kappalogic.matrix_backend
============================
v0.43〜v0.45: 行列(非可換)版kappalogicへの拡張。

- Eigenlogic (Vourdas & Dubois, 2020, *Entropy*): 射影演算子から
  AND/OR/XOR等をCayley-Hamilton定理とLagrange補間で演算子として
  合成する研究。原論文を実際に読み、多入力の論理演算が「入力ごとに
  別々のヒルベルト空間を割り当て、その**テンソル積**上でKronecker積
  により演算子を組み合わせる」構成であることを確認した(式(4)(6):
  F_AND = Pi_A (x) Pi_B、Piは真理値{0,1}の射影演算子)。
- 非加法性演算子(Vourdas, 2015): 射影演算子の交換子と結びつく量
  (まだ論文未確認、今後の課題)。
- 行列版フェルミ占有 rho=1-tanh(beta/2*(H-mu)) (1994年以来の技術、
  Fermi Operator Expansion): 対角化せずに行列の掛け算だけで密度
  行列を再帰展開する。この再帰展開は、identities.pyの n_tuple_angle
  (倍角公式 kappa(2x)=2*kappa(x)/(1+kappa(x)^2))を行列に適用した
  ものと数学的に同じ。

**Eigenlogic論文を読んで気づいた重要な点**: Eigenlogicの射影演算子
Pi(固有値{0,1}、"真/偽"の射影)は、kappalogicのreg(=k^2、偶関数で
符号を区別しない)ではなく、**k自体(符号を保つ)から作った"柔らかい
射影"soft_projector(H;xi)=(I+k(H;xi))/2に対応する**。これは
core.pyのAND定義「a,bが共に非0のとき1」(符号を区別しない"非ゼロ
検出")と、標準的なBoolean論理「a,bが共に真のとき1」(符号で真偽を
区別する)が、kappalogic内部でそもそも別の概念だったことに気づく
契機になった。

このため、非可換版のAND/ORは2種類実装している:
- `tensor_AND`/`tensor_OR`: kappalogic本体のAND/OR(符号を区別
  しない"共に非ゼロ"判定)をそのまま行列に一般化。1x1行列の場合に
  本体の関数と機械精度で厳密一致。
- `eigenlogic_AND`/`eigenlogic_OR`: Eigenlogic論文の構成
  (F_AND=Pi_A(x)Pi_B)を、soft_projectorで一般化したもの。
  符号を区別する正真正銘のBoolean論理になっており、xi->0の極限で
  通常の真理値表と厳密に一致する。
- `matrix_AND`(v0.43、旧版): Jordan積(H1H2+H2H1)/2による素朴な
  拡張。対称性の保証だけが目的の場当たり的な構成で、tensor_AND/
  eigenlogic_ANDより原理的な根拠に乏しい——後方互換のため残す。

いずれのテンソル積ベースの構成(tensor_*、eigenlogic_*)も、H1,H2が
非可換でも常にHermitianになる(Jordan積のような場当たり的な補正が
不要——テンソル積という構造だけで非可換性の問題が自動的に解消される)。

**本モジュールはVourdasの非加法性演算子(2015)を正確に再現したもの
ではない**(原論文はまだ読み込めていない)。Eigenlogic(2020)は原論文
を読んで構成を確認したが、多値論理・エンタングルメント等の発展的な
話題までは追えていない。

すべてkernel="tanh"限定、対称(Hermitian実対称)行列を前提にする。
"""
import numpy as np


def matrix_k(H, xi, n_doublings=25):
    """
    行列版のk(H)=tanh(H/xi)を、対角化せずに再帰的な倍角公式で計算する
    (Fermi Operator Expansionと同じ発想)。

    identities.pyのn_tuple_angle(倍角公式
    kappa(2x)=2*kappa(x)/(1+kappa(x)^2))を、スカラーではなく行列
    (行列の積・逆行列)に読み替えるだけで、対角化なしにtanh(H/xi)が
    計算できる。H/(xi*2^n_doublings)が十分小さい(kappa(x)~=xとみなせる)
    ところから出発し、倍角公式をn_doublings回繰り返す。

    検証: 対称行列H(5x5〜8x8)で、scipy.linalg.eighによる直接対角化
    (kappa(H)=V diag(tanh(w/xi)) V^T)と比較し、n_doublings=25で
    誤差~1e-15(機械精度)で一致することを確認済み。
    """
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    X = H / (xi * 2 ** n_doublings)
    I = np.eye(n)
    for _ in range(n_doublings):
        X = 2 * X @ np.linalg.inv(I + X @ X)
    return X


def matrix_reg(H, xi, n_doublings=25):
    """行列版のreg(H)=tanh(H/xi)^2 = matrix_k(H,xi) @ matrix_k(H,xi)。"""
    k = matrix_k(H, xi, n_doublings)
    return k @ k


def matrix_fermi_occupation(H, mu, kT, n_doublings=25):
    """
    行列版フェルミ・ディラック占有数(密度行列):

        rho = (I + matrix_k(mu*I - H, 2*kT)) / 2

    electronic_structure.pyのスカラー版fermi_occupation(eps,mu,kT)
    =(1+sgn(mu-eps,2*kT))/2 を、対角化せずに行列へ拡張したもの。

    検証: 1x1行列でスカラー版のfermi_occupationと機械精度で一致。
    ランダムな対称行列(5x5〜8x8)で、scipy.linalg.eighによる直接対角化
    (rho = V diag(1/(1+exp((w-mu)/kT))) V^T)と比較し、誤差~1e-15
    (機械精度)で一致。密度行列の固有値が[0,1]に収まること(占有数
    としての物理的妥当性)、トレース(電子数)が対角化版の
    sum(1/(1+exp((w-mu)/kT)))と一致することも確認済み。
    """
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    xi = 2 * kT
    return (np.eye(n) + matrix_k(mu * np.eye(n) - H, xi, n_doublings)) / 2


def commutator(H1, H2):
    """[H1,H2] = H1@H2 - H2@H1。非可換性の直接の指標。"""
    return H1 @ H2 - H2 @ H1


def commutator_norm(H1, H2):
    """||[H1,H2]||_F (Frobenius ノルム)。H1,H2が可換なら厳密に0。"""
    return float(np.linalg.norm(commutator(H1, H2), "fro"))


def matrix_AND(H1, H2, xi, n_doublings=25):
    """
    非可換版AND(H1,H2;xi)、v0.43の最初の試み(Jordan積)。

    AND(a,b;xi)=reg(a*b;xi)を素朴に行列へ拡張するとreg(H1@H2;xi)に
    なるが、H1,H2が対称でもH1@H2は対称とは限らない(非可換なら)。
    ここではJordan積(反交換子)(H1@H2+H2@H1)/2を使うことで、常に
    対称な行列を保証する(H1,H2が可換ならH1@H2そのものと一致する
    ——commutator_norm(H1,H2)=0の場合に確認済み)。

    **v0.44追記**: この構成は同じヒルベルト空間上でH1,H2を直接
    掛け合わせるため、非可換性の問題が残る(Jordan積は対称性を
    保証するだけの場当たり的な補正)。より原理的な構成は
    `tensor_AND`/`tensor_OR`(Eigenlogic流のテンソル積)を参照。
    """
    jordan_product = (H1 @ H2 + H2 @ H1) / 2
    return matrix_reg(jordan_product, xi, n_doublings)


def matrix_NOT(H, xi, n_doublings=25):
    """行列版のNOT(H;xi) = I - matrix_reg(H,xi)。"""
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    return np.eye(n) - matrix_reg(H, xi, n_doublings)


def soft_projector(H, xi, n_doublings=25):
    """
    【v0.44追記: Eigenlogicの"種"演算子との対応】

    Eigenlogic (Vourdas & Dubois, 2020)の射影演算子Pi(固有値{0,1}、
    「真」の固有空間への射影)に対応する、符号に敏感な"柔らかい射影"。

        Pi_soft(H;xi) := (I + matrix_k(H;xi)) / 2

    **重要な注意**: `matrix_reg(H,xi)=matrix_k(H,xi)^2`は偶関数
    (tanh^2)なので符号を区別できない(kappalogic本体のAND(a,b)=
    reg(a*b;xi)は「a,bが共に非ゼロか」を判定するもので、「a,bが
    共に正(真)か」を判定するものではない——core.pyのAND docstring
    「a,bが共に非0のとき1」を参照)。Eigenlogicの射影演算子Piは
    「真/偽」という符号の情報を持つ射影なので、対応する"柔らかい"版は
    reg(=k^2、符号を消す)ではなく、k自体(符号を保つ)から作る必要が
    ある。fermi_occupationが(1+sgn(...))/2という同じ形を既に
    使っている。

    xi->0でPi_soft(H;xi)は真の射影演算子(固有値が厳密に0か1)になり、
    Hの正の固有空間への射影と一致することを確認済み。
    """
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    return (np.eye(n) + matrix_k(H, xi, n_doublings)) / 2


def eigenlogic_AND(H1, H2, xi, n_doublings=25):
    """
    Eigenlogic流のAND: F_AND = Pi_A (x) Pi_B (Eigenlogic論文の式(4)(6)、
    2入力の"atomic proposition"をKronecker積で独立に拡張する構成)を、
    soft_projector(符号に敏感な柔らかい射影)で一般化したもの。

        eigenlogic_AND(H1,H2;xi) := kron(soft_projector(H1,xi), soft_projector(H2,xi))

    検証: xi->0の極限で、真理値表(true,true)->1, (true,false)->0,
    (false,false)->0 という、通常のBoolean ANDの真理値表と厳密に
    一致することを確認した(true/falseはHの固有値の符号で表す)。

    **tensor_ANDとの違い**: tensor_AND(kappalogic本体のAND(a,b)=
    reg(a*b;xi)の拡張)は符号を区別しない「共に非ゼロか」の判定。
    eigenlogic_AND(Eigenlogic論文の構成をそのまま真似たもの)は
    符号を区別する、正真正銘のBoolean AND。目的に応じて使い分ける。
    """
    return np.kron(soft_projector(H1, xi, n_doublings), soft_projector(H2, xi, n_doublings))


def eigenlogic_OR(H1, H2, xi, n_doublings=25):
    """
    Eigenlogic流のOR。eigenlogic_ANDと対になる、符号に敏感な
    (正真正銘のBoolean)OR:

        eigenlogic_OR(H1,H2;xi) := I - kron(I-soft_projector(H1,xi), I-soft_projector(H2,xi))

    検証: xi->0の極限で(true,true)->1, (true,false)->1,
    (false,false)->0 という通常のBoolean ORの真理値表と厳密に一致。
    """
    n1, n2 = H1.shape[0], H2.shape[0]
    not1 = np.eye(n1) - soft_projector(H1, xi, n_doublings)
    not2 = np.eye(n2) - soft_projector(H2, xi, n_doublings)
    return np.eye(n1 * n2) - np.kron(not1, not2)


def tensor_AND(H1, H2, xi, n_doublings=25):
    """
    【v0.44】Eigenlogic流の"テンソル積"によるAND(H1,H2;xi)。

    Eigenlogicは多入力の論理演算を、入力ごとに別々のヒルベルト空間を
    割り当て、その**テンソル積**の上で射影演算子を組み合わせて構成する
    ——これなら、そもそも「異なる入力の演算子が可換かどうか」という
    問題が起きない(各入力は自分の空間にしか作用しないので、
    テンソル積に拡張した演算子同士は自動的に可換になる)。

    kappalogicのAND(a,b;xi)=reg(a*b;xi)は、a,bの積をreg一回だけに
    通す構造なので、この発想をそのまま真似ると:

        tensor_AND(H1,H2;xi) := matrix_reg(kron(H1,H2), xi)

    (Kronecker積kron(H1,H2)の固有値は、H1の固有値とH2の固有値の
    "全部の組み合わせの積"になる——スカラーのa*bを、複数の
    固有値の組み合わせに一般化したもの)。

    検証: 1x1行列(スカラー)の場合、kappalogic本体のAND(a,b;xi)と
    機械精度で一致することを確認した(v0.43のJordan積版は一致
    **しなかった**——reg(a)*reg(b)とreg(a*b)は別物であるため)。
    H1,H2が非可換でも、kron()の構造だけからtensor_ANDは常に対称
    (Hermitian)になる(Jordan積のような場当たり的な補正が不要)。

    戻り値の次元はdim(H1)*dim(H2)(テンソル積空間)になることに注意
    ——同じ次元空間で閉じるmatrix_AND(Jordan積版)とは違う設計。

    **符号について**: kappalogicのreg=k^2は偶関数なので、tensor_ANDも
    符号を区別しない(「共に非ゼロか」の判定、core.pyのAND docstring
    通り)。符号を区別する正真正銘のBoolean ANDが欲しい場合は
    `eigenlogic_AND`(soft_projectorを使う版)を使う。
    """
    return matrix_reg(np.kron(H1, H2), xi, n_doublings)


def tensor_OR(H1, H2, xi, n_doublings=25):
    """
    【v0.44】Eigenlogic流の"テンソル積"によるOR(H1,H2;xi)。

    kappalogicのOR(a,b;xi)=NOT(NOT(a;xi)*NOT(b;xi);xi)(命題4以降で
    繰り返し使ってきた"二重NOT"の構造)を、そのままテンソル積に
    読み替える:

        tensor_OR(H1,H2;xi) := I - matrix_reg(kron(NOT(H1;xi), NOT(H2;xi)), xi)

    tensor_ANDと同様、1x1行列(スカラー)の場合にkappalogic本体の
    OR(a,b;xi)と機械精度で一致することを確認した。H1,H2が非可換でも
    常に対称(Hermitian)になる。
    """
    n1, n2 = H1.shape[0], H2.shape[0]
    P = np.kron(matrix_NOT(H1, xi, n_doublings), matrix_NOT(H2, xi, n_doublings))
    return np.eye(n1 * n2) - matrix_reg(P, xi, n_doublings)


def matrix_susceptibility(H, A, xi, gate_fn=None, h=1e-6, n_doublings=25):
    """
    v0.46: TODO.md「行列版の摂動論(自動微分に相当)」への対応。

    Niklasson et al. (2024, "Susceptibility Formulation of Density
    Matrix Perturbation Theory")のEq.(13): 対称行列X、対称な"観測量"A
    について、

        d(Tr[A * f(X)]) / dX = d(f(X + lambda*A)) / dlambda |_{lambda=0}

    という恒等式が任意の(多項式で書ける)行列関数f(X)について成り立つ
    (両辺とも対称行列)。つまり、Tr[A*f(H)]のHに関する**全微分行列**
    (n×n個の偏微分)が、**たった1方向の摂動**(H+lambda*Aとlambda
    について微分するだけ)で計算できる——各成分ごとに数値微分する
    (n^2回の評価が要る)のに比べてO(n^2)倍速い。

    kappalogic自身のmatrix_k(FOEの再帰倍角展開)にこの恒等式を
    直接適用し、中心差分でlambda微分を計算する。

    検証: f=matrix_kとして、成分ごとの数値微分(n^2回評価)と、
    この方向微分トリック(2回評価)が、4x4行列で誤差~1e-9で一致
    することを確認した。

    引数:
        H: 対称行列(基準点)
        A: 対称行列("観測量"、方向微分の向き)
        xi: スケールパラメータ
        gate_fn: 使う行列関数(デフォルトはmatrix_k)。matrix_regや
            matrix_fermi_occupationのような、Hのみを引数に取る形に
            部分適用した関数も渡せる。
        h: 中心差分のステップ幅

    戻り値: d(Tr[A*f(H)])/dH に等しい対称行列(f(H+h*A)とf(H-h*A)の
    中心差分)。
    """
    if gate_fn is None:
        gate_fn = lambda X: matrix_k(X, xi, n_doublings)
    return (gate_fn(H + h * A) - gate_fn(H - h * A)) / (2 * h)
