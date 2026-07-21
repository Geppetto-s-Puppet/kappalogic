"""
kappalogic.quantum_logic
==========================
v0.77: soft logic を量子測定論に接続する——ξ を「測定強度」として読む。

kappalogic の "柔らかい射影" soft_projector(H;ξ) = (I + k(H;ξ))/2
(matrix_backend、Eigenlogic の射影 Π に対応)は、実は**有効な2値POVM効果**
である: その固有値は (1 + tanh(λ/ξ))/2 ∈ (0,1) で 0<=E<=I を満たし、
E+ = soft_projector, E- = I - E+ は E+ + E- = I(完全性)を満たす。

したがって {E+, E-} は「命題 H>0(その状態は"真"か)」を測る2値測定であり、
ξ がその**鋭さ(測定強度)**を連続的に制御する:

    ξ -> 0 : E+ -> H の正固有空間への射影 = 射影(強)測定。tanh -> 符号。
    ξ 大   : E+ -> I/2 = 情報を得ない自明(最弱)測定。tanh -> 0。

これは kappalogic の中心テーマ「ξ->0 で滑らかな緩和が厳密な離散論理へ」の
量子版であり、ξ の意味論(鋭さ・温度・拡散時間・RGスケール・Fisher-Rao 尺度)
に「**弱測定の強度**」を加える。測定強度は kappalogic の検出器 reg そのもので
書ける: S(H;ξ) = Tr(reg(H;ξ))/n = reg 検出器の平均応答。

正直な位置づけ:
- POVM としての妥当性・射影極限・強度の単調性は厳密(下記検証)。
- ここでの「量子論理」は、Eigenlogic のゲート演算子を状態の期待値
  <psi|F|psi> で評価する(量子真理値)という標準的な埋め込み。ゲート演算子は
  テンソル因子ごとの可換な射影から作られるので、**基底状態では厳密な真理値表、
  積状態では周辺確率の積**になる。エンタングル状態(例: Bell 状態)では真理値が
  相関するが、その相関は**古典的な相関源でも再現でき**(可換な射影の同時測定の
  統計は古典確率で書ける)、それ自体は非古典性の証拠ではない——本モジュールは
  そこを誇張しない。真に非古典な現象(文脈依存性・CHSH 破れ)は非可換な命題の
  同時測定を要し、本モジュールの範囲外。
- soft_projector 自体は matrix_backend の Eigenlogic 実装(v0.44)が初出。本
  モジュールはそれを「弱測定 POVM」として読み直し、測定強度・Kraus 作用素・
  量子真理値を足したもの。
"""
import numpy as np

from .matrix_backend import soft_projector, matrix_reg


def soft_povm(H, xi, n_doublings=25):
    """
    命題 "H>0"(状態は"真"か)を測る2値POVM {E+, E-} を返す。

        E+ = soft_projector(H;ξ) = (I + k(H;ξ))/2
        E- = I - E+             = (I - k(H;ξ))/2

    各 E は 0<=E<=I(有効な効果)、E+ + E- = I(完全)。ξ->0 で射影測定、
    ξ 大で自明測定(I/2)に連続変化する。戻り値: (E_plus, E_minus)。
    """
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    Ep = soft_projector(H, xi, n_doublings)
    return Ep, np.eye(n) - Ep


def is_valid_povm(effects, tol=1e-9):
    """
    効果の並び {E_k} が有効なPOVMか判定する: 各 E_k がエルミートで
    0<=E_k<=I(固有値が[−tol,1+tol])、かつ Σ_k E_k = I(誤差 tol 以内)。
    """
    effects = [np.asarray(E, dtype=float) for E in effects]
    n = effects[0].shape[0]
    total = np.zeros((n, n))
    for E in effects:
        if np.linalg.norm(E - E.T) > tol:
            return False
        w = np.linalg.eigvalsh(E)
        if w.min() < -tol or w.max() > 1 + tol:
            return False
        total = total + E
    return bool(np.linalg.norm(total - np.eye(n)) < tol)


def measurement_strength(H, xi, n_doublings=25):
    """
    2値POVM {E+, E-}(soft_povm)の**測定強度** S(H;ξ) ∈ [0,1]。

        S(H;ξ) = Tr(reg(H;ξ)) / n = <tanh²(λ_i/ξ)> の固有値平均

    reg=k²=tanh² は kappalogic の"非ゼロ検出器"そのもの——測定強度は
    reg 検出器の平均応答に一致する。各効果 E=(1+k)/2 の"曖昧さ"(Busch の
    unsharpness)は E(I-E)=(I-k²)/4=sech²/4 で、S=1-4·<E(I-E)> = <k²>。

    ξ->0 で S->1(射影測定、全固有値で reg->1、ただし固有値0の閾値上を除く)、
    ξ->∞ で S->0(自明測定、E->I/2)。単調な弱→強測定の補間。
    """
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    return float(np.trace(matrix_reg(H, xi, n_doublings)) / n)


def soft_measurement_kraus(H, xi, n_doublings=25):
    """
    soft 測定 {E+, E-} の Kraus 作用素 {M+, M-} = {sqrt(E+), sqrt(E-)}
    (E±>=0 の主平方根、エルミート)。M+†M+ + M-†M- = E+ + E- = I なので
    有効な量子操作。測定後状態は ρ_± = M_± ρ M_±† / Tr(M_± ρ M_±†)。
    弱測定(ξ 大)では M_± ~ I/√2 で状態をほとんど乱さないが情報も得ない。
    戻り値: (M_plus, M_minus)。
    """
    Ep, Em = soft_povm(H, xi, n_doublings)

    def sqrtm_sym(E):
        w, V = np.linalg.eigh(E)
        w = np.clip(w, 0.0, None)
        return (V * np.sqrt(w)) @ V.T

    return sqrtm_sym(Ep), sqrtm_sym(Em)


def quantum_truth_value(F, psi):
    """
    論理ゲート演算子 F(Eigenlogic の eigenlogic_AND/OR 等、テンソル積空間上の
    エルミート"真理値"作用素)の、状態 |psi> における**量子真理値** <psi|F|psi>。

    - 計算基底の固有状態では厳密な真理値表(0/1)。
    - 積状態 |a>⊗|b> では周辺確率の積(独立入力)。
    - エンタングル状態では相関した真理値(ただし可換射影の同時測定ゆえ
      古典相関でも再現可能——非古典性の主張はしない、モジュール docstring 参照)。

    psi は自動的に正規化する。戻り値は実スカラー(F はエルミートを仮定)。
    """
    F = np.asarray(F)
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return float(np.real(psi.conj() @ F @ psi))
