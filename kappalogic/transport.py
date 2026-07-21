"""
kappalogic.transport
======================
v0.79: 電子輸送(電気/熱伝導・電子比熱)を kappalogic の検出器で計算する。

核心の観察: フェルミ・ディラック占有 f(E)=1/(1+e^{(E-μ)/kT}) のエネルギー微分
(=電流・熱を運ぶ"輸送窓")は、**kappalogic の NOT 検出器そのもの**である:

    w(E) := -∂f/∂E = sech²((E-μ)/2kT) / (4kT) = NOT(E-μ; 2kT) / (4kT)

(NOT(x;ξ)=1-reg(x;ξ)=sech²(x/ξ)、electronic_structure.fermi_occupation と
地続き。ξ=2kT の熱カーネル。)つまり「フェルミ準位まわり幅 kT の輸送窓」=
kappalogic の "OFF 検出器" NOT。窓は規格化されており ∫ w dE = 1。

この窓のモーメント L_n = ∫ (E-μ)^n w(E) dE から、輸送の普遍則が出る:
    L_0 = 1,  L_1 = 0,  L_2 = (π²/3)(kT)²  (数値・解析で確認)
- 電気伝導度 σ ∝ L_0、電子熱伝導度 κ ∝ L_2/T ⇒
  **Wiedemann-Franz 則 κ/(σT) = (π²/3)(k_B/e)² = 2.44e-8 WΩ/K²(ローレンツ数)**
  ——輸送の緩和時間や状態密度に依らない普遍定数を再現。
- 電子比熱(ゾンマーフェルト)C_v = (π²/3) k_B² g(μ) T(T に線形、係数 γ)。

外部から借用する理論はボルツマン/ゾンマーフェルト輸送のみ。これは教科書物性の
再現であって新物理ではないが、「輸送窓 = kappalogic の NOT」という接続で、
物性(比熱・伝導・Wiedemann-Franz)を kappalogic の延長で計算できることを示す。
"""
import numpy as np
from scipy.integrate import quad

from .core import NOT

# ローレンツ数 L0 = π²/3 (k_B/e)^2 (Wiedemann-Franz 則の普遍定数)
_KB = 1.380649e-23        # J/K
_E = 1.602176634e-19      # C
LORENZ_NUMBER = (np.pi**2 / 3.0) * (_KB / _E)**2   # ≈ 2.44e-8 W·Ω·K^-2


def thermal_transport_window(E, mu, kT):
    """
    電子輸送窓 w(E) = -∂f/∂E = NOT(E-μ; 2kT)/(4kT)(kappalogic の NOT 検出器)。
    フェルミ準位 μ まわり幅 ~kT に局在し、∫ w dE = 1 に規格化される。電流・熱を
    運ぶのはこの窓の中の電子だけ、という輸送の描像を kappalogic の言葉で表す。
    """
    return NOT(np.asarray(E, dtype=float) - mu, 2.0 * kT) / (4.0 * kT)


def transport_moment(n, mu, kT, n_kT=60):
    """
    輸送窓のモーメント L_n = ∫ (E-μ)^n w(E) dE(±n_kT·kT を数値積分)。
    解析値: L_0=1, L_1=0(対称), L_2=(π²/3)(kT)², 奇数 n は 0。
    """
    val, _ = quad(lambda E: (E - mu)**n * thermal_transport_window(E, mu, kT),
                  mu - n_kT * kT, mu + n_kT * kT, limit=300)
    return val


def lorenz_ratio(kT):
    """
    輸送窓から計算したローレンツ比 L_2 / (L_0 · (kT)²)。弱い T 依存を除き
    π²/3 = 3.2899 に収束する(Wiedemann-Franz)。単位 (k_B/e)² を掛けると
    SI のローレンツ数 LORENZ_NUMBER になる。
    """
    L0 = transport_moment(0, 0.0, kT)
    L2 = transport_moment(2, 0.0, kT)
    return L2 / (L0 * kT**2)


def sommerfeld_heat_capacity_coefficient(dos_at_mu):
    """
    電子比熱のゾンマーフェルト係数 γ = (π²/3) k_B² g(μ)(C_v = γ T、T に線形)。
    g(μ) はフェルミ準位での状態密度。C_v が輸送窓の2次モーメント L_2 由来
    ((π²/3)(kT)²)であることの帰結。単位は dos_at_mu の単位に従う(k_B=1 の
    自然単位なら γ=(π²/3) g)。
    """
    return (np.pi**2 / 3.0) * dos_at_mu   # k_B=1 自然単位; ×k_B² で SI


def electronic_heat_capacity(dos_at_mu, T):
    """電子比熱 C_v = γ T = (π²/3) k_B² g(μ) T(k_B=1 自然単位)。"""
    return sommerfeld_heat_capacity_coefficient(dos_at_mu) * T


def kubo_dc_conductivity(hamiltonian, mu=0.0, kT=0.05, eta=0.15, positions=None):
    """
    【v0.86: 乱れ×輸送】久保-Greenwood の DC 伝導度を、**kappalogic の検出器を
    2つとも使って**実空間ハミルトニアンから直接計算する(並進対称性は不要なので
    乱れた系にそのまま使える——disorder.disordered_chain を食わせられる)。

        σ_DC ∝ (1/N) Σ_{i≠j} |v_ij|² · δ_η(E_i - E_j) · w(E_i)
                                ───────      ────────      ────
                          速度行列要素   デルタ検出器   輸送窓(NOT)

    - 速度演算子は v = i[H, x̂] なので ⟨i|v|j⟩ = i(E_i-E_j)⟨i|x̂|j⟩、
      よって |v_ij|² = (E_i-E_j)²|x_ij|²。
    - エネルギー保存(準位のほぼ縮退)は funcs.delta_approx(= NOT の規格化版)。
    - フェルミ面の重みは thermal_transport_window(= NOT/(4kT))。
    つまり optics.eps2_from_hamiltonian と**同じ2つの検出器**で、光学(縦の遷移)
    と輸送(横の伝導)が書ける。η は準位の広がり(ξ の意味論のひとつ)。

    == 内的アンカー: 独立2ルートで測った局在長が一致する ==
    1次元の乱れた鎖(N=200, kT=0.05, η=0.15, μ=0 バンド中心、8配位平均)で、
    本関数の σ_DC と、波動関数から測った局在長 ζ=1/IPR
    (disorder.inverse_participation_ratio)を並べると:

        W    :   0.0     0.5     1.0     2.0     4.0     8.0
        σ_DC : 1.8235  1.5519  1.1776  0.4372  0.0616  0.0044
        ζ    : 134.00   99.83   69.14   27.38    8.84    3.27
        σ/ζ  : 0.0136  0.0155  0.0170  0.0160  0.0070  0.0014

    **弱〜中程度の乱れ(W≲2)で σ/ζ がほぼ一定** ——すなわち σ_DC ∝ ζ。
    輸送(久保公式)と波動関数の広がり(IPR)という**まったく独立な2つの測り方**が
    同じ局在長を見ている、という一致。σ 自体は乱れで3桁落ちる(1.82→0.0044)。

    正直な限界: 比例 σ∝ζ は強乱れ(W≳4、ζ が数サイト)で崩れる(σ/ζ が
    0.007→0.0014 と落ちる)。拡散的な輸送から強局在(ホッピング)へ移るためで、
    物理的に正しい振る舞い。また σ の絶対値は任意単位(e²/ħ 等の前因子は付けて
    いない)——意味があるのは乱れ依存性と ζ との比例関係の方。η を極端に小さく
    すると有限系の離散準位が拾えなくなるので、準位間隔程度に取ること。

    **3D Anderson 転移のプローブとしては使えない(v0.87 の否定的知見)**:
    厳密対角化できる範囲(L<=10、N<=1000)では、本関数の σ は転移の有限サイズ
    signature を出さない。η を固定すると L 増で準位間隔が詰まり η 内の準位対が
    増えるだけの人工効果が支配し(W によらず σ が L とともに増える)、逆に
    η を準位間隔に比例させると η 内の準位対が激減して統計ノイズに埋もれる
    (最も局在した W=30 が「金属的」と出るなど物理的に誤った結果になった)。
    3D 転移を見るには転送行列の Λ(命題39、disorder.transfer_matrix_lambda)か
    IPR×N(disorder.ipr_times_n_3d)を使うこと。σ が有効なのは、本 docstring
    冒頭で示した**1次元での乱れ依存性と ζ との比例**のような用途。

    positions: 各サイトの座標(電流の向きに対応する成分)。既定 None なら
    1次元鎖とみなして arange(N) を使う。**3D 等では x 座標を明示的に渡すこと**
    (格子の平坦インデックスをそのまま座標に使うのは誤り)。
    """
    H = np.asarray(hamiltonian, dtype=float)
    N = H.shape[0]
    E, V = np.linalg.eigh(H)
    pos = np.arange(N, dtype=float) if positions is None \
        else np.asarray(positions, dtype=float)
    X = (V.T * pos) @ V                              # <i|x|j>
    dE = E[None, :] - E[:, None]
    v2 = (dE ** 2) * np.abs(X) ** 2                  # |v_ij|^2
    np.fill_diagonal(v2, 0.0)
    w = thermal_transport_window(E, mu, kT)
    from .funcs import delta_approx
    return float(np.sum(v2 * delta_approx(dE, eta) * w[:, None]) / N)


def wiedemann_franz_check(kT, g=1.0, vF=1.0, tau=1.0):
    """
    単純な Drude/Sommerfeld 模型(状態密度 g・フェルミ速度 vF・緩和時間 tau)で
    電気伝導度 σ と電子熱伝導度 κ を輸送窓のモーメントから計算し、
    Wiedemann-Franz 比 κ/(σ·T) を返す。緩和時間・DOS・速度は比で相殺し、
    κ/(σT) = L_2/(L_0 (kT)²) = π²/3(自然単位、k_B=e=1)に収束する。

    戻り値: dict(sigma, kappa, ratio=κ/(σT), pi2_over_3=π²/3)。
    """
    T = kT   # k_B=1
    D = g * vF**2 * tau / 3.0            # 共通の輸送係数
    L0 = transport_moment(0, 0.0, kT)
    L2 = transport_moment(2, 0.0, kT)
    sigma = D * L0                       # e=1
    kappa = D * L2 / T
    return {
        "sigma": sigma,
        "kappa": kappa,
        "ratio": kappa / (sigma * T),
        "pi2_over_3": np.pi**2 / 3.0,
    }
