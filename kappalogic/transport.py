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
