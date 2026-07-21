"""
kappalogic.optics
===================
v0.83: 光学応答と「色」を kappalogic の検出器から組み立てる。

核心: 誘電関数の虚部(吸収)は、**2つの kappalogic 検出器の積**で書ける:

    ε₂(ω) ∝ (1/ω²) Σ_k |M_k|² · δ_ξ(ΔE_k - ω) · [f_v - f_c]
              ─────────────────   ────────────    ─────────
              遷移の重み           エネルギー保存    占有差
                                  = delta_approx    = fermi_occupation
                                  = NOT(x;ξ)/(2ξ)   = tanh ベース

つまり「エネルギー保存のデルタ」は funcs.delta_approx(=NOT 検出器の規格化版、
transport.py の輸送窓と同じ NOT の別スケール)、「占有差」は tanh のフェルミ分布。
光学応答まるごとが kappalogic の部品で書ける。

**ξ に5つ目の意味論**: ここでの ξ は**スペクトル線幅**(励起の有限寿命による
自然幅)である。既存の意味論(鋭さ・温度・拡散時間・RGスケール・Fisher-Rao 尺度・
弱測定の強度)に「線幅(1/寿命)」が加わる。

**内的な自己検証(アンカー)**: kappalogic の delta_approx は ∫δ_ξ dx = 1 に
**厳密に**規格化されているので、総和則
    ∫ ε₂ dω  と  ∫ ω ε₂ dω
は **ξ(線幅)に依存しない**。実測: SSH 鎖(t1=1.0,t2=0.6)で ξ=0.02/0.05/0.1 の
いずれでも ∫ε₂dω=1.000000、∫ωε₂dω=2.184477 と完全一致(ξ=0.2 では積分窓から
裾が漏れて 0.999925 とわずかにずれる)。線幅を変えてもスペクトル重みが保存する、
という物理的に正しい性質が、検出器の規格化から自動的に従う。

そして応用として、吸収スペクトルから**実際の色(sRGB)**を計算する:
可視域の透過率 T(λ)=exp(-α(λ)d) を CIE 等色関数(解析フィット、外部データ不要)で
積分して XYZ→sRGB に変換する。バンドギャップから物質の色が出る。

借用した外部理論: 直接遷移の吸収の標準形、CIE 1931 等色関数の解析近似
(Wyman-Sloan-Shirley 型の多ローブ Gauss フィット)、sRGB 変換行列。
これらは教科書・標準規格であり、kappalogic 由来ではない(と明示する)。
"""
import numpy as np

from .funcs import delta_approx

# 光子エネルギー(eV) と波長(nm): E = HC_EV_NM / lambda
HC_EV_NM = 1239.841984


def ssh_transition_energies(t1, t2, nk=800):
    """
    交替ホッピング(SSH)1D 鎖の直接遷移エネルギー ΔE(k)=2|t1+t2 e^{ik}|。
    バンドギャップは 2|t1-t2|、最大遷移エネルギーは 2(t1+t2)。
    戻り値: ΔE の配列(k を等間隔に取ったもの)。
    """
    k = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    return 2.0 * np.abs(t1 + t2 * np.exp(1j * k))


def dielectric_imaginary(omega, transition_energies, xi, weights=None):
    """
    ε₂(ω) を kappalogic のデルタ検出器で組む:

        ε₂(ω) = < w_k · δ_ξ(ΔE_k - ω) >_k        （δ_ξ = delta_approx）

    (占有差は T=0 の直接遷移を想定して 1、有限温度で使う場合は weights に
     [f_v - f_c] を掛けて渡す。1/ω² のような前因子も weights で調整できる。)

    omega: スカラーまたは配列。transition_energies: ΔE_k の配列。
    xi: 線幅(スペクトル広がり)。weights: 各遷移の重み |M_k|²(既定は 1)。
    """
    dE = np.asarray(transition_energies, dtype=float)
    w = np.ones_like(dE) if weights is None else np.asarray(weights, dtype=float)
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.array([np.mean(w * delta_approx(dE - o, xi)) for o in omega_arr])
    return out if np.ndim(omega) else float(out[0])


def spectral_weight(omega_grid, eps2, moment=0):
    """
    総和則 ∫ ω^moment ε₂(ω) dω(台形則)。delta_approx が厳密に規格化されて
    いるため、これは**線幅 ξ に依存しない**(本モジュールの内的アンカー)。
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    return float(np.trapezoid(omega_grid**moment * np.asarray(eps2, dtype=float),
                              omega_grid))


def absorbance(omega, eps2, prefactor=1.0):
    """
    吸収係数 α(ω) ∝ ω·ε₂(ω)(直接遷移の標準形、借用)。prefactor で
    材料の厚み・単位をまとめて調整する。
    """
    return prefactor * np.asarray(omega, dtype=float) * np.asarray(eps2, dtype=float)


# ---------- CIE 1931 等色関数(解析フィット、外部データ不要) ----------
def _piecewise_gauss(x, alpha, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return alpha * np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_color_matching(lam_nm):
    """
    CIE 1931 の等色関数 x̄,ȳ,z̄ の多ローブ Gauss 解析近似(Wyman-Sloan-Shirley 型)。
    外部データファイルを使わずに可視域の色計算をするための**借用した標準近似**。
    """
    lam = np.asarray(lam_nm, dtype=float)
    x = (_piecewise_gauss(lam, 1.056, 599.8, 37.9, 31.0)
         + _piecewise_gauss(lam, 0.362, 442.0, 16.0, 26.7)
         + _piecewise_gauss(lam, -0.065, 501.1, 20.4, 26.2))
    y = (_piecewise_gauss(lam, 0.821, 568.8, 46.9, 40.5)
         + _piecewise_gauss(lam, 0.286, 530.9, 16.3, 31.1))
    z = (_piecewise_gauss(lam, 1.217, 437.0, 11.8, 36.0)
         + _piecewise_gauss(lam, 0.681, 459.0, 26.0, 13.8))
    return x, y, z


def spectrum_to_xyz(lam_nm, intensity):
    """可視スペクトル(強度 vs 波長 nm)を CIE XYZ に積分する(Y で正規化)。"""
    lam = np.asarray(lam_nm, dtype=float)
    I = np.asarray(intensity, dtype=float)
    xb, yb, zb = cie_color_matching(lam)
    X = np.trapezoid(I * xb, lam)
    Y = np.trapezoid(I * yb, lam)
    Z = np.trapezoid(I * zb, lam)
    norm = np.trapezoid(yb, lam)
    return X / norm, Y / norm, Z / norm


def xyz_to_srgb(X, Y, Z):
    """CIE XYZ → sRGB(標準変換行列 + ガンマ、借用)。0..1 にクリップして返す。"""
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    out = []
    for c in (r, g, b):
        c = max(0.0, min(1.0, float(c)))
        c = 1.055 * c ** (1 / 2.4) - 0.055 if c > 0.0031308 else 12.92 * c
        out.append(max(0.0, min(1.0, c)))
    return tuple(out)


def srgb_to_hex(rgb):
    """sRGB(0..1 の3組)を "#rrggbb" に変換する。"""
    return "#" + "".join(f"{int(round(255 * c)):02x}" for c in rgb)


def eps2_from_hamiltonian(hamiltonian, omega, xi, n_occ=None):
    """
    実空間ハミルトニアンから直接 ε₂(ω) を組む(**モジュール合流**: disorder.py の
    乱れた鎖をそのまま食わせられる)。

    対角化して固有値 E_i・固有ベクトルを得たあと、占有(下位 n_occ 個)から
    非占有への遷移を、**双極子行列要素** |⟨i|x̂|j⟩|²(x̂ はサイト座標)で重み付けし、
    エネルギー保存を kappalogic のデルタ検出器で課す:

        ε₂(ω) = (1/N) Σ_{i∈occ, j∈unocc} |⟨i|x̂|j⟩|² · δ_ξ(E_j - E_i - ω)

    バンド構造版(dielectric_imaginary)が k 空間の直接遷移だったのに対し、
    こちらは**並進対称性が無くてよい**——乱れた系に使えるのが要点。
    """
    H = np.asarray(hamiltonian, dtype=float)
    N = H.shape[0]
    E, V = np.linalg.eigh(H)
    if n_occ is None:
        n_occ = N // 2
    Xop = (V.T * np.arange(N, dtype=float)) @ V      # <i|x|j>
    occ = np.arange(n_occ)
    unocc = np.arange(n_occ, N)
    dE = E[unocc][None, :] - E[occ][:, None]
    M2 = np.abs(Xop[np.ix_(occ, unocc)]) ** 2
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.array([np.sum(M2 * delta_approx(dE - w, xi)) for w in omega_arr]) / N
    return out if np.ndim(omega) else float(out[0])


def color_saturation(rgb):
    """
    色の**彩度**(max(R,G,B) - min(R,G,B))。0 なら無彩色(灰/白/黒)。
    乱れが色を「洗い流す」度合いを測る素直な指標(下記 disordered_material_color)。
    """
    return float(max(rgb) - min(rgb))


def material_color_from_hamiltonian(hamiltonian, xi, thickness=8.0, n_occ=None,
                                    lam_min=380.0, lam_max=780.0, n_lam=161):
    """
    実空間ハミルトニアン(乱れていてよい)から、その物質を透過した光の色を計算する。
    eps2_from_hamiltonian → 吸収 → 透過率 → CIE → sRGB。

    **合流の成果(実測)**: 交替ホッピング鎖(clean gap 2.4 eV、N=120)に箱型乱れ W を
    入れ、12配位でスペクトルをアンサンブル平均して色を出すと——

        W=0.0 #ffc01a(彩度229)  W=1.0 #ff823c(195)  W=2.0 #e87a58(144)
        W=3.0 #e1a46e(115)      W=4.0 #deb991(77)

    **彩度が単調に落ちる=乱れが色を洗い流す**。機構はギャップ内吸収(Urbach 裾)で、
    ε₂(1.2 eV) は W≤2 で 0、W=3 で 0.0081、W=4 で 0.0294 と立ち上がる。

    正直な注記: 輝度(明るさ)は単調ではない(W=2 で最も暗く、W≥3 でむしろ明るく
    なる)。強い乱れはバンド構造を壊してスペクトル重みを広いエネルギーへ薄く
    分散させるため、可視域の吸収が弱まって淡くなるからで、**単調な指標は彩度の方**。

    戻り値: dict(rgb, hex, saturation, wavelengths, transmittance)
    """
    lam = np.linspace(lam_min, lam_max, n_lam)
    E = HC_EV_NM / lam
    e2 = eps2_from_hamiltonian(hamiltonian, E, xi, n_occ)
    T = np.exp(-absorbance(E, e2) * thickness)
    rgb = xyz_to_srgb(*spectrum_to_xyz(lam, T))
    return {"rgb": rgb, "hex": srgb_to_hex(rgb), "saturation": color_saturation(rgb),
            "wavelengths": lam, "transmittance": T}


def transmitted_color(transition_energies, xi, thickness=1.0,
                      lam_min=380.0, lam_max=780.0, n_lam=401, weights=None):
    """
    バンド構造(遷移エネルギー ΔE_k、単位 eV)から、その物質を透過した光の
    **色**を計算する。

    手順: 可視域の各波長 λ → 光子エネルギー E=HC/λ → ε₂(E) を kappalogic の
    デルタ検出器で計算 → 吸収 α∝E·ε₂ → 透過率 T=exp(-α·thickness) →
    CIE 等色関数で積分 → XYZ → sRGB。

    直感: ギャップが可視域より大きい(>~3.1 eV)なら可視光を吸わず**無色透明**、
    ギャップが可視域の中程(~2 eV)なら青〜緑を吸って**赤〜橙**に見え、
    ギャップが小さい(<~1.6 eV)なら可視光を全部吸って**黒**に近づく。

    戻り値: dict(rgb, hex, wavelengths, transmittance)
    """
    lam = np.linspace(lam_min, lam_max, n_lam)
    E = HC_EV_NM / lam
    e2 = dielectric_imaginary(E, transition_energies, xi, weights)
    alpha = absorbance(E, e2)
    T = np.exp(-alpha * thickness)
    rgb = xyz_to_srgb(*spectrum_to_xyz(lam, T))
    return {"rgb": rgb, "hex": srgb_to_hex(rgb),
            "wavelengths": lam, "transmittance": T}
