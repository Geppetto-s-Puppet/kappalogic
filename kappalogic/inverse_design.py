"""
kappalogic.inverse_design
===========================
v0.89: 材料の**逆設計** —— 目標のスペクトルから、ハミルトニアンのパラメータを
勾配降下で求める。kappalogic 固有の強みを最も直接的に使う場所。

**なぜ kappalogic でこれができるか**: 本ライブラリは論理ゲートから物性まで、
すべてを単一のカーネル k(x;ξ)=tanh(x/ξ) とその派生(reg / NOT / delta_approx)で
組んでいる。したがって

    ハミルトニアンのパラメータ → 固有分解 → デルタ検出器 → スペクトル → 損失

の全経路が**滑らかで微分可能**であり、d(物性)/d(パラメータ) が自動微分で
そのまま取れる。標準的な電子構造コードは端から端まで微分可能ではないので、
逆問題は差分近似(パラメータ数に比例したコスト)か、物性ごとに随伴法を手で
導出することになる。ここでは ξ>0 が**滑らかさを保証する正則化**として働く。

本モジュールは PyTorch を使う(オプション依存。未インストールなら import 時に
ImportError)。numpy 版の前向き計算は optics.eps2_from_hamiltonian にある。

== 【v0.89 の主結果】ξ は離散化スケールより大きく保て ==

逆設計の損失地形が ξ にどう依存するかを、**系のサイズ N と ω 格子の細かさを
変えながら**測った(目標 t=(1.4,0.6)、初期 t=(2.2,1.9) というスペクトルが
乖離した難しい配置。目標と予測は同じ ξ で評価するので損失はパラメータのズレ
だけを反映する)。降下方向 -∇loss と真値方向のコサイン:

    ω格子/N      ξ=0.01 の|grad|   ξ=0.01 の cos    ξ=1.0 の cos
     40 / 32        2.1e+00          -0.185          **+0.293**
     40 / 48        7.6e-01          **-0.733**      **+0.298**
    120 / 48        1.5e+04          +0.207          **+0.300**
    400 / 48        1.8e+05          +0.205          **+0.301**

**大きい ξ の方向品質は全設定で +0.29〜+0.30 と安定**しているのに対し、
**小さい ξ は離散化に強く依存して不安定**: 勾配の大きさが格子の取り方で
**5桁**振れ、方向も cos=-0.733 まで悪化しうる。

理由は明快で、**ξ が離散化スケール(有限系の準位間隔 Δε、観測の ω 格子間隔 Δω)
より小さいと、デルタ検出器が物理ではなく離散化のアーティファクトを解像して
しまう**から。鋭いピークが格子の隙間に落ちれば勾配は消え、格子点に乗れば
爆発する——どちらも物理ではない。これは v0.87 で久保伝導度の広がり η を
小さくしすぎて有限系の離散準位を拾ってしまった現象と**同じ規則**である。

    **指針: ξ は max(準位間隔, 格子間隔) より十分大きく取る。**

実際に4つの初期値から Adam 250 ステップで解かせると(平均パラメータ誤差、
N=48・ω60点):

    固定 ξ=0.02 : 0.987   固定 ξ=0.15 : 0.679
    固定 ξ=1.0  : **0.414(最良、ある初期値では誤差 0.0000 で真値を厳密回復)**
    ξアニーリング(1.0→0.02) : 0.767

**論理側で確立した ξ アニーリングは、ここでは効かない**(むしろ大きい ξ で
得た良解を壊す)。理由は明快で:

    - 論理/SAT では**答えそのものが離散**(ξ→0 が本番)。だから鋭くしないと
      真の答えに到達できず、アニーリングが要る。
    - 材料の逆設計では**パラメータは連続**で、ξ は単なる観測の平滑化。
      滑らかなスペクトルだけで既にパラメータが決まる(SSH ならギャップ
      2|t1-t2| と帯域上端 2(t1+t2) の2つで一意)。鋭くしても**情報は増えず、
      地形のギザギザだけが増える**。

したがって指針は「**ξ アニーリングは答えが離散のときに使う。答えが連続で ξ が
観測パラメータに過ぎないなら、ξ は大きいまま保つ**」。これは v0.53(アニーリング
スケジュール比較で差が出なかった)や v0.59-0.60(対数領域で勾配を保つ)と並ぶ、
ξ の使い分けについての知見。

正直な限界: 検証したのは 1次元 SSH 鎖の2パラメータ(t1,t2)という小さな逆問題
のみ。パラメータ数が増えた場合、乱れを含む場合、ギャップの無い金属の場合は
未検証。「大きい ξ が常に良い」と無条件に一般化はできない——言えるのは
(a)**ξ は離散化スケールより大きく保て**(でないと物理でなくアーティファクトを
解像する)、(b)**ξ アニーリングは答えが離散のときに使う道具であって、答えが
連続で ξ が観測パラメータに過ぎないなら不要どころか有害**、という2つの判断基準。
また最良の固定 ξ でも4初期値中2つは局所解に落ちており(誤差0.9前後)、
逆問題自体が易しいわけではない。
"""
import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "kappalogic.inverse_design requires PyTorch: pip install kappalogic[torch]"
    ) from exc


def ssh_hamiltonian_torch(t1, t2, n_sites=48):
    """交替ホッピング(SSH)鎖のハミルトニアン(torch、t1/t2 について微分可能)。"""
    idx = torch.arange(n_sites - 1)
    hop = torch.where(idx % 2 == 0, t1, t2)
    return torch.diag(-hop, 1) + torch.diag(-hop, -1)


def delta_detector_torch(x, xi):
    """kappalogic のデルタ検出器 delta_approx = NOT(x;ξ)/(2ξ) の torch 版。"""
    return (1.0 - torch.tanh(x / xi) ** 2) / (2.0 * xi)


def differentiable_eps2(t1, t2, omegas, xi, n_sites=48):
    """
    ε₂(ω) を **t1,t2 について微分可能な形**で計算する(占有→非占有の双極子遷移、
    エネルギー保存は kappalogic のデルタ検出器)。optics.eps2_from_hamiltonian の
    torch 版に相当し、逆設計の前向き計算になる。
    """
    H = ssh_hamiltonian_torch(t1, t2, n_sites)
    E, V = torch.linalg.eigh(H)
    n_occ = n_sites // 2
    pos = torch.arange(n_sites, dtype=H.dtype)
    X = (V.T * pos) @ V
    dE = E[n_occ:].unsqueeze(0) - E[:n_occ].unsqueeze(1)
    M2 = X[:n_occ, n_occ:] ** 2
    return torch.stack([(M2 * delta_detector_torch(dE - w, xi)).sum() / n_sites
                        for w in omegas])


def fit_hopping_to_spectrum(target_spectrum, omegas, xi, init=(1.0, 1.0),
                            steps=250, lr=0.05, n_sites=48, xi_final=None):
    """
    目標スペクトルに合うホッピング (t1,t2) を勾配降下で求める(逆設計)。

    xi: 観測の平滑化(デルタ検出器の幅)。**上の docstring の結論に従い、
        既定では固定して使う**(大きめの ξ が良い)。
    xi_final: 与えると ξ を xi → xi_final へ幾何学的にアニールする。v0.89 の
        実験では**アニーリングは効かなかった**ので既定 None(固定)。離散的な
        答えを求める用途と対比するために残してある。

    戻り値: dict(t1, t2, loss, history)
    """
    dtype = torch.float64
    t1 = torch.tensor(float(init[0]), dtype=dtype, requires_grad=True)
    t2 = torch.tensor(float(init[1]), dtype=dtype, requires_grad=True)
    om = torch.as_tensor(np.asarray(omegas, dtype=float), dtype=dtype)
    tgt = torch.as_tensor(np.asarray(target_spectrum, dtype=float), dtype=dtype)
    opt = torch.optim.Adam([t1, t2], lr=lr)
    history = []
    for s in range(steps):
        cur_xi = xi if xi_final is None else xi * (xi_final / xi) ** (s / max(steps - 1, 1))
        opt.zero_grad()
        loss = ((differentiable_eps2(t1, t2, om, cur_xi, n_sites) - tgt) ** 2).sum()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return {"t1": float(t1.item()), "t2": float(t2.item()),
            "loss": history[-1], "history": history}


def gradient_direction_quality(t1, t2, true_t1, true_t2, omegas, xi, n_sites=48):
    """
    **v0.89 の中心的な診断**: 現在のパラメータでの降下方向 -∇loss が、真値へ向かう
    方向とどれだけ揃っているか(コサイン)を返す。

    +1 に近いほど勾配が正しい方向を指す。**ξ が小さいと負になりうる**
    (勾配は巨大なのに誤った方向を指す)ことが本モジュールの主結果。
    戻り値: dict(loss, grad_norm, cosine)
    """
    dtype = torch.float64
    a = torch.tensor(float(t1), dtype=dtype, requires_grad=True)
    b = torch.tensor(float(t2), dtype=dtype, requires_grad=True)
    om = torch.as_tensor(np.asarray(omegas, dtype=float), dtype=dtype)
    with torch.no_grad():
        tgt = differentiable_eps2(torch.tensor(float(true_t1), dtype=dtype),
                                  torch.tensor(float(true_t2), dtype=dtype),
                                  om, xi, n_sites)
    loss = ((differentiable_eps2(a, b, om, xi, n_sites) - tgt) ** 2).sum()
    loss.backward()
    g1, g2 = float(a.grad.item()), float(b.grad.item())
    gnorm = float(np.hypot(g1, g2))
    d1, d2 = float(true_t1) - float(t1), float(true_t2) - float(t2)
    dnorm = float(np.hypot(d1, d2))
    cos = (-g1 * d1 - g2 * d2) / (gnorm * dnorm) if gnorm > 0 and dnorm > 0 else 0.0
    return {"loss": float(loss.item()), "grad_norm": gnorm, "cosine": float(cos)}
