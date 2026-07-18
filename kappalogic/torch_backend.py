"""
kappalogic.torch_backend
==========================
v0.18: TODO.md H項「PyTorch対応」への着手。

**なぜ入れたか**: ここまでの探索(命題4・6など)は、numpyの有限差分
(h=1e-6での中心差分)や、数十万点のグリッド探索で最適点を探す、
という力技に頼っていた。PyTorchのautograd(自動微分)と勾配法の
最適化(Adam等)を使えば、
    - 閉形式の検証が「有限差分の打ち切り誤差」なしに、機械精度に近い
      精度でできる
    - 最適点(共鳴点)を、数十万点のグリッド探索ではなく数百ステップの
      勾配降下で見つけられる(計算コストが大幅に下がる)
    - (a,b)平面全体のような2次元以上の"危険領域"の地図を、
      バッチ処理でまとめて計算できる
ようになる。ユーザーの提案(「計算をラクにするため、トークンを
消費しないためにpytorchとか統合する?」)を受けて、この目的に絞って
最小限の橋渡しを実装した。

**位置づけ**: これはkappalogic本体の"ニューラルネット対応"
(TODO.md H項が本来想定していた、torch.nn.Moduleとしての利用)には
まだ踏み込んでいない。あくまで「研究・検証を効率化するための道具」
としての最小限の統合。torchはオプション依存(pyproject.tomlの
[project.optional-dependencies]の`torch`エクストラ)であり、
torchがインストールされていなくてもkappalogic本体は問題なく動く。

すべての関数は`core.py`のnumpy版と数学的に同一の定義
(kernel="tanh"のみ対応、他カーネルは今回は移植していない)。
"""
try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "kappalogic.torch_backend requires PyTorch, which is an optional "
        "dependency. Install it with `pip install kappalogic[torch]` or "
        "`pip install torch`."
    ) from e


def k(x, xi=1e-3):
    """kappa(x/xi) = tanh(x/xi)。torch版のsgn/kappa(kernel="tanh"限定)。"""
    return torch.tanh(x / xi)


def reg(x, xi=1e-3):
    """reg(x) = tanh(x/xi)^2。"""
    return torch.tanh(x / xi) ** 2


def NOT(x, xi=1e-3):
    return 1 - reg(x, xi)


def AND(a, b, xi=1e-3):
    return reg(a * b, xi)


def OR(a, b, xi=1e-3):
    return NOT(NOT(a, xi) * NOT(b, xi), xi)


def AND_n(*vals, xi=1e-3):
    prod = vals[0]
    for v in vals[1:]:
        prod = prod * v
    return reg(prod, xi)


def OR_n(*vals, xi=1e-3):
    prod = 1 - reg(vals[0], xi)
    for v in vals[1:]:
        prod = prod * (1 - reg(v, xi))
    return 1 - reg(prod, xi)


def autograd_gradient(fn, *args, xi=1e-3):
    """
    fn(*args, xi=xi) の各引数についての勾配をtorch.autogradで計算する。
    有限差分(h次第で誤差が乗る)ではなく、計算グラフに基づく厳密な
    微分なので、命題4・6のような閉形式の検証に使うと打ち切り誤差が
    完全に無くなる。

    引数はfloat/np.ndarrayで渡してよい(内部でrequires_grad=Trueの
    torch tensorに変換する)。戻り値: 各引数に対応する勾配(torch tensor)
    のタプル。
    """
    tensors = [torch.as_tensor(a, dtype=torch.float64).requires_grad_(True) for a in args]
    out = fn(*tensors, xi=xi)
    grads = torch.autograd.grad(out.sum(), tensors)
    return grads


def find_argmax_1d(fn, v0=1.0, lr=0.05, steps=2000):
    """
    fn(v) (torch tensor -> torch tensor, スカラー)を、勾配上昇法
    (Adam)で最大化するvを探す。命題6のようなargmax探索を、
    数十万点のグリッド探索ではなく数百〜数千ステップの最適化に
    置き換えるためのヘルパー。

    戻り値: 収束したvの値(float)。
    """
    v = torch.tensor([float(v0)], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = -fn(v)
        loss.backward()
        opt.step()
    return float(v.detach()[0])


def or_danger_landscape(xi, u_range=(0.01, 6.0), v_range=(0.01, 6.0), n=400):
    """
    OR(a,b;xi)の勾配の大きさ |grad OR| を、u=a/xi, v=b/xi の
    グリッド全体についてバッチでまとめて計算する(2次元の"危険地図")。
    torch.autogradを使い、n*n点分の勾配を1回のbackwardでまとめて
    計算できるので、numpyの二重forループで有限差分するより大幅に速い。

    戻り値: {"u": u軸の値(n,), "v": v軸の値(n,),
             "grad_mag": |grad OR|の値(n,n)のtorch tensor}
    """
    u = torch.linspace(u_range[0], u_range[1], n, dtype=torch.float64)
    v = torch.linspace(v_range[0], v_range[1], n, dtype=torch.float64)
    U, V = torch.meshgrid(u, v, indexing="ij")
    a = (U * xi).clone().requires_grad_(True)
    b = (V * xi).clone().requires_grad_(True)

    out = OR(a, b, xi)
    grad_a, grad_b = torch.autograd.grad(out.sum(), [a, b])
    grad_mag = torch.sqrt(grad_a ** 2 + grad_b ** 2)
    return {"u": u, "v": v, "grad_mag": grad_mag.detach()}


def exact_danger_curve_residual(a, b, xi):
    """
    命題6の「(ii)の共鳴条件」を表す厳密な(漸近近似なしの)式:
        (1-reg(a;xi)) * (1-reg(b;xi)) = xi * u*      (u* = arctanh(1/sqrt(3)))
    の左辺-右辺(残差)を返す。この式=0を満たす(a,b)の軌跡が、
    "第二共鳴"が起きる(a,b)平面上の曲線(命題6のv*はこの曲線上の
    a=c*xi断面での話だった)。torch実装なのでautogradでこの曲線に
    沿った勾配やヤコビアンも計算できる。
    """
    u_star = torch.arctanh(torch.tensor(1.0 / (3.0 ** 0.5), dtype=torch.float64))
    p = (1 - reg(a, xi)) * (1 - reg(b, xi))
    return p - xi * u_star
