"""
kappalogic.funcs
==================
定義14〜18: int(x), par(x), max/min/clamp

注意 (実装時に見つかった原設計の誤り。詳細はdev_notes.md):
    ユーザー原案の定義15 par(x):=int(x)-2*int(x/2) は偶数で-1を返す
    バグがあったため、int(x)-int(x/2) に修正済み。

v0.21で追加: `modf`(MOD演算子)と`delta_approx`(ディラックのデルタ
函数の近似)。TODO.md B項に対応。
"""
import numpy as np
from .core import gt, lt, NOT, DEFAULT_XI, DEFAULT_KERNEL


def intf(x):
    """int(x): xが整数のとき1、さもなければ0"""
    x = np.asarray(x, dtype=float)
    return NOT(np.sin(np.pi * x))


def par(x):
    """par(x): xが奇数のとき1、さもなければ0 (0や偶数は0)"""
    x = np.asarray(x, dtype=float)
    return intf(x) - intf(x / 2)


def maxf(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """max(A,B) = (A-B)(A>B) + B"""
    return (a - b) * gt(a, b, xi, kernel) + b


def minf(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """min(A,B) = (A-B)(A<B) + B"""
    return (a - b) * lt(a, b, xi, kernel) + b


def clamp(lo, x, hi, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """clamp(A,x,B): x<A なら A、x>B なら B、さもなければ x"""
    return (lo - x) * lt(x, lo, xi, kernel) + x + (hi - x) * gt(x, hi, xi, kernel)


def modf(x, n, xi=DEFAULT_XI):
    """
    x mod n (nは正の整数)。TODO.md B項「MOD演算子を作る」に対応。

    par(x)(=intf(x)-intf(x/2)、xが奇数かどうかの検出)を一般化する
    形で作った: 「x ≡ r (mod n)」の検出は intf((x-r)/n) で書ける
    ((x-r)がnで割り切れる整数かどうかを判定するだけなので)。
    これをr=0..n-1について「rが正解の検出器の値」で重み付けして
    足し合わせれば、x mod nの値そのものが得られる:

        modf(x,n) = sum_{r=0}^{n-1} r * intf((x-r)/n)

    xが整数のとき、Pythonの`x % n`と一致する(負の数にも対応、
    floor除算の余りの慣習に従う。int(x)がxが整数かを判定する
    だけなので、この一致は`intf`の周期性から自動的に成り立つ)。
    xが整数でない場合は「近くの整数のmod値」に緩やかに近づく
    連続的な近似になる(par(x)と同じ性格の関数)。

    検証: x=5,7,-1,10,-7、n=3,4で、整数xについてPythonの`x%n`と
    完全一致することを確認済み(誤差<1e-9、tests参照)。
    """
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x)
    for r in range(n):
        total = total + r * intf((x - r) / n)
    return total


def delta_approx(x, xi=DEFAULT_XI):
    """
    ディラックのデルタ函数δ(x)の近似。TODO.md B項
    「ディラックのデルタ関数を再現する」に対応
    (応用19で軽く触れられていたが、実装はされていなかった)。

    導出: sgn(x;xi)=tanh(x/xi)を[0,1]のシグモイド形に直した
    (sgn(x;xi)+1)/2 の導関数を取ると

        d/dx[(sgn(x;xi)+1)/2] = (1/(2*xi)) * sech^2(x/xi)
                                = NOT(x;xi) / (2*xi)

    となり、reg/NOTを使って既にあるプリミティブだけで書ける。

    検証済みの性質:
    - 正規化: ∫δ_xi(x)dx = 1 (数値積分で誤差<1e-12を確認。
      ∫sech^2(x/xi)dx = 2*xi なので、2*xiで割ると厳密に1になる
      ことは解析的にも従う)
    - ふるい分け性質(sifting property): ∫δ_xi(x)*f(x)dx -> f(0)
      (xi->0の極限。数値積分でf(x)=sin(x)+2、xi=0.01のとき、
      積分値2.0000...とf(0)=2.0が一致することを確認)
    - xi->0で原点に鋭く集中していく(ピーク高さ1/(2*xi)がxi->0で発散、
      裾の減衰は指数的)
    """
    return NOT(x, xi) / (2 * xi)
