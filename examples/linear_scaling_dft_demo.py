"""
linear_scaling_dft_demo.py
==========================
kappalogic の tanh 核 = Fermi Operator Expansion / SP2 純化 が、実在の
「線形スケーリング電子構造(O(N) DFT)」とどう繋がるかの実演。

線形スケーリング DFT が可能な物理的根拠は Kohn の "nearsightedness"(近視性):
ギャップのある(絶縁体的な)系では、絶対零度の密度行列 P_ij が原子間距離
|i-j| に対して**指数的に減衰**するので、遠い成分を切り捨てても精度を保てる
——疎行列として O(N) で扱える。ここでは 1D タイトバインディング鎖で:

  1. SP2 純化(matrix_backend.sp2_density_matrix、行列積のみ・対角化なし)で
     T=0 密度行列を作り、直接対角化と一致することを確認。
  2. ギャップを開くほど P_ij の減衰長が短くなる(近視性が強まる)ことを実演。
  3. kappalogic の tanh-FOE(matrix_fermi_occupation)の kT->0 極限とも一致。

実行: py -3.14 examples/linear_scaling_dft_demo.py
"""
import os
import sys

import numpy as np
from numpy.linalg import eigh, eigvalsh

# allow running directly from the repo without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappalogic.matrix_backend import sp2_density_matrix, matrix_fermi_occupation


def dimerized_chain(N, t1, t2):
    """
    交替ホッピング(SSH型)の 1D 鎖。t1 != t2 でギャップが開く(絶縁体)。
    半充填(占有=N/2)。周期境界。ホッピング差 |t1-t2| がギャップを制御。
    """
    H = np.zeros((N, N))
    for i in range(N):
        t = t1 if i % 2 == 0 else t2
        j = (i + 1) % N
        H[i, j] = H[j, i] = -t
    return H


def decay_length(P):
    """密度行列 P_0j の |j| に対する指数減衰長 λ を線形回帰で推定
    (ln|P_0j| ~ -|j|/λ)。金属(power-law)なら fit は不安定。"""
    N = P.shape[0]
    dists, logs = [], []
    for j in range(1, N // 2):
        v = abs(P[0, j])
        if v > 1e-14:
            dists.append(j)
            logs.append(np.log(v))
    if len(dists) < 3:
        return None
    slope = np.polyfit(dists, logs, 1)[0]
    return -1.0 / slope if slope < 0 else np.inf


def main():
    np.set_printoptions(precision=3, suppress=True)
    N = 40
    n_occ = N // 2
    print(f"1D dimerized tight-binding chain, N={N}, half-filling (n_occ={n_occ})\n")

    print(" (t1, t2)   gap      |SP2-exact|   |SP2-FOE(kT->0)|   decay length lambda")
    print(" " + "-" * 74)
    for t1, t2 in [(1.0, 1.0), (1.0, 0.7), (1.0, 0.4), (1.0, 0.15)]:
        H = dimerized_chain(N, t1, t2)
        w = eigvalsh(H)
        gap = w[n_occ] - w[n_occ - 1]

        P_sp2 = sp2_density_matrix(H, n_occ)
        # exact projector
        wv, V = eigh(H)
        occ = np.zeros(N); occ[:n_occ] = 1.0
        P_exact = V @ np.diag(occ) @ V.T
        err_exact = np.linalg.norm(P_sp2 - P_exact)

        mu = (w[n_occ - 1] + w[n_occ]) / 2
        P_foe = matrix_fermi_occupation(H, mu, kT=max(gap, 1e-6) * 1e-3)
        err_foe = np.linalg.norm(P_sp2 - P_foe)

        lam = decay_length(P_sp2)
        conv = "OK" if err_exact < 1e-9 else "NO GAP->発散"
        lam_s = "power-law (metal)" if (lam is None or lam == np.inf or lam > 50) else f"{lam:5.2f} sites"
        print(f" ({t1:.2f},{t2:.2f})  {gap:6.3f}   {err_exact:.2e}[{conv:>10}]  {err_foe:.2e}   {lam_s}")

    print("\n読み方:")
    print(" - **ギャップのある(絶縁体)系では SP2(行列積のみ・対角化なし)は直接")
    print("   対角化と機械精度(~1e-15)で一致。**")
    print(" - ギャップ 0(t1=t2, 金属)では SP2 は収束しない(誤差 ~0.8)——これは")
    print("   バグではなく物理: T=0 の射影(ステップ関数)はフェルミ準位が縮退した")
    print("   金属では定義が曖昧。金属は有限温度の tanh-FOE(matrix_fermi_occupation)")
    print("   が要る。SP2 が「ギャップを要求する」こと自体が近視性の裏返し。")
    print(" - 絶縁体側: ギャップを開くほど密度行列の減衰長 lambda が短くなる")
    print("   (3.78 -> 1.86 -> 0.97 sites)=P が急速に疎になる。これが絶縁体で")
    print("   O(N) DFT が効く物理的根拠(Kohn の近視性)。")
    print(" - SP2 は tanh-FOE の kT->0 極限と一致——kappalogic の『ξ->0 で離散へ』")
    print("   という中心テーマの、電子構造(密度行列)版。")


if __name__ == "__main__":
    main()
