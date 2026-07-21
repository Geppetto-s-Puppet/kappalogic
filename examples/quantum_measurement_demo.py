"""
quantum_measurement_demo.py
===========================
kappalogic の soft logic を量子測定として読む(v0.77)。

soft_projector(H;ξ) = (I + tanh(H/ξ))/2 は命題「H>0(その状態は"真"か)」を
測る**弱測定POVM**の効果であり、ξ がその測定強度を制御する:
  ξ→0  射影(強)測定    ξ大  自明(最弱)測定

実行: py -3.14 examples/quantum_measurement_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappalogic.quantum_logic import (
    soft_povm, is_valid_povm, measurement_strength, quantum_truth_value,
)
from kappalogic.matrix_backend import eigenlogic_AND, eigenlogic_OR


def main():
    np.set_printoptions(precision=3, suppress=True)

    # a single "proposition" qubit: H = Z (|1>=true has eigenvalue +1)
    Z = np.array([[-1.0, 0.0], [0.0, 1.0]])

    print("=== ξ as measurement strength (POVM {E+,E-} for 'H>0?') ===")
    print(" ξ        valid POVM   strength S=Tr(reg)/n   E+ eigenvalues")
    for xi in [3.0, 1.0, 0.3, 0.1, 0.03]:
        Ep, Em = soft_povm(Z, xi)
        S = measurement_strength(Z, xi)
        eig = np.linalg.eigvalsh(Ep)
        print(f" {xi:4}    {str(is_valid_povm([Ep, Em])):>5}        {S:.4f}"
              f"                {eig}")
    print(" ξ→0: E+ eigenvalues → {0,1} (projective/strong). ξ大: → {0.5,0.5} (trivial/weak).")
    print(" 測定強度 S は kappalogic の検出器 reg=tanh² の平均そのもの。\n")

    print("=== quantum truth values of a 2-qubit AND/OR gate ===")
    xi = 0.01
    F_and = eigenlogic_AND(Z, Z, xi)
    F_or = eigenlogic_OR(Z, Z, xi)
    ket = {'00': [1, 0, 0, 0], '01': [0, 1, 0, 0],
           '10': [0, 0, 1, 0], '11': [0, 0, 0, 1]}
    print(" computational basis (exact truth table):")
    for s in ['00', '01', '10', '11']:
        print(f"   |{s}>: AND={quantum_truth_value(F_and, ket[s]):.2f}"
              f"  OR={quantum_truth_value(F_or, ket[s]):.2f}")

    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    prod = np.kron(plus, plus)
    print(f"\n product |+>|+> (independent 50/50): AND={quantum_truth_value(F_and, prod):.2f}"
          f"  = P(A)·P(B)=0.25")

    bell = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
    print(f" Bell (|00>+|11>)/√2 (correlated):  AND={quantum_truth_value(F_and, bell):.2f}"
          f"  OR={quantum_truth_value(F_or, bell):.2f}")
    print("   → 入力が完全相関(both-true w.p.1/2)なので AND=OR=0.5。")
    print("   正直な注記: この相関は古典的な相関源でも再現でき、非古典性の証拠では")
    print("   ない(可換な射影の同時測定の統計は古典確率で書ける)。真に非古典な現象")
    print("   (文脈依存性・CHSH破れ)は非可換命題の同時測定が要り、本デモの範囲外。")


if __name__ == "__main__":
    main()
