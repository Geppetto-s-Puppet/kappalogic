import numpy as np
from kappalogic import (
    gate_dilation_type, nand_threshold_ab, nand_threshold_numeric,
    AND, NAND, OR, NOR, XOR, XNOR,
)


def test_and_is_dilation_invariant():
    assert gate_dilation_type("AND") is True


def test_other_five_gates_are_not_dilation_invariant():
    for gate in ("NAND", "OR", "NOR", "XOR", "XNOR"):
        assert gate_dilation_type(gate) is False


def test_classification_holds_for_multiple_parameter_choices():
    rng = np.random.default_rng(0)
    for _ in range(10):
        xi = rng.uniform(0.3, 1.0)
        # a, b as moderate multiples of xi to avoid trivial saturation (both OR->1)
        a = rng.uniform(0.2, 1.5) * xi
        b = rng.uniform(0.2, 1.5) * xi
        lam = rng.uniform(1.2, 3.0)
        assert gate_dilation_type("AND", a=a, b=b, xi=xi, lam=lam) is True
        assert gate_dilation_type("OR", a=a, b=b, xi=xi, lam=lam) is False


def test_nand_threshold_matches_numeric_and_converges():
    diffs = []
    for xi in (1e-2, 1e-4, 1e-6, 1e-8):
        pred = nand_threshold_ab(xi)
        actual = nand_threshold_numeric(xi)
        diffs.append(abs(actual / pred - 1.0))
    # 相対誤差はxiが小さくなるほど縮むはず
    assert diffs[0] > diffs[1] > diffs[2] > diffs[3]
    assert diffs[-1] < 1e-4


def test_nand_threshold_scales_like_xi_to_1p5_not_xi():
    # NANDの閾値はxi^1.5でスケールする(xi^1ではない)ことを確認
    ratios = []
    for xi in (1e-4, 1e-6):
        ratios.append(nand_threshold_ab(xi) / xi)
    # xi^1スケールなら比は一定のはずだが、xi^1.5スケールなら
    # xiを1/100にするとthreshold/xiは1/10に縮むはず
    assert ratios[1] / ratios[0] < 0.5  # 大きく縮んでいる(単純な比例ではない)

    ratios_15 = []
    for xi in (1e-4, 1e-6):
        ratios_15.append(nand_threshold_ab(xi) / xi ** 1.5)
    # xi^1.5で正規化すればほぼ一定になるはず
    assert abs(ratios_15[1] / ratios_15[0] - 1.0) < 1e-9


def test_nand_does_not_broadly_misclassify_like_or():
    # NANDは(NORやXORと違い)a,bがそこそこ非ゼロなだけの
    # "そこまで極端でない"領域では、単純にNOT(AND)らしく振る舞う
    # (=命題8のような広い誤分類領域は持たない)
    xi = 1e-3
    # a,bがxiの1〜2倍程度(命題8的にはOR側が誤分類する典型的な範囲)
    for u in (0.5, 1.0, 1.5, 2.0):
        val = float(NAND(u * xi, u * xi, xi=xi))
        # NAND(a,a)は、abがまだ小さい(閾値xi^1.5に遠く届かない)間は
        # 正しく「1に近い」(真)はず
        assert val > 0.9


def test_nor_inherits_or_style_broad_misclassification():
    # NOR = NOT(OR) は、命題8のOR誤分類領域をそのまま裏返した形で
    # 広い誤分類領域を持つ(a,bが両方非ゼロなのにNOR~1になってしまう)
    xi = 1e-3
    misclassified = 0
    for u in (0.5, 1.0, 1.5):
        val = float(NOR(u * xi, u * xi, xi=xi))
        if val > 0.5:  # 本来は非ゼロ入力同士なのでNORは0に近いべき
            misclassified += 1
    assert misclassified >= 2


def test_xor_of_identical_inputs_can_misclassify_broadly():
    # XOR(a,a)は本来常に0(偽)のはずだが、命題8的な仕組みにより
    # aが小さすぎない中途半端な範囲で誤って1になることがある
    xi = 1e-3
    found_misclassification = False
    for u in (0.5, 1.0, 2.0):
        val = float(XOR(u * xi, u * xi, xi=xi))
        if val > 0.5:
            found_misclassification = True
    assert found_misclassification
