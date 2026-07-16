# kappalogic (v0.9)

`kappa(x/ξ)`(デフォルトは`tanh`、`erf`/`algebraic`にも切り替え可能)を
基本ブロックにして、離散的な論理演算・比較演算・場合分けを「実数上の
滑らかな式」として書き直すためのライブラリ。

開発の経緯・迷った点・見つけたことの詳しい記録は `dev_notes.md` を参照
(こちらは要約と現状の参照用)。

```bash
pip install -e .
```

## 構成

| モジュール | 内容 |
|---|---|
| `core.py` | sgn/reg/AND/OR/XOR/比較演算子など基本ゲート、n項融合版`AND_n`/`OR_n` |
| `funcs.py` | int/par(整数・奇偶判定)、max/min/clamp |
| `kernels.py` | `tanh`/`erf`/`algebraic`カーネルの切り替え |
| `applications.py` | クロネッカーのデルタ、コラッツ漸化式、板チョコのPoisson源項 |
| `heat.py` | ξと拡散時間tの対応、熱核によるソフトな一致度報酬 |
| `search.py` | 勾配ベース探索専用の"softモード"とアニーリング解法(`anneal_solve`, `l2_penalty`) |
| `quantum_well.py` | 有限区間の熱核・無限井戸型ポテンシャルの量子プロパゲータ(鏡像法) |
| `field_theory.py` | φ^4理論のキンクソリトン(sgnが厳密解であることの証明付き) |
| `stat_mech.py` | 熱核による分配関数、SUSY QMのWitten指数(McKean-Singer型) |
| `info_theory.py` | de Bruijnの恒等式(熱拡散のエントロピー変化率とフィッシャー情報量) |
| `topology.py` | モース理論によるオイラー標数の計算 |
| `spacetime.py` | 光円錐の指示関数 |
| `bridge.py` | Σ(離散和)と∫(連続積分)の橋渡し、階乗とΓ関数 |
| `electronic_structure.py` | フェルミ・ディラック占有数=sgn(x,2kT)、DFTの電子スメアリング |
| `theory.py` | reg/AND系の勾配構造の厳密な命題群(証明付き) |
| `examples/` | 実行できるデモ4本 |

## 使用例

```python
from kappalogic import gt, AND, par, collatz_sequence, choc_bar_source
from kappalogic import xi_of_time, heat_step_profile, anneal_solve, soft_gt, l2_penalty
from kappalogic import kink_profile, kink_energy_exact, fermi_occupation

gt(5, 3)                          # 1.0
kink_energy_exact(xi=1.0)         # 4/3 (ソリトンの厳密な質量)
fermi_occupation(eps=0.3, mu=0.5, kT=0.1)  # フェルミ・ディラック分布と機械精度で一致
```

## 検証済みの主な結果

**v0.1で見つかった原案のバグ3件(修正済み、回帰テストあり)**
- `(A>B)`,`(A<B)`: `reg()`が符号を2乗で消すため、実際には`(A≠B)`と同じ値になっていた
- `par(x)`: 偶数で`-1`を返していた(`int(x)-int(x/2)`に修正)
- コラッツ漸化式: 「ステップ番号の偶奇」でなく「値自体の偶奇」で分岐するよう修正

**検証済みの技術的な発見**
- `kernel="erf"`が階段状初期条件の熱拡散の厳密解と一致(有限差分と誤差5.7e-5)
- 有限区間の熱核・無限井戸型ポテンシャルの量子プロパゲータが鏡像法+εプリスクリプションで固有関数展開と機械精度(diff~1e-13)で一致
- SUSY QMのWitten指数がβ=0.01〜10で厳密に1.000000(McKean-Singer型)
- de Bruijnの恒等式(熱拡散のエントロピー変化率=フィッシャー情報量)が誤差1e-10で一致
- モース理論によるオイラー標数がトーラス(0)・球面(2)で教科書通り
- フェルミ・ディラック分布が`sgn(x,2kT)`と機械精度で一致(DFTの電子スメアリングと同一構造)
- Σ_{n=0}^N n! と∫_0^{N+1}(floor(x))!dxが相対誤差6e-5で一致
- キンクソリトン`tanh(x/ξ)`がφ^4理論のEOMを厳密に(sympyの記号微分で)満たし、厳密エネルギー`E=4/(3ξ)`を導出
- `AND_n`融合とnaive foldの不一致は「途中の部分積がξと同スケールになったとき」に起きることを2万件のランダム試行で99.95%の精度で特定(`fusion_is_safe`)

## 先行研究との関係(誠実な位置づけ)

要素技術のほとんどは既存研究の再発明である:

| このライブラリの要素 | 対応する既存研究 |
|---|---|
| tanhベースの滑らかな指示関数 | 相分離理論(Allen-Cahn方程式、拡散界面法、1979年〜) |
| reg/ANDの勾配消失 | van Krieken et al. (2022, *Artificial Intelligence*)が命題として証明済み |
| 論理ゲート×連続緩和×アニーリング | Petersen et al. (2022, NeurIPS)、実装ライブラリ`difflogic`も既存 |
| ξ(t)=2√(Dt)のアニーリングスケジュール | 拡散モデルの"Variance Exploding SDE"と数学的に同一 |
| Dirac delta/Heaviside の微分可能な近似 | Smoothing methods for AD、mollifier理論、SPHカーネルとして既に成熟 |
| フェルミ・ディラック占有数=sgn(x,2kT) | DFT計算の電子スメアリング手法(実務で広く使われる既存技法) |

**見つけた本物の区別**: 標準的なt-norm(van Krieken論文の分析対象)は
真偽度[0,1]の範囲で単調だが、`reg(x)=tanh(x/xi)^2`は任意の実数を
受け取り「0か非0か」を判定する非単調な"検出器"であり、t-normの
枠組みには存在しない。むしろmollifier/AD平滑化の系譜に近い。この
2つの研究コミュニティを明示的に繋いだ文献は検索した範囲では見当たらなかった。
新規性を主張できるとすれば、この細い糸のみ(詳細はdev_notes.md)。

## 率直な限界(誇張しないための注記)

- hardモード(reg/AND/OR)は離散的な場合分けをif文なしで解析式に埋め込む
  用途には向くが、**それ自体を勾配降下の目的関数にしてNP完全問題を
  解くのには向かない**(勾配消失のため)。
- softモード+アニーリングは「うまくいく場合がある局所探索の補助」で
  あって、P=NPを覆すようなものではない。
- 個々の要素技術はほぼ全て先行研究があり(一部はより厳密な形で)、
  「大発見」と呼べる新定理・新予想は今のところ出ていない。

## テスト・サンプル実行

```bash
pytest tests/ -v   # 110 tests

python examples/heat_equation_demo.py
python examples/karnaugh_fusion_benchmark.py
python examples/sat_3var_demo.py
python examples/infinite_well_demo.py
```
