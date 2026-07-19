# kappalogic

`kappa(x/ξ) = tanh(x/ξ)`(切り替え可能なカーネル、デフォルトは`tanh`。他に`erf`/`algebraic`)を唯一の基本ブロックにして、離散的な論理演算・比較演算・場合分けを「実数上の滑らかな式」として書き直すライブラリ。

ξ→0で厳密な離散論理に、ξ>0で微分可能な緩和になる。同じ`tanh(x/ξ)`が、AND/OR/XORのような論理ゲート・比較演算子だけでなく、熱核・統計力学(フェルミ分布)・場の理論(キンクソリトン)・双曲幾何(ゲージ理論的構造)・情報幾何(Fisher-Rao計量)など、複数の分野に共通して現れることを、実際に手を動かして検証しながら積み上げている。

PyPIに公開済み: `pip install kappalogic`

```bash
pip install -e .          # ローカル開発の場合
pip install kappalogic    # PyPI経由
pip install kappalogic[torch]  # PyTorchバックエンド(オプション)も使う場合
```

## 使用例

```python
from kappalogic import gt, AND, OR, k, sgn
from kappalogic import kink_energy_exact, fermi_occupation
from kappalogic import rapidity, or_n_fusion_is_safe, not_tower_threshold

gt(5, 3)                                    # 1.0 (5 > 3)
AND(0.9, 0.8)                                # ほぼ1.0 (両方とも十分非ゼロ)
kink_energy_exact(xi=1.0)                    # 4/3 (φ^4理論のキンクソリトンの厳密な質量)
fermi_occupation(eps=0.3, mu=0.5, kT=0.1)    # フェルミ・ディラック分布と機械精度で一致

k(5) == sgn(5)                               # True (kはsgnの別名)
rapidity(1.0, xi=0.5)                        # (1+k(x))/(1-k(x)) = exp(2x/xi)
or_n_fusion_is_safe([3.0, 0.01, 0.02], xi=0.1)  # OR_nの安全な融合条件を判定
```

## 構成

| モジュール | 内容 |
|---|---|
| `core.py` | `sgn`/`reg`/`AND`/`OR`/`NAND`/`NOR`/`XOR`/`XNOR`/比較演算子、n項融合版`AND_n`/`OR_n` |
| `funcs.py` | 整数・奇偶判定(`intf`/`par`)、`max`/`min`/`clamp`、MOD演算子、ディラックのデルタ函数近似 |
| `kernels.py` | `tanh`/`erf`/`algebraic`カーネルの切り替え |
| `theory.py` | reg/AND/OR系の勾配構造・誤分類境界の厳密な命題群(命題1〜21、証明・数値検証付き) |
| `identities.py` | `k(x)=tanh(x/ξ)`の加法定理・n倍角公式・連分数・部分分数展開・無限積・Gudermannian函数・積分形(kernel="tanh"限定) |
| `gauge.py` | ξの大域/局所変換とアフィン群構造、(x,ξ)半平面のキリングベクトル場、測地線の保存量 |
| `dynamics.py` | NOT写像の自然な放物型不動点、Koenigs座標・アベル函数・連続反復 |
| `info_theory.py` | de Bruijnの恒等式、フィッシャー情報幾何とgauge.pyの双曲計量の一致 |
| `bridge.py` | Σ(離散和)↔∫(連続積分)、Π(離散積)↔∫の橋渡し、階乗とΓ関数 |
| `torch_backend.py` | PyTorchのautogradによる高速な数値検証・可視化・学習可能なξ(オプション依存) |
| `matrix_backend.py` | 非可換(行列/演算子)版kappalogic。FOE方式で対角化せず行列版k/reg/フェルミ占有数を計算、Eigenlogic流のテンソル積で符号を区別しない版と区別する版のAND/OR、DMPT由来の高速な行列版感度解析(matrix_susceptibility) |
| `heat.py` / `quantum_well.py` | 拡散時間としてのξ、熱核・無限井戸型ポテンシャルの量子プロパゲータ |
| `field_theory.py` | φ^4理論のキンクソリトン(EOM・厳密エネルギーの証明付き)、線形安定性スペクトル(Poschl-Teller型ポテンシャル、shape mode)、キンク・反キンク衝突の動的シミュレーション |
| `stat_mech.py` / `electronic_structure.py` | 分配関数、Witten指数、フェルミ・ディラック占有数 |
| `topology.py` / `spacetime.py` | モース理論によるオイラー標数、光円錐の指示関数 |
| `applications.py` / `search.py` | クロネッカーのデルタ・コラッツ漸化式、勾配ベース探索・アニーリング |
| `examples/` | 実行できるデモ4本 |

開発の経緯・迷った点・次にやりたいことは`dev_notes.md`、進行中の課題は`TODO.md`を参照。

## 命題まとめ(理論的な柱)

`theory.py`を中心に、`reg(x;ξ)=tanh(x/ξ)²`という単一の「非単調な検出器」から作られる6つの論理ゲート(AND, OR, NAND, NOR, XOR, XNOR)の勾配構造・誤分類境界を、閉じた形の式として体系的に導出・数値検証した。すべて`tests/`に対応するテストがある。

| # | 対象 | 結果 |
|---|---|---|
| 1 | `reg'(x)`の最大値 | `x* = ξ·arctanh(1/√3)` で最大 |
| 3 | AND_nの安全条件(修正版) | 全部の値が`|a_k| > C·ξ`なら誤差`≤ n·4·exp(-2C)`(元の"部分積"条件には反例あり、修正済み) |
| 4 | ORの勾配(二重共鳴) | `d/da OR = reg'(a)·(1-reg(b))·reg'(p)`、`p=(1-reg(a))(1-reg(b))` |
| 5 | AND/ORの分類 | ANDは`a·b/ξ`という1変数に還元できる(部分ディレーション不変)、ORはできない |
| 6 | ORの第二共鳴・2D危険地図の尾根 | `v* = -½ln(ξ) - ½ln(w0/(4sech²(c)))`(`w0`は超越方程式の解、(a,b)平面全体の真の尾根でもある) |
| 7 | ゲージ理論的構造(`gauge.py`) | ξの並進+ディレーションがアフィン群Aff(1,R)、(x,ξ)半平面の双曲計量のキリングベクトル場 |
| 8 | ORの誤分類境界(2変数) | `u+v = ½ln(1/ξ) + K`、`K=½ln(16/arctanh(1/√2))` |
| 9 | OR_nの誤分類境界(n変数) | `Σu_k = ½ln(1/ξ) + K(n)`、`K(n)=½ln(4ⁿ/A)` |
| 10 | **OR_nのfold-vs-fuse安全条件** | 少なくとも1つが`ξ·(C*(ξ)+M)`を超え、かつ`ξ≤e^{-2M}`(証明に必要な技術条件、数値的に必要性も確認)なら誤差`≤exp(-4M)`。**v0.56で完全な初等証明を与えた**(数値検証だけの命題から昇格)。詳細は`papers/or_n_safety_theorem.md` |
| 11 | NANDの閾値 | `a·b ≈ √A·ξ^1.5`(ANDとは異なるスケーリング則) |
| 12 | NORの誤分類境界 | `u+v = ½ln(1/ξ) + K_NOR(ξ)`(二重対数補正) |
| 13〜15 | XOR/XNORのb=0・対角線断面 | 対数のネスト構造、対角線ではLambert W函数が登場 |
| 16 | NOT合成塔の統一理論 | 命題11・13・15を「NOTをn回重ねる」という1つの一般式に統合。`Ψ_n(n≥2)`は「XORにNOTをn-2回重ねたゲートのb=0断面」に厳密一致(n≥4にも具体的なゲートがあった) |
| 17〜18 | NOT写像の自然な不動点(`dynamics.py`) | 乗数がちょうど-1になる臨界ξ、自乗すると3次接触の縮退した放物型になる一般補題 |
| 19〜20 | XOR/XNORの2D境界 | 片方が大きい極限で`v≈½ln(4u/threshold)`という対数スケールの境界 |
| 21 | 厳密な双曲線恒等式 | `Σln(cosh(a_k/ξ)) = ½ln(1/(ξA))` — 命題6・8・9・12・13・15・19・20の"近似"の正体は、この(初等的だが見落としていた)厳密な恒等式だった。数学的には`sech=1/cosh`の言い換えに過ぎず新しい数学ではないが、複数の命題を統一する整理として有用 |
| 22 | NOT合成塔の厳密版と力学系(命題17〜18)への接続 | `S_{k-1}:=ξ·arctanh(√(1-S_k))`という厳密な逆関数(これも初等的な代数、新しい数学ではない)で命題16を非漸近化。**この逆再帰が実はNOT写像(命題17)の逆軌道そのものだったと判明**——ξ<ξ_c(命題17の臨界値)でのみ収束し、収束レートは`1/多重子`に厳密一致。無限段のNOT合成塔の収束限界と、パラボリック分岐の臨界点が同一だった、という内部的な統一が主な収穫 |

情報幾何: ガウス分布族`N(μ,σ²)`のフィッシャー情報計量が、`μ':=μ/√2`という再スケールで命題7の双曲計量と厳密に(定数2倍を除いて)一致する(ガウス曲率`K=-1/2`)。

## 論文化を目指している結果

`papers/or_n_safety_theorem.md`: 命題10(OR_nのfold-vs-fuse安全条件)を
完全な初等証明付きの定理として書き下した技術ノート(英語、学会・
プレプリント投稿の叩き台)。van Krieken (2022)・Petersen (2022, 2024)の
微分可能論理ゲート研究に対する位置づけ、証明、数値検証、正直な限界
(証明は緩い、AND側には同様の閉形式がないことの確認、必要十分条件は
未解決)まで含む。

## 検証済みの主な結果

- 原案の3件のバグを発見・修正(`(A>B)`が実質`(A≠B)`と同じ値だった、`par(x)`の偶奇判定ミス、コラッツ漸化式の分岐条件ミス)
- `kernel="erf"`が階段状初期条件の熱拡散の厳密解と一致(誤差5.7e-5)
- 有限区間の熱核・無限井戸型ポテンシャルの量子プロパゲータが固有関数展開と機械精度(~1e-13)で一致
- SUSY QMのWitten指数がβ=0.01〜10で厳密に1.000000
- キンクソリトン`tanh(x/ξ)`がφ^4理論のEOMを厳密に満たし、エネルギー`E=4/(3ξ)`を導出
- キンクの線形安定性スペクトルがl=2のPoschl-Teller型ポテンシャルに厳密一致(零モード`ω=0`、内部モード`ω=√3/ξ`、連続スペクトルは`ω≥2/ξ`から。数値対角化でも確認)
- キンク・反キンク衝突の数値シミュレーションで、低速度では"捕獲"、高速度では衝突後に分離する"脱出"という教科書的な速度依存性を定性的に再現。細かい速度走査では、捕獲領域の中に孤立した脱出の窓が現れる"共鳴ウィンドウ"らしき非単調構造も確認(窓の正確な境界は解像度に敏感)
- フェルミ・ディラック分布が`sgn(x,2kT)`と機械精度で一致

## 非可換(行列/演算子)版への拡張(進行中)

`matrix_backend.py`で、行列版のk/reg/フェルミ占有数を、対角化せずに
再帰的な倍角公式(`identities.py`の`n_tuple_angle`を行列に読み替えた
もの、いわゆるFermi Operator Expansion)で計算する。

Eigenlogic (Vourdas & Dubois, 2020)の原論文を実際に読んで確認した
ところ、多入力の論理演算は入力ごとに別々のヒルベルト空間を割り当て、
その**テンソル積**上でKronecker積により演算子を組み合わせる構成
だった(`F_AND = Π_A ⊗ Π_B`)。ここで重要な発見があった: Eigenlogic
の射影演算子`Π`(符号で真偽を区別する)は、kappalogicの`reg`(偶関数
で符号を区別しない)ではなく、`k`自体から作る"柔らかい射影"
`soft_projector(H;ξ)=(I+k(H;ξ))/2`に対応する。つまり**kappalogic
本体のAND(a,b)=「a,bが共に非ゼロか」と、標準的なBoolean論理の
AND=「a,bが共に真か」は、そもそも別の概念だった**。

このため非可換版のAND/ORは2種類実装している:
- `tensor_AND`/`tensor_OR`: kappalogic本体のAND/OR(符号を区別しない)
  をそのまま行列に一般化。1x1行列でkappalogic本体と機械精度で一致。
- `eigenlogic_AND`/`eigenlogic_OR`: Eigenlogic論文の構成を
  `soft_projector`で一般化。符号を区別する正真正銘のBoolean論理で、
  ξ→0で通常の真理値表と厳密に一致する。

いずれもH1,H2が非可換でも常にHermitianになる(v0.43のJordan積の
ような場当たり的な補正が不要)。

**Vourdasの非加法性演算子(2015)は、原論文を読んで実際に検証した
結果、kappalogicとは別物だと判明した**——真の部分空間射影演算子で
検証すると、Vourdasの式(交換子との厳密な関係)は機械精度で成立する
が、kappalogicのJordan積ベースのAND/ORは、xi→0でもこの真の
"交わり・結び"射影には収束しない(むしろ乖離が大きくなる)。無理に
対応させず、「kappalogicの範囲外」と見極めた。

**DMPT (Niklasson et al. 2024)から輸入できた実用的な高速化**:
`matrix_susceptibility`関数を追加した。`Tr[A·f(H)]`のHに関する
全微分行列が、`f(H+λA)`をλについて1方向だけ微分するだけで求まる
(成分ごとの数値微分よりO(n²)倍速い、8x8行列で実測60倍)。
kappalogic自身のmatrix_k(FOEの再帰倍角展開)にそのまま適用できる
ことを検証した(誤差~1e-9で成分ごとの数値微分と一致)。

## 先行研究との関係(誠実な位置づけ)

要素技術のほとんどは既存研究の再発明である。

| このライブラリの要素 | 対応する既存研究 |
|---|---|
| tanhベースの滑らかな指示関数 | 相分離理論(Allen-Cahn方程式、1979年〜) |
| 微分可能論理ゲート・fuzzy論理演算子の勾配消失分析 | van Krieken et al. (2022, *Artificial Intelligence*)、Petersen et al. (2022 NeurIPS, 2024 CVPR)の`difflogic`、2025〜2026年の後続研究 |
| フィッシャー情報幾何(ガウス族)=双曲平面、曲率-1/2 | Rao (1945)以来の古典的結果。Costa, Santos & Strapasson (2015)が同じ再スケール式を明示 |
| アフィン群/双曲計量のキリングベクトル場 | 双曲幾何・連続ウェーブレット変換の表現論の古典的結果("ax+b group") |
| `k(x)=tanh(x/ξ)`の加法定理・連分数・無限積 | 双曲線関数論の古典的結果(Lambertの連分数は1768年) |
| 放物型不動点(乗数-1) | 周期倍分岐そのもの(ロジスティック写像等で教科書的)。「自乗すると3次接触になる」という補題は文献に見当たらなかった |
| フェルミ・ディラック占有数=`sgn(x,2kT)` | DFT計算の電子スメアリング手法(既存技法) |
| 単一プリミティブから論理ゲート全部を演算子として合成(テンソル積で多入力を扱う) | [Eigenlogic (Vourdas, *Entropy* 2020)](https://www.mdpi.com/1099-4300/22/2/139) ([PMC版](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516549/)): 射影演算子からCayley-Hamilton定理・Lagrange補間で構成 |
| 交換子による非可換補正の定式化 | [Möbius operators and non-additive quantum probabilities (Vourdas, 2015)](https://arxiv.org/pdf/1512.07846): 射影演算子の交換子と厳密に結びつく非加法性演算子 |
| 行列版フェルミ占有数`ρ=1-tanh(β/2(H-μ))`、対角化不要の再帰展開(Fermi Operator Expansion) | [Multipole Representation of the Fermi Operator](https://arxiv.org/pdf/0812.4352)(tanhの極展開・Matsubara形式、`identities.py`のMittag-Leffler展開と同じ対象)、[Goedecker (1999) 線形スケーリング電子構造](https://arxiv.org/pdf/cond-mat/9806073)、[Tensor Coreでの混合精度FOE (2021)](https://arxiv.org/pdf/2101.06385)、[大規模DFT向けsubspace再帰FOE (2023)](https://arxiv.org/pdf/2301.04642)、[核物理へのFOE応用](https://arxiv.org/pdf/2211.09448) |
| 密度行列の摂動論(行列版の自動微分に相当) | [Susceptibility Formulation of Density Matrix Perturbation Theory (2024)](https://arxiv.org/pdf/2409.17033)、[Graph-based Quantum Response Theory (2022)](https://arxiv.org/pdf/2212.01997) |
| 非可換fuzzy論理の抽象枠組み | Pykacz (1987〜94年)以来の系譜、[量子論理の幾何学的量子化](https://arxiv.org/pdf/2004.03395)。閉形式の誤分類境界のような具体性はない |
| ξ(温度)をアニーリングしながら学習し最後に離散へ落とす運用 | [Softsign: Smooth Sign in Your Optimizer (2026)](https://arxiv.org/pdf/2605.31371)、[HESTIA: Hessian-Guided Differentiable Quantization (2026)](https://arxiv.org/pdf/2601.20745)。論理ゲートの文脈とは無関係だが同じ構造(τ→0/∞で厳密な離散に収束) |

**確認できていないこと(=おそらく未踏)**: `kappa(x/ξ)=tanh(x/ξ)`という単一の検出器プリミティブからAND/OR/NAND/NOR/XOR/XNOR全部を合成し、その誤分類境界を`log(1/ξ)`スケールの閉形式群として統一的に導出する、という構成そのものは、検索した範囲で他に見当たらなかった。標準的な微分可能論理ゲート研究は`[0,1]`上で単調な演算子(積t-norm等)を対象にしており、`reg(x)=tanh(x/ξ)²`(実数全体を受け取る非単調な検出器)とは系統が異なる。ξを陽に持つ連続緩和パラメータとして非可換版を構成する試み(`matrix_backend.py`)も、検索した範囲では他に見当たらなかった。

## 率直な限界(誇張しないための注記)

- hardモード(reg/AND/OR)は離散的な場合分けをif文なしで解析式に埋め込む用途には向くが、**それ自体を勾配降下の目的関数にしてNP完全問題を解くのには向かない**(勾配消失のため)。
- softモード+アニーリングは「うまくいく場合がある局所探索の補助」であって、P=NPを覆すようなものではない。
- 命題10(OR_nの安全条件)は十分条件であり、必要十分条件までは詰めていない。
- 個々の要素技術はほぼ全て先行研究があり、上の「命題まとめ」もその多くは既知の道具(tanhの飽和、双曲幾何、周期倍分岐)の組み合わせに過ぎない。「大発見」と呼べる新定理・新予想は今のところ出ていない。

## 可視化

![kappalogic overview](https://github.com/user-attachments/assets/34c1ae48-4430-4f56-9d0e-af0ef6f047fd)

左: OR(a,b)の誤分類地図(命題8)。a,bが両方とも明らかに非ゼロなのに、広い範囲でOR≈0(誤り)になる領域があり、その境界(u+v=一定、黒破線)がぴったり一致することが一目で分かる。
右: NOT合成塔(命題16)とANDの共鳴点のξ→0スケーリング則の比較(log-logプロット)。ゲートによって傾き(スケーリング指数)がはっきり違うことが視覚的に分かる。

![tanh smoothness as xi varies](https://github.com/user-attachments/assets/30e62061-cce7-41fe-b4a8-dcc90b8405b0)

`kappa(x/ξ)=tanh(x/ξ)`が、ξを小さくするにつれて滑らかな曲線から鋭い階段関数(符号関数)へ収束していく様子。

生成スクリプト: `visualization/overview_figure.py`(`pip install -e .`後に
`python visualization/overview_figure.py`で`kappalogic_overview.png`が
生成される)。

## テスト・サンプル実行

```bash
pytest tests/ -v   # torchが無い環境ではtorch関連のテストは自動でskip
python examples/heat_equation_demo.py  # 他、examples/内のデモも実行可能
```
