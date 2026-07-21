# CUDA SP2 — GPU 版の線形スケーリング密度行列(実験・任意)

`kappalogic.matrix_backend.sp2_density_matrix`(SP2 spectral-projection 純化、
T=0 密度行列を行列積だけで作る線形スケーリング電子構造法)の **GPU 実装**。
SP2 は matmul が主体なので GPU との相性が良い、という点の実証。

**これは実験的な外付けデモで、pytest のテストスイートには含まれません**
(CUDA ツールチェーンと NVIDIA GPU が必要なため)。本体(`kappalogic/`)は
純粋な CPU/numpy 実装のままで、これに依存しません。

## 必要なもの

- NVIDIA GPU
- CUDA Toolkit(`nvcc`)+ cuBLAS
- Windows: Visual Studio(MSVC)。`build.bat` 内の `vcvars64.bat` のパスを
  自分の環境に合わせて調整する。

## ビルドと実行

```bat
build.bat
sp2_cuda.exe 2048 1024      REM n=2048, n_occ=1024
```

## 実測(NVIDIA RTX 4060 Laptop, CUDA 13.2, FP32)

```
SP2 on GPU (cuBLAS FP32): n=2048, n_occ=1024
  converged at iter=63  best_idem(FP32)=2.96e-05  trace=1024.0000 (target 1024)
  GPU SP2 time: ~490 ms  (64 iters of 2048x2048 SGEMM)
  single 2048x2048 matmul: naive-CPU(ijk)=~4550 ms  GPU(cuBLAS)=~3.8 ms  =~1190x
```

つまり SP2 は GPU 上で正しく収束し(trace が n_occ に厳密、冪等性
`||X - X^2||` が FP32 相応の ~1e-5 まで純化)、cuBLAS の SGEMM は naive な
CPU 三重ループ比で ~1000 倍以上速い。

## 正直な限界

- **精度は FP32**。CPU の numpy 実装(FP64)が冪等性 ~1e-15 まで純化するのに
  対し、この GPU 版は ~1e-5 止まり。DFT では混合/低精度 FOE は確立した技法
  (README が引く "Tensor Core での混合精度 FOE" 等)だが、機械精度が要る用途
  では FP64(RTX 4060 のような民生 GPU では FP64 スループットが 1/64 で遅い)や
  混合精度の反復改良が要る。
- **速度比較の基準が甘い**。CPU 側は最適化されていない素朴な三重ループなので、
  "~1190x" は誇張気味。MKL/OpenBLAS の SGEMM と比べれば差は大幅に縮む。ここで
  示したのは「GPU で SP2 が正しく走り、cuBLAS で実用的に高速化できる」という
  存在証明であって、厳密な速度優位性の主張ではない。
- **密行列のみ**。真の線形スケーリング O(N) は、近視性(密度行列の指数減衰、
  `examples/linear_scaling_dft_demo.py` 参照)を使って疎行列積に置き換えて
  初めて実現する。cuBLAS の密 SGEMM は O(N^3) のまま(定数が小さいだけ)。

## 次の一手

- FP64 / 混合精度(FP32 反復 + FP64 仕上げ)での精度改善。
- cuSPARSE による疎行列 SP2 で真の O(N) を GPU 上で実測。
- 有限温度(金属)向けの Chebyshev 展開版(tanh の多項式近似)。
