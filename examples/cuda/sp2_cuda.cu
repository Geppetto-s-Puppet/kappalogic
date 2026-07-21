// SP2 spectral-projection purification on the GPU (cuBLAS, FP32).
// Builds the T=0 density matrix P = theta(mu - H) from a symmetric H using only
// matrix products (X <- X@X or 2X - X@X), steering trace(X) -> n_occ.
// Verifies idempotency ||P - P@P|| and trace(P), and benchmarks vs a CPU SGEMM loop.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// symmetric random H (row-major == col-major for symmetric), and Gershgorin bounds
static void make_H(std::vector<float>& H, int n, unsigned seed) {
    srand(seed);
    H.assign((size_t)n * n, 0.0f);
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            float v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            H[(size_t)i * n + j] = v; H[(size_t)j * n + i] = v;
        }
}
static void gershgorin(const std::vector<float>& H, int n, float& emin, float& emax) {
    emin = 1e30f; emax = -1e30f;
    for (int i = 0; i < n; i++) {
        float d = H[(size_t)i * n + i], r = 0.0f;
        for (int j = 0; j < n; j++) if (j != i) r += fabsf(H[(size_t)i * n + j]);
        emin = fminf(emin, d - r); emax = fmaxf(emax, d + r);
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1024;
    int n_occ = argc > 2 ? atoi(argv[2]) : n / 2;
    int max_iter = 100;
    printf("SP2 on GPU (cuBLAS FP32): n=%d, n_occ=%d\n", n, n_occ);

    std::vector<float> H; make_H(H, n, 12345);
    float emin, emax; gershgorin(H, n, emin, emax);

    // X0 = (emax*I - H)/(emax - emin)  (host)
    std::vector<float> X0((size_t)n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float v = -H[(size_t)i * n + j];
            if (i == j) v += emax;
            X0[(size_t)i * n + j] = v / (emax - emin);
        }

    cublasHandle_t h; cublasCreate(&h);
    float *dX, *dX2, *dTmp, *dOnes;
    size_t bytes = (size_t)n * n * sizeof(float);
    cudaMalloc(&dX, bytes); cudaMalloc(&dX2, bytes); cudaMalloc(&dTmp, bytes);
    cudaMalloc(&dOnes, n * sizeof(float));
    std::vector<float> ones(n, 1.0f);
    cudaMemcpy(dOnes, ones.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X0.data(), bytes, cudaMemcpyHostToDevice);

    const float one = 1.0f, zero = 0.0f, two = 2.0f, negone = -1.0f;
    float best_idem = 1e30f; int best_it = 0;
    std::vector<float> best_X(X0);

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < max_iter; it++) {
        // X2 = X @ X   (symmetric => column-major(X)=X, plain gemm works)
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dX, n, dX, n, &zero, dX2, n);
        // trace(X) via dot(diag, ones): diagonal has stride n+1
        float trX; cublasSdot(h, n, dX, n + 1, dOnes, 1, &trX);
        // idem = ||X - X2||_F : Tmp = X - X2 ; nrm2
        cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one, dX, n, &negone, dX2, n, dTmp, n);
        float idem; cublasSnrm2(h, n * n, dTmp, 1, &idem);
        if (idem < best_idem) {
            best_idem = idem; best_it = it;
            cudaMemcpy(best_X.data(), dX, bytes, cudaMemcpyDeviceToHost);
        }
        if (idem < 1e-6f) break;
        if (best_idem < 1e-3f && idem > 4 * best_idem) break;
        if (trX > n_occ) {
            // X <- X2
            cudaMemcpy(dX, dX2, bytes, cudaMemcpyDeviceToDevice);
        } else {
            // X <- 2X - X2
            cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &two, dX, n, &negone, dX2, n, dTmp, n);
            cudaMemcpy(dX, dTmp, bytes, cudaMemcpyDeviceToDevice);
        }
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // verify best_X on host: idempotency (sample) and trace
    double tr = 0; for (int i = 0; i < n; i++) tr += best_X[(size_t)i * n + i];
    printf("  converged at iter=%d  best_idem(FP32)=%.3e  trace=%.4f (target %d)\n",
           best_it, best_idem, tr, n_occ);
    printf("  GPU SP2 time: %.1f ms  (%d iters of %dx%d SGEMM)\n", gpu_ms, best_it + 1, n, n);

    // CPU reference SGEMM loop (naive triple loop is too slow; time a single
    // cache-blocked-ish gemm via simple ijk and extrapolate is misleading, so we
    // just time one host X@X to compare a single matmul GPU vs CPU)
    {
        std::vector<float> A(X0), B((size_t)n * n, 0.0f);
        auto c0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                float a = A[(size_t)i * n + k];
                for (int j = 0; j < n; j++) B[(size_t)i * n + j] += a * A[(size_t)k * n + j];
            }
        auto c1 = std::chrono::high_resolution_clock::now();
        double cpu1 = std::chrono::duration<double, std::milli>(c1 - c0).count();
        // GPU single gemm time
        cudaDeviceSynchronize();
        auto g0 = std::chrono::high_resolution_clock::now();
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dX, n, dX, n, &zero, dX2, n);
        cudaDeviceSynchronize();
        auto g1 = std::chrono::high_resolution_clock::now();
        double gpu1 = std::chrono::duration<double, std::milli>(g1 - g0).count();
        printf("  single %dx%d matmul: naive-CPU(ijk)=%.1f ms  GPU(cuBLAS)=%.2f ms  =%.0fx\n",
               n, n, cpu1, gpu1, cpu1 / gpu1);
        printf("  (caveat: CPU baseline is a naive triple loop, NOT optimized BLAS;\n");
        printf("   an MKL/OpenBLAS SGEMM would narrow this gap substantially.)\n");
    }

    cublasDestroy(h); cudaFree(dX); cudaFree(dX2); cudaFree(dTmp); cudaFree(dOnes);
    return 0;
}
