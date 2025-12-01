#ifndef MY_IMAGE_FILTER_CUH
#define MY_IMAGE_FILTER_CUH

#include <cmath>
#include <vector>
#include <stdio.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// ===============================================
// Safe clamp (no std::clamp needed)
// ===============================================
__host__ __device__ inline int clampi(int v, int low, int high) {
    return (v < low ? low : (v > high ? high : v));
}

// ===============================================
// PREDEFINED FILTER KERNELS
// ===============================================

// Gaussian kernel generated at runtime
inline std::vector<float> createGaussianKernel(int size, float sigma) {
    if (size % 2 == 0) size++;

    std::vector<float> kernel(size * size);
    float sum = 0;
    int h = size / 2;

    for (int y = -h; y <= h; y++) {
        for (int x = -h; x <= h; x++) {
            float v = expf(-(x*x + y*y) / (2 * sigma * sigma));
            kernel[(y + h) * size + (x + h)] = v;
            sum += v;
        }
    }
    for (float &k : kernel) k /= sum;

    return kernel;
}

// Box blur (uniform)
inline std::vector<float> createBoxKernel(int size) {
    std::vector<float> k(size * size, 1.0f / (size * size));
    return k;
}

// Sharpen kernel (fixed 3×3)
inline std::vector<float> createSharpenKernel() {
    return {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };
}

// Edge detection (Laplacian 3×3)
inline std::vector<float> createEdgeKernel() {
    return {
         0,  1,  0,
         1, -4,  1,
         0,  1,  0
    };
}

inline std::vector<float> createSobelX() {
    return {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
}

inline std::vector<float> createSobelY() {
    return {
        -1,-2,-1,
         0, 0, 0,
         1, 2, 1
    };
}

// =====================================================
// CPU FILTER (RGB)
// =====================================================
inline void cpu_filter_rgb(
    const unsigned char* in, unsigned char* out,
    int w, int h, int ch,
    const float* K, int ks)
{
    int half = ks / 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {

            float R=0, G=0, B=0;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {

                    int yy = clampi(y + ky, 0, h - 1);
                    int xx = clampi(x + kx, 0, w - 1);
                    int idx = (yy * w + xx) * ch;

                    float coeff = K[(ky + half) * ks + (kx + half)];

                    R += coeff * in[idx + 0];
                    G += coeff * in[idx + 1];
                    B += coeff * in[idx + 2];
                }
            }

            int o = (y * w + x) * ch;
            out[o + 0] = clampi((int)R, 0, 255);
            out[o + 1] = clampi((int)G, 0, 255);
            out[o + 2] = clampi((int)B, 0, 255);
        }
    }
}

// =====================================================
// CUDA KERNEL (RGB)
// =====================================================
__global__ void cuda_filter_rgb(
    const unsigned char* in,
    unsigned char* out,
    const float* K,
    int w, int h,
    int ks)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int half = ks / 2;

    float R=0, G=0, B=0;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {

            int xx = min(max(x + kx, 0), w - 1);
            int yy = min(max(y + ky, 0), h - 1);

            int idx = (yy * w + xx) * 3;
            float coeff = K[(ky + half) * ks + (kx + half)];

            R += coeff * in[idx + 0];
            G += coeff * in[idx + 1];
            B += coeff * in[idx + 2];
        }
    }

    int o = (y * w + x) * 3;
    out[o + 0] = clampi((int)R, 0, 255);
    out[o + 1] = clampi((int)G, 0, 255);
    out[o + 2] = clampi((int)B, 0, 255);
}

// =====================================================
// CUDA LAUNCHER
// =====================================================
inline float launch_filter_rgb(
    const unsigned char* h_in,
    unsigned char* h_out,
    int w, int h,
    const float* h_kernel,
    int ks)
{
    size_t imgBytes = size_t(w) * h * 3;
    size_t kernelBytes = ks * ks * sizeof(float);

    unsigned char *d_in, *d_out;
    float* d_kernel;

    CUDA_CHECK(cudaMalloc(&d_in, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelBytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    cudaEventRecord(s);

    cuda_filter_rgb<<<grid, block>>>(d_in, d_out, d_kernel, w, h, ks);

    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, imgBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);

    return ms;
}

#endif
