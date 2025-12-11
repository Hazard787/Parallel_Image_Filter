#ifndef MY_IMAGE_FILTER_CUH
#define MY_IMAGE_FILTER_CUH

#include <cmath>
#include <vector>
#include <stdio.h>
#include <stdint.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// safe clamp for host/device
__host__ __device__ inline int clampi(int v, int a, int b) {
    return (v < a ? a : (v > b ? b : v));
}

// ======================================================
// Kernel generators (1D Gaussian and box 1D)
// ======================================================
inline std::vector<float> createGaussianKernel1D(int size, float sigma) {
    if (size % 2 == 0) ++size;
    std::vector<float> k(size);
    int half = size / 2;
    float sum = 0.0f;
    for (int i = -half; i <= half; ++i) {
        float v = expf(-(i*i) / (2.0f * sigma * sigma));
        k[i + half] = v;
        sum += v;
    }
    for (int i = 0; i < size; ++i) k[i] /= sum;
    return k;
}

inline std::vector<float> createBoxKernel1D(int size) {
    if (size % 2 == 0) ++size;
    std::vector<float> k(size, 1.0f / float(size));
    return k;
}

// ======================================================
// CPU filter (separable convolution using 1D kernel)
// - in/out are RGB (3 channels)
// - K is 1D kernel of length ks (ks odd)
// ======================================================
inline void cpu_filter_separable_rgb(
    const unsigned char* in,
    unsigned char* out,
    int w, int h, int ch,
    const float* K, int ks)
{
    int half = ks / 2;
    // temp buffer for intermediate horizontal pass
    std::vector<float> tmp(float(w * h * ch));
    // horizontal pass -> tmp (float)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c) {
                float sum = 0.0f;
                for (int k = -half; k <= half; ++k) {
                    int xx = clampi(x + k, 0, w - 1);
                    int idx = (y * w + xx) * ch + c;
                    sum += K[k + half] * float(in[idx]);
                }
                tmp[(y * w + x) * ch + c] = sum;
            }
        }
    }
    // vertical pass from tmp -> out (clamped)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c) {
                float sum = 0.0f;
                for (int k = -half; k <= half; ++k) {
                    int yy = clampi(y + k, 0, h - 1);
                    sum += K[k + half] * tmp[(yy * w + x) * ch + c];
                }
                int o = (y * w + x) * ch + c;
                int v = int(sum + 0.5f);
                out[o] = (unsigned char)clampi(v, 0, 255);
            }
        }
    }
}

// ======================================================
// CUDA separable kernels (horizontal and vertical)
// - uses global memory for simplicity and correctness
// - kernel is passed as array of floats (1D)
// ======================================================
__global__ void separable_horiz_rgb(
    const unsigned char* d_in,
    float* d_tmp, // float intermediate buffer
    const float* d_kernel,
    int w, int h, int ch, int ks)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int half = ks / 2;
    int base = (y * w + x) * ch;
    for (int c = 0; c < ch; ++c) {
        float s = 0.0f;
        for (int k = -half; k <= half; ++k) {
            int xx = x + k;
            xx = (xx < 0 ? 0 : (xx >= w ? (w-1) : xx));
            int idx = (y * w + xx) * ch + c;
            s += d_kernel[k + half] * float(d_in[idx]);
        }
        d_tmp[base + c] = s;
    }
}

__global__ void separable_vert_rgb(
    const float* d_tmp,
    unsigned char* d_out,
    const float* d_kernel,
    int w, int h, int ch, int ks)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int half = ks / 2;
    int base = (y * w + x) * ch;
    for (int c = 0; c < ch; ++c) {
        float s = 0.0f;
        for (int k = -half; k <= half; ++k) {
            int yy = y + k;
            yy = (yy < 0 ? 0 : (yy >= h ? (h-1) : yy));
            s += d_kernel[k + half] * d_tmp[(yy * w + x) * ch + c];
        }
        int val = int(s + 0.5f);
        d_out[base + c] = (unsigned char)clampi(val, 0, 255);
    }
}

// ======================================================
// Launcher: launch_filter_separable
// - copies kernel (1D) to device, allocates intermediate float buffer
// - returns GPU ms measured using cuda events
// ======================================================
inline float launch_filter_separable(
    const unsigned char* h_in,
    unsigned char* h_out,
    int w, int h,
    const float* h_kernel,
    int ks)
{
    int ch = 3;
    size_t imgBytes = size_t(w) * h * ch;
    size_t tmpBytes = sizeof(float) * size_t(w) * h * ch;
    size_t kBytes = ks * sizeof(float);

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;
    float* d_tmp = nullptr;
    float* d_kernel = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, tmpBytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kBytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    separable_horiz_rgb<<<grid, block>>>(d_in, d_tmp, d_kernel, w, h, ch, ks);
    // Ensure horiz done before vertical
    separable_vert_rgb<<<grid, block>>>(d_tmp, d_out, d_kernel, w, h, ch, ks);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, imgBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tmp);
    cudaFree(d_kernel);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

#endif // MY_IMAGE_FILTER_CUH
