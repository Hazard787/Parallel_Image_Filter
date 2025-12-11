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

// safe clamp for host+device
__host__ __device__ inline int clampi(int v, int low, int high) {
    return (v < low ? low : (v > high ? high : v));
}

// ------------------------ kernel generators ------------------------
inline std::vector<float> createGaussianKernel(int size, float sigma) {
    if (size % 2 == 0) size++;        // enforce odd
    if (size < 1) size = 3;
    const int half = size / 2;
    std::vector<float> K(size * size);
    float sum = 0.0f;
    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            float v = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            K[(y + half) * size + (x + half)] = v;
            sum += v;
        }
    }
    for (auto &v : K) v /= sum;
    return K;
}

inline std::vector<float> createBoxKernel(int size) {
    if (size % 2 == 0) size++;
    if (size < 1) size = 3;
    std::vector<float> K(size * size, 1.0f / (size * size));
    return K;
}

inline std::vector<float> createSharpenKernel() {
    return { 0, -1, 0,
            -1,  5,-1,
             0, -1, 0 };
}

inline std::vector<float> createEdgeKernel() {
    return { -1, -1, -1,
             -1,  8, -1,
             -1, -1, -1 };
}

inline std::vector<float> createSobelX() {
    return { -1, 0, 1,
             -2, 0, 2,
             -1, 0, 1 };
}

inline std::vector<float> createSobelY() {
    return { -1,-2,-1,
              0, 0, 0,
              1, 2, 1 };
}

// ------------------------ CPU reference (RGB) ------------------------
inline void cpu_filter_rgb(
    const unsigned char* in,
    unsigned char* out,
    int w, int h, int ch,
    const float* K, int ks)
{
    const int half = ks / 2;
    const int stride = w * ch;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float accR = 0.0f, accG = 0.0f, accB = 0.0f;
            for (int ky = -half; ky <= half; ++ky) {
                int yy = clampi(y + ky, 0, h - 1);
                for (int kx = -half; kx <= half; ++kx) {
                    int xx = clampi(x + kx, 0, w - 1);
                    int gid = (yy * w + xx) * ch;
                    float coeff = K[(ky + half) * ks + (kx + half)];
                    accR += coeff * float(in[gid + 0]);
                    accG += coeff * float(in[gid + 1]);
                    accB += coeff * float(in[gid + 2]);
                }
            }
            int outi = (y * w + x) * ch;
            int r = (int)(accR + 0.5f);
            int g = (int)(accG + 0.5f);
            int b = (int)(accB + 0.5f);
            out[outi + 0] = (unsigned char)clampi(r, 0, 255);
            out[outi + 1] = (unsigned char)clampi(g, 0, 255);
            out[outi + 2] = (unsigned char)clampi(b, 0, 255);
        }
    }
}

// ------------------------ CUDA kernel (RGB, generic ks) ------------------------
__global__ void cuda_filter_rgb(
    const unsigned char* in,
    unsigned char* out,
    const float* K,
    int w, int h, int ch, int ks)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int half = ks / 2;
    float accR = 0.0f, accG = 0.0f, accB = 0.0f;

    for (int ky = -half; ky <= half; ++ky) {
        int yy = y + ky;
        // clamp
        if (yy < 0) yy = 0;
        else if (yy >= h) yy = h - 1;
        for (int kx = -half; kx <= half; ++kx) {
            int xx = x + kx;
            if (xx < 0) xx = 0;
            else if (xx >= w) xx = w - 1;
            int idx = (yy * w + xx) * ch;
            float coeff = K[(ky + half) * ks + (kx + half)];
            accR += coeff * float(in[idx + 0]);
            accG += coeff * float(in[idx + 1]);
            accB += coeff * float(in[idx + 2]);
        }
    }

    int outi = (y * w + x) * ch;
    int r = (int)(accR + 0.5f);
    int g = (int)(accG + 0.5f);
    int b = (int)(accB + 0.5f);
    out[outi + 0] = (unsigned char)clampi(r, 0, 255);
    out[outi + 1] = (unsigned char)clampi(g, 0, 255);
    out[outi + 2] = (unsigned char)clampi(b, 0, 255);
}

// ------------------------ CUDA launcher (safe) ------------------------
inline float launch_filter_rgb(
    const unsigned char* h_in,
    unsigned char* h_out,
    int w, int h, int ch,
    const float* h_kernel,
    int ks)
{
    if (ks <= 0) return 0.0f;
    if (ks % 2 == 0) --ks; // force odd (defensive)
    if (ks < 1) ks = 3;
    if (ks > 99) ks = 99;  // safety cap

    size_t imgBytes = size_t(w) * h * ch;
    size_t kernelBytes = size_t(ks) * ks * sizeof(float);

    unsigned char *d_in = nullptr, *d_out = nullptr;
    float *d_kernel = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_in, imgBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, imgBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_kernel, kernelBytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice));

    // choose block size conservatively
    dim3 block(16, 16);
    dim3 grid( (w + block.x - 1) / block.x, (h + block.y - 1) / block.y );

    // timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // launch kernel
    cuda_filter_rgb<<<grid, block>>>(d_in, d_out, d_kernel, w, h, ch, ks);

    // check kernel launch error
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        // free and exit gracefully
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_kernel);
        return -1.0f;
    }

    // synchronize and time
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // copy back
    CUDA_CHECK(cudaMemcpy(h_out, d_out, imgBytes, cudaMemcpyDeviceToHost));

    // cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

#endif // MY_IMAGE_FILTER_CUH
