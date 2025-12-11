// main.cu
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "my_image_filter.cuh"

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    std::string inputPath  = (argc > 1) ? argv[1] : "input.png";
    std::string outDir     = (argc > 2) ? argv[2] : "results";

    // create results folder (if not exists)
    fs::create_directories(outDir);

    int width=0, height=0, comp=0;
    unsigned char* input = stbi_load(inputPath.c_str(), &width, &height, &comp, 3);
    if (!input) {
        std::cerr << "Failed to load image: " << inputPath << "\n";
        return -1;
    }
    std::cout << "Loaded: " << inputPath << " (" << width << "x" << height << ")\n";

    size_t imageBytes = size_t(width) * height * 3;
    std::vector<unsigned char> cpuOut(imageBytes);
    std::vector<unsigned char> gpuOut(imageBytes);

    std::ofstream csv(fs::path(outDir) / "timings.csv");
    csv << "filter,kernel_size,cpu_ms,gpu_ms,speedup\n";
    csv << std::fixed << std::setprecision(6);

    for (int ks = 3; ks <= 49; ks += 2) {
        float sigma = std::max(0.8f, ks / 3.0f);
        std::vector<float> k1d = createGaussianKernel1D(ks, sigma);

        std::cout << "Running GAUSSIAN kernel size " << ks << " (sigma=" << sigma << ") ...\n";

        // CPU timing (separable)
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_filter_separable_rgb(input, cpuOut.data(), width, height, 3, k1d.data(), ks);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // GPU timing (separable)
        float gpu_ms = launch_filter_separable(input, gpuOut.data(), width, height, k1d.data(), ks);

        double speedup = cpu_ms / (gpu_ms + 1e-9);
        csv << "gaussian," << ks << "," << cpu_ms << "," << gpu_ms << "," << speedup << "\n";

        // save result
        std::ostringstream ss;
        ss << outDir << "/gaussian_k" << std::setw(2) << std::setfill('0') << ks << ".png";
        stbi_write_png(ss.str().c_str(), width, height, 3, gpuOut.data(), width * 3);

        std::cout << " -> CPU: " << cpu_ms << " ms, GPU: " << gpu_ms << " ms, speedup: " << speedup << "x\n";
    }

    csv.close();
    stbi_image_free(input);
    std::cout << "\nDone. Results saved in: " << outDir << "\n";
    return 0;
}
