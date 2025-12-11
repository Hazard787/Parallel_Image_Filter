// main.cu
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

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

    // create results folder (ignore error if already exists)
    fs::create_directories(outDir);

    int width = 0, height = 0, comp = 0;
    unsigned char* input = stbi_load(inputPath.c_str(), &width, &height, &comp, 3);
    if (!input) {
        std::cerr << "Failed to load image: " << inputPath << "\n";
        return -1;
    }
    std::cout << "Loaded: " << inputPath << " (" << width << "x" << height << ")\n";
    size_t imageBytes = size_t(width) * height * 3;

    std::vector<unsigned char> cpuOut(imageBytes);
    std::vector<unsigned char> gpuOut(imageBytes);

    // open CSV to write timings
    std::ofstream csv((fs::path(outDir) / "timings.csv").string());
    csv << "filter,kernel_size,cpu_ms,gpu_ms,speedup\n";
    csv << std::fixed << std::setprecision(6);

    // iterate odd kernel sizes from 3 to 49
    for (int ks = 3; ks <= 49; ks += 2) {

        // ---------------- GAUSSIAN ----------------
        float sigma = std::max(0.8f, ks / 3.0f);
        std::vector<float> gk = createGaussianKernel(ks, sigma);

        std::cout << "Running GAUSSIAN kernel size " << ks << " (sigma=" << sigma << ") ...\n";

        // CPU timing
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_filter_rgb(input, cpuOut.data(), width, height, 3, gk.data(), ks);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // GPU timing
        float gpu_ms = launch_filter_rgb(input, gpuOut.data(), width, height, 3, gk.data(), ks);

        if (gpu_ms < 0.0f) {
            std::cerr << "GPU kernel error for gaussian ks=" << ks << " (skipping)\n";
        } else {
            double speedup = cpu_ms / (gpu_ms + 1e-9);
            csv << "gaussian," << ks << "," << cpu_ms << "," << gpu_ms << "," << speedup << "\n";

            // write output image (safe filename)
            std::ostringstream ossG;
            ossG << outDir << "/gaussian_k" << std::setw(2) << std::setfill('0') << ks << ".png";
            stbi_write_png(ossG.str().c_str(), width, height, 3, gpuOut.data(), width * 3);

            std::cout << " -> CPU: " << cpu_ms << " ms, GPU: " << gpu_ms << " ms, speedup: " << speedup << "x\n";
        }

        // ---------------- BOX ----------------
        std::cout << "Running BOX kernel size " << ks << " ...\n";
        std::vector<float> bk = createBoxKernel(ks);

        auto t2 = std::chrono::high_resolution_clock::now();
        cpu_filter_rgb(input, cpuOut.data(), width, height, 3, bk.data(), ks);
        auto t3 = std::chrono::high_resolution_clock::now();
        double cpu_ms_b = std::chrono::duration<double, std::milli>(t3 - t2).count();

        float gpu_ms_b = launch_filter_rgb(input, gpuOut.data(), width, height, 3, bk.data(), ks);

        if (gpu_ms_b < 0.0f) {
            std::cerr << "GPU kernel error for box ks=" << ks << " (skipping)\n";
        } else {
            double speedup_b = cpu_ms_b / (gpu_ms_b + 1e-9);
            csv << "box," << ks << "," << cpu_ms_b << "," << gpu_ms_b << "," << speedup_b << "\n";

            std::ostringstream ossB;
            ossB << outDir << "/box_k" << std::setw(2) << std::setfill('0') << ks << ".png";
            stbi_write_png(ossB.str().c_str(), width, height, 3, gpuOut.data(), width * 3);

            std::cout << " -> CPU: " << cpu_ms_b << " ms, GPU: " << gpu_ms_b << " ms, speedup: " << speedup_b << "x\n";
        }
    }

    csv.close();
    stbi_image_free(input);

    std::cout << "\nAll experiments finished. Results saved in folder: " << outDir << "\n";
    std::cout << "(CSV: " << (fs::path(outDir) / "timings.csv").string() << ")\n";

    return 0;
}
