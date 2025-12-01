#include <iostream>
#include <vector>
#include <chrono>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "my_image_filter.cuh"

std::vector<float> makeBoxKernel(int size)
{
    std::vector<float> k(size * size, 1.0f / (size * size));
    return k;
}

std::vector<float> makeSharpenKernel()
{
    return { 0, -1, 0,
            -1, 5, -1,
             0, -1, 0 };
}

std::vector<float> makeEdgeKernel()
{
    return { -1, -1, -1,
             -1,  8, -1,
             -1, -1, -1 };
}

std::vector<float> makeSobelX()
{
    return { -1, 0, 1,
             -2, 0, 2,
             -1, 0, 1 };
}

std::vector<float> makeSobelY()
{
    return { -1, -2, -1,
              0,  0,  0,
              1,  2,  1 };
}


int main(int argc, char** argv)
{
    std::string inputPath  = (argc > 1) ? argv[1] : "input.png";
    std::string outputPath = (argc > 2) ? argv[2] : "output.png";

    int width = 0, height = 0, comp = 0;
    unsigned char* input = stbi_load(inputPath.c_str(), &width, &height, &comp, 3);
    if (!input) {
        std::cerr << "Failed to load image: " << inputPath << "\n";
        return -1;
    }

    std::cout << "Loaded: " << inputPath << " (" << width << "x" << height << ")\n\n";

    // ============================
    // FILTER MENU
    // ============================
    std::cout << "Choose filter:\n";
    std::cout << "1. Gaussian Blur\n";
    std::cout << "2. Box Blur\n";
    std::cout << "3. Sharpen\n";
    std::cout << "4. Edge Detection\n";
    std::cout << "5. Sobel X\n";
    std::cout << "6. Sobel Y\n";
    std::cout << "Enter choice: ";

    int choice;
    std::cin >> choice;

    std::vector<float> kernel;
    int KERNEL_SIZE = 3;

    switch (choice)
    {
        case 1:
            KERNEL_SIZE = 5;
            kernel = createGaussianKernel(KERNEL_SIZE, 1.0f);
            break;

        case 2:
            KERNEL_SIZE = 5;
            kernel = makeBoxKernel(KERNEL_SIZE);
            break;

        case 3:
            KERNEL_SIZE = 3;
            kernel = makeSharpenKernel();
            break;

        case 4:
            KERNEL_SIZE = 3;
            kernel = makeEdgeKernel();
            break;

        case 5:
            KERNEL_SIZE = 3;
            kernel = makeSobelX();
            break;

        case 6:
            KERNEL_SIZE = 3;
            kernel = makeSobelY();
            break;

        default:
            std::cout << "Invalid choice. Using Gaussian.\n";
            KERNEL_SIZE = 5;
            kernel = createGaussianKernel(KERNEL_SIZE, 1.0f);
            break;
    }

    int channels = 3;
    size_t imageBytes = size_t(width) * height * channels;

    std::vector<unsigned char> cpuOut(imageBytes);
    std::vector<unsigned char> gpuOut(imageBytes);

    // ============================
    // CPU FILTER
    // ============================
    std::cout << "\n--- CPU Filtering ---\n";
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_filter_rgb(input, cpuOut.data(), width, height, channels, kernel.data(), KERNEL_SIZE);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "CPU Time: " << cpuTime << " ms\n";

    // ============================
    // GPU FILTER
    // ============================
    std::cout << "\n--- GPU Filtering (CUDA) ---\n";
    float gpuTime = launch_filter_rgb(input, gpuOut.data(), width, height, kernel.data(), KERNEL_SIZE);
    std::cout << "GPU Time: " << gpuTime << " ms\n";

    std::cout << "\nSpeedup: " << cpuTime / gpuTime << "x\n";

    // Save file
    stbi_write_png(outputPath.c_str(), width, height, 3, gpuOut.data(), width * 3);
    std::cout << "Saved output: " << outputPath << "\n";

    // Validate
    size_t diff = 0;
    for (size_t i = 0; i < imageBytes; i++)
        if (abs(cpuOut[i] - gpuOut[i]) > 1)
            diff++;

    std::cout << "Verification differences: " << diff << "\n";

    stbi_image_free(input);
    return 0;
}
