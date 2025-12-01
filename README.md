
# 🚀 CUDA Image Filtering 

This project implements **high‑performance image filtering** using **NVIDIA CUDA**, benchmarking it against a traditional **CPU implementation**.  
It demonstrates massive GPU speedups (100×–380×) using real‑world image processing workloads.

---

# 📌 Overview

The program loads an input PNG/JPG image, applies a selected filter using:

- **CPU reference implementation**
- **CUDA GPU accelerated implementation**

Then it:

- Measures CPU time (ms)
- Measures GPU time (ms)
- Calculates Speedup = CPU Time / GPU Time
- Saves processed output image
- Verifies correctness (pixel‑wise difference)

---

# 🖼️ Supported Filters

| Filter Name | Description |
|-------------|-------------|
| **Gaussian Blur** | Smoothens the image using weighted kernel |
| **Box Blur** | Simple averaging blur |
| **Sharpen** | Enhances edges, increases image clarity |
| **Edge Detection** | Detects high‑gradient edges (Laplacian kernel) |
| **Sobel X** | Horizontal edge detection |
| **Sobel Y** | Vertical edge detection |

All filters run **both on CPU & GPU** for accurate benchmarking.

---

# 📂 Project Structure

```
PP Project Final/
│── main.cu                 # Main CUDA+CPU application
│── my_image_filter.cuh     # All filter kernels & CPU implementations
│── stb_image.h             # Image loading (header-only)
│── stb_image_write.h       # Image saving (header-only)
│── input.png               # Sample input file
│── app.exe                 # Compiled executable
│── README.md               # Documentation
```

---

# ⚙️ Requirements

### ✔ Hardware
- NVIDIA GPU (Compute Capability ≥ 5.0)
- Minimum 2GB VRAM recommended for large images

### ✔ Software
- **CUDA Toolkit 11+**
- **MSVC Build Tools** (Windows)
- **NVCC compiler**
- Visual Studio Developer Command Prompt (optional)
- C++17 or later

---

# 🛠️ How to Compile (Windows)

### ✅ Method 1 — Using NVCC directly (with MSVC compiler)

```sh
nvcc main.cu -o app.exe --compiler-bindir "C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\BuildTools\VC\Tools\MSVC\<VERSION>\bin\Hostx64\x64"
```

Replace `<VERSION>` with your actual MSVC version.

### ✅ Method 2 — Using Visual Studio Developer Command Prompt

```sh
nvcc main.cu -o app.exe
```

---

# ▶️ How to Run the Program

### Basic command:

```sh
app.exe input.png output.png
```

### Program Flow:

1. Loads the image
2. Prompts for filter selection:

```
Choose filter:
1. Gaussian Blur
2. Box Blur
3. Sharpen
4. Edge Detection
5. Sobel X
6. Sobel Y
Enter choice:
```

3. Runs CPU version
4. Runs GPU CUDA version
5. Prints timing:
```
CPU Time: XXX ms
GPU Time: XXX ms
Speedup: XXXx
```
6. Saves output image
7. Shows verification status

---

# 📊 Sample Benchmark Results

| Image Index | CPU (ms) | GPU (ms) | Speedup |
|-------------|----------|----------|---------|
| 1 | 71.02 | 0.68 | 104× |
| 2 | 140.29 | 0.87 | 160× |
| 3 | 307.67 | 1.18 | 260× |
| 4 | 489.95 | 1.46 | 334× |
| 5 | 1096.02 | 2.88 | 380× |

GPU execution is consistently **100×–380× faster**.

---

# 🧪 Verification

Every output undergoes pixel‑wise comparison:

```
Verification differences: 0
```

Ensures GPU output matches CPU output (within numerical tolerance).

---

# 💡 Notes

- Works with **any PNG/JPG image**
- Large images show higher speedup
- CPU performance varies by system load & threading
- GPU results depend on core count & memory bandwidth

---

# 👨‍💻 Author

Developed by **Jainwin Boys**  

---

# ✅ License

This project is free for academic, research, and personal use.
