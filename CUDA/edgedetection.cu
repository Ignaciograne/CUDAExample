#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "imageLoader.cpp"

/*struct imgData {
    imgData(unsigned char* pix = nullptr, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {}
    unsigned char* pixels;
    unsigned int width;
    unsigned int height;
};*/

// Roberts Cross Operator in CUDA (Parallel Edge Detection)
__global__
void robertsCrossEdgeDetectionKernel(imgData inputImage, imgData outputImage) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // X-coordinate
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Y-coordinate

    // Check if the thread is within image bounds
    if (row < inputImage.height - 1 && col < inputImage.width - 1) {
        int index = row * inputImage.width + col;
        int indexRight = index + 1;
        int indexBottom = index + inputImage.width;
        int indexBottomRight = indexBottom + 1;

        // Get the pixel intensities (assuming grayscale image)
        unsigned char pixel = inputImage.pixels[index];
        unsigned char pixelRight = inputImage.pixels[indexRight];
        unsigned char pixelBottom = inputImage.pixels[indexBottom];
        unsigned char pixelBottomRight = inputImage.pixels[indexBottomRight];

        // Apply Roberts Cross Operator
        int Gx = pixel - pixelBottomRight;  // Horizontal gradient
        int Gy = pixelRight - pixelBottom;  // Vertical gradient

        // Calculate gradient magnitude
        int G = (int)sqrtf(Gx * Gx + Gy * Gy);

        // Clamp the result to [0, 255]
        G = min(max(G, 0), 255);

        // Store the result in the output image
        outputImage.pixels[index] = static_cast<unsigned char>(G);
    }
}

// Roberts Cross Edge Detection (Sequential, CPU Version)
void robertsCrossEdgeDetectionCPU(imgData& inputImage, imgData& outputImage) {
    // Iterate through each pixel (except the last row and column)
    for (unsigned int row = 0; row < inputImage.height - 1; ++row) {
        for (unsigned int col = 0; col < inputImage.width - 1; ++col) {
            int index = row * inputImage.width + col;
            int indexRight = index + 1;
            int indexBottom = index + inputImage.width;
            int indexBottomRight = indexBottom + 1;

            // Get the pixel intensities (assuming grayscale image)
            unsigned char pixel = inputImage.pixels[index];
            unsigned char pixelRight = inputImage.pixels[indexRight];
            unsigned char pixelBottom = inputImage.pixels[indexBottom];
            unsigned char pixelBottomRight = inputImage.pixels[indexBottomRight];

            // Apply Roberts Cross Operator
            int Gx = pixel - pixelBottomRight;  // Horizontal gradient
            int Gy = pixelRight - pixelBottom;  // Vertical gradient

            // Calculate gradient magnitude
            int G = static_cast<int>(sqrt(Gx * Gx + Gy * Gy));

            // Clamp the result to [0, 255]
            G = std::min(std::max(G, 0), 255);

            // Store the result in the output image
            outputImage.pixels[index] = static_cast<unsigned char>(G);
        }
    }
}

int main(int argc, char*argv[]) {
    /** Check command line arguments **/
    if(argc != 2) {
        printf("%s: Invalid number of command line arguments. Exiting program\n", argv[0]);
        printf("Usage: %s [image.png]", argv[0]);
        return 1;
    }

    /** Load our img and allocate space for our modified images **/
    imgData origImg = loadImage(argv[1]);

    // Image size (example)
    unsigned int width = origImg.width;
    unsigned int height = origImg.height;

    // Allocate host memory for input and output images
    //unsigned char* h_inputPixels = new unsigned char[width * height];
    unsigned char* h_outputPixels = new unsigned char[width * height];
    unsigned char* h_outputPixels2 = new unsigned char[width * height];

    // Initialize input image data (for testing purposes, using a dummy image)
    /*for (int i = 0; i < width * height; ++i) {
        h_inputPixels[i] = static_cast<unsigned char>(rand() % 256);  // Random pixel intensities
    }*/

    // Initialize imgData structs
    imgData h_inputImage(origImg.pixels, width, height);
    imgData h_outputImage(h_outputPixels, width, height);
    imgData h_outputImage2(h_outputPixels2, width, height);

    // Allocate device memory for input and output images
    unsigned char* d_inputPixels;
    unsigned char* d_outputPixels;
    cudaMalloc(&d_inputPixels, width * height * sizeof(unsigned char));
    cudaMalloc(&d_outputPixels, width * height * sizeof(unsigned char));

    // Copy input image to device
    cudaMemcpy(d_inputPixels, origImg.pixels, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Create device imgData structs
    imgData d_inputImage(d_inputPixels, width, height);
    imgData d_outputImage(d_outputPixels, width, height);

    // Define block and grid sizes for parallel processing
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    clock_t start_d=clock();
    printf("Doing GPU Edge detection\n");
    
    // Call the kernel for parallel edge detection
    robertsCrossEdgeDetectionKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage);

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    clock_t end_d = clock();

    clock_t start_h=clock();
    printf("Doing CPU Edge detection\n");

    robertsCrossEdgeDetectionCPU(h_inputImage, h_outputImage2);
    clock_t end_h = clock();

    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    // Copy the result back to host
    cudaMemcpy(h_outputPixels, d_outputPixels, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    printf("Resulting image = %d x %d \t GPU time = %f \t CPU time = %f \n", width, height, time_d, time_h);
    // Perform CPU-based serialized edge detection (e.g., thresholding)
    //imgData h_serialOutput(h_outputPixels, width, height);
    //robertsCrossEdgeDetectCPU(h_serialOutput);

    // (Optional) Save or display the output image
    // ...

    // Clean up memory
    delete[] origImg.pixels;
    delete[] h_outputPixels;
    cudaFree(d_inputPixels);
    cudaFree(d_outputPixels);

    return 0;
}
