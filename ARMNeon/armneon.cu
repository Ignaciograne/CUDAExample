#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;

// Kernel CUDA para el filtro Robert Cross
__global__ void robertCrossKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width - 1 && y < height - 1) {
        int Gx = input[y * width + x] - input[(y + 1) * width + (x + 1)];
        int Gy = input[(y + 1) * width + x] - input[y * width + (x + 1)];

        int grad = abs(Gx) + abs(Gy);
        output[y * width + x] = min(grad, 255);  // Limitar a 255 (valor máximo en una imagen de 8 bits)
    }
}

// Función para aplicar Robert Cross en CUDA y medir el tiempo
void applyRobertCrossCUDA(const Mat &img, Mat &output) {
    int width = img.cols;
    int height = img.rows;

    size_t imgSize = width * height * sizeof(unsigned char);

    // Reservar memoria en la GPU
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, imgSize);
    cudaMalloc((void**)&d_output, imgSize);

    // Copiar la imagen de entrada a la GPU
    cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice);

    // Configurar la cuadrícula y los bloques
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Medir el tiempo de ejecución del kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Ejecutar el kernel
    robertCrossKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    // Detener la medición de tiempo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecución del kernel Robert Cross: %f ms\n", milliseconds);

    // Copiar la imagen de salida de vuelta al host
    cudaMemcpy(output.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    // Liberar memoria en la GPU
    cudaFree(d_input);
    cudaFree(d_output);
}

void processImage(const char* imagePath, const char* outputPath) {
    // Cargar la imagen en escala de grises
    Mat img = imread(imagePath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        printf("Error al cargar la imagen %s\n", imagePath);
        return;
    }

    // Crear una imagen de salida del mismo tamaño
    Mat output = Mat::zeros(img.size(), CV_8U);

    // Aplicar el filtro Robert Cross en CUDA
    applyRobertCrossCUDA(img, output);

    // Guardar la imagen resultante
    imwrite(outputPath, output);
    printf("Imagen procesada y guardada en %s\n", outputPath);
}

int main() {
    // Procesar 5 imágenes distintas y medir tiempos
    processImage("imagen1.jpg", "resultado1.jpg");
    processImage("imagen2.jpg", "resultado2.jpg");
    processImage("imagen3.jpg", "resultado3.jpg");
    processImage("imagen4.jpg", "resultado4.jpg");
    processImage("imagen5.jpg", "resultado5.jpg");

    return 0;
}

