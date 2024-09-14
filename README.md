
# Para ejecutar el ejemplo de multiplicación de matrices

## cd a Carpeta
```
cd /CUDA/MatrixMultiplication
```

## Compilar matmul.cu
Para compilar se usa de la siguiente manera

```
nvcc ./matmul.cu -o malmul
```

Para ejecutar el programa
```
./matmul.exe
```

# Para ejecutar los ejemplos de Edge Detection

## En Cuda

## cd a Carpeta
```
cd /CUDA/EdgeDetection
```

### Compilar edgedetection

Para compilar el programa
```
nvcc -o edge ./edgedetection.cu ./lodepng.cpp
```

Para correr el programa
```
./edge ../../Imgs/imagenN.png 
```
Donde N es la imagen que se quiere analizar.

#### Nota
Este código hace uso como base de código del usuario lukas783 de github para la conversión de imagenes a bytes.


- [CUDA-Sobel-Filter](https://github.com/lukas783/CUDA-Sobel-Filter)

## En Neo Arms

## cd a Carpeta
```
cd /ARMNeon
```

Para compilar el programa
```
nvcc -o armneon armneon.cu `pkg-config --cflags --libs opencv4
```

Para correr el programa
```
./armneon ../Imgs/imagenN.jpg
```
Donde N es la imagen que se quiere analizar.