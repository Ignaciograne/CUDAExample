
## Para ejecutar el ejemplo de multiplicación de matrices

### Compilar matmul.cu
Para compilar se usa de la siguiente manera

```
nvcc ./matmul.cu -o malmul
```

Para ejecutar el programa
```
./matmul.exe
```

## Para ejecutar los ejemplos de Edge Detection en Cuda

### Compilar edgedetection

Para compilar el programa
```
nvcc -o edge ./edgedetection.cu ./lodepng.cpp
```

Para correr el programa
```
./edge imagen.png
```
