# CUDAExample

Taller del curso CE4302: Arquitectura de Computadores II

## Dependencias

Para compilar y ejecutar los programas incluidos en este repositorio, necesitarás las siguientes herramientas y bibliotecas:

- CUDA Toolkit: Necesario para compilar y ejecutar los programas `.cu`.
- Compilador de C++ para los archivos `.cpp` involucrados en algunos programas CUDA.
- Bash shell para ejecutar scripts de pruebas `.sh`.

Puedes descargar e instalar CUDA Toolkit desde el [sitio oficial de NVIDIA](https://developer.nvidia.com/cuda-downloads).

## Ejemplo `vecadd.cu`

### Compilación

Para compilar el programa `vecadd`, utiliza el siguiente comando en la terminal:

```bash
nvcc -o vecadd vecadd.cu
```

### Ejecución

Para ejecutar el programa `vecadd`, utiliza el siguiente comando en la terminal:

```bash
./vecadd <vector_size> <block_size>
```

### Scripts

Tienes un script `run_vecadd_tests.sh` para ejecutar los tests de `vecadd.cu`. Pero, primero tienes que darle permisos de ejecución a este script.

```bash
chmod +x run_vecadd_tests.sh
```

Para ejecutar el script `run_vecadd_tests.sh`, utiliza el siguiente comando en la terminal:

```bash
./run_vecadd_tests.sh
```

## Multiplicación de Matrices con CUDA

### Compilación

Para compilar el programa de multiplicación de matrices, utiliza el siguiente comando en la terminal:

```bash
nvcc ./matmul.cu -o matmul
```

### Ejecución

Para ejecutar el programa de multiplicación de matrices, utiliza el siguiente comando en la terminal:

```bash
./matmul
```

## Dectección de bordes con CUDA

### Compilación

Para compilar el programa de detección de bordes, utiliza el siguiente comando en la terminal:

```bash
nvcc -o edge ./edgedetection.cu ./lodepng.cpp
```

### Ejecución

Para ejecutar el programa de detección de bordes, utiliza el siguiente comando en la terminal:

```bash
./edge imagen.png
```

### Scripts

Tienes un script `run_edge_tests.sh` para ejecutar los tests de `edgedetection.cu`. Pero, primero tienes que darle permisos de ejecución a este script.

```bash
chmod +x run_edge_tests.sh
```

Para ejecutar el script `run_edge_tests.sh`, utiliza el siguiente comando en la terminal:

```bash
./run_edge_tests.sh
```
