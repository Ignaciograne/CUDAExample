#!/bin/bash

# Compilar el programa CUDA
nvcc -o vecAdd vecadd.cu

# Array con tamaños de vectores
vector_sizes=(1024 1048576 262144 262144 1048576)

# Array con tamaños de bloques
block_sizes=(32 32 128 512 512)

# Array con descripciones de los tests para imprimir
descriptions=(
    "Pequeño número de hilos por bloque con tamaño de vector pequeño"
    "Pequeño número de hilos por bloque con tamaño de vector grande"
    "Número mediano de hilos por bloque con tamaño de vector mediano"
    "Gran número de hilos por bloque con tamaño de vector mediano"
    "Gran número de hilos por bloque con tamaño de vector grande"
)

# Ejecutar los tests
for i in ${!vector_sizes[@]}; do
    echo "Ejecutando test: ${descriptions[$i]}"
    echo "Tamaño del vector: ${vector_sizes[$i]}, Hilos por bloque: ${block_sizes[$i]}"
    ./vecAdd ${vector_sizes[$i]} ${block_sizes[$i]}
    echo "-----------------------------------------------------------"
done

