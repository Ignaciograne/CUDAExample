#!/bin/bash

# Este script ejecuta './edge' en cinco imágenes secuenciales.

for i in {1..5}
do
  echo "Procesando imagen$i.png..."
  ./edge ../imagen$i.png
  echo "Imagen $i procesada."
done

echo "Procesamiento completado para todas las imágenes."
