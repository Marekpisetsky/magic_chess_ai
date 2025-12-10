#!/usr/bin/env bash
# Ejecuta el modelo NVIDIA Llama-3.1 Nemotron Nano VL 8B v1 como NIM usando Podman.

# 1) EDITA ESTA L�NEA Y PON TU API KEY DE NGC:
export NGC_API_KEY="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"

if [ "$NGC_API_KEY" = "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" ]; then
  echo "ERROR: primero edita run_nemotron_nim.sh y pon tu NGC_API_KEY."
  exit 1
fi

CONTAINER_NAME="llama-nemotron-vl-8b"
IMG_NAME="nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1:1"

# Carpeta local para cachear el modelo
LOCAL_NIM_CACHE="$HOME/.cache/nim"
mkdir -p "$LOCAL_NIM_CACHE"

echo "============================================"
echo "  Iniciando NIM Nemotron VL 8B con Podman"
echo "--------------------------------------------"
echo "  Contenedor:   $CONTAINER_NAME"
echo "  Imagen:       $IMG_NAME"
echo "  Cache local:  $LOCAL_NIM_CACHE"
echo "  Puerto host:  8000"
echo "============================================"
echo

# NOTA: Si tu Podman usa hooks de NVIDIA, puedes cambiar --device por --hooks-dir.
podman run --rm -it \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --env NGC_API_KEY="$NGC_API_KEY" \
  --shm-size=16G \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8000:8000 \
  "$IMG_NAME"