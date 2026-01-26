#!/usr/bin/env bash
set -euo pipefail

# Build/push the Dynamo dev base image for SGLang with arch-specific tags.
#
# Resulting tags:
# - nvcr.io/nvidian/dynamo-dev/warnold-utils:sglang-dd-base-img-v1-${ARCH}
# - cache: nvcr.io/nvidian/dynamo-dev/warnold-utils:sglang-dd-base-img-cache-${ARCH}
#
# You can override defaults via env vars:
# - CUDA_VERSION (default: 13.0.1)
# - TAG_VERSION (default: v1)
# - IMAGE_REPO (default: nvcr.io/nvidian/dynamo-dev/warnold-utils)
# - DRY_RUN=1 (print command only)
#
# Flags:
# - --local: Build locally without buildx/push, tag as sgl-dd-local
# - --no-cache-export: Skip --cache-to (faster export, no intermediate layer caching)

LOCAL_BUILD=0
NO_CACHE_EXPORT=0
for arg in "$@"; do
  case "$arg" in
    --local) LOCAL_BUILD=1 ;;
    --no-cache-export) NO_CACHE_EXPORT=1 ;;
  esac
done

CUDA_VERSION="${CUDA_VERSION:-13.0.1}"
TAG_VERSION="${TAG_VERSION:-v1}"
IMAGE_REPO="${IMAGE_REPO:-nvcr.io/nvidian/dynamo-dev/warnold-utils}"

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ARCH_TAG="amd64" ;;
  aarch64|arm64) ARCH_TAG="arm64" ;;
  *)
    echo "Unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

PLATFORM="linux/${ARCH_TAG}"
CACHE_REF="${IMAGE_REPO}:sglang-dd-base-img-cache-${ARCH_TAG}"
IMAGE_TAG="${IMAGE_REPO}:sglang-dd-base-img-${TAG_VERSION}-${ARCH_TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$LOCAL_BUILD" == "1" ]]; then
  IMAGE_TAG="sgl-dd-local"
  CMD=(
    docker build
    --build-arg "CUDA_VERSION=${CUDA_VERSION}"
    --build-arg "BRANCH_TYPE=local"
    -t "${IMAGE_TAG}"
    -f docker/Dockerfile
    .
  )
  echo "Building locally: ${IMAGE_TAG}"
  echo "  cuda: ${CUDA_VERSION}"
else
  CMD=(
    docker buildx build
    --platform "${PLATFORM}"
    --build-arg "CUDA_VERSION=${CUDA_VERSION}"
    --build-arg "BRANCH_TYPE=local"
    --cache-from "type=registry,ref=${CACHE_REF}"
  )
  # Only add --cache-to if not disabled (significantly speeds up export)
  # Using mode=min (default) instead of mode=max to only cache final image layers
  if [[ "$NO_CACHE_EXPORT" == "0" ]]; then
    CMD+=(--cache-to "type=registry,ref=${CACHE_REF}")
  fi
  CMD+=(
    -t "${IMAGE_TAG}"
    --push
    -f docker/Dockerfile
    .
  )
  echo "Building/pushing: ${IMAGE_TAG}"
  echo "  platform: ${PLATFORM}"
  echo "  cache:    ${CACHE_REF}"
  echo "  cuda:     ${CUDA_VERSION}"
  if [[ "$NO_CACHE_EXPORT" == "1" ]]; then
    echo "  cache-export: disabled (--no-cache-export)"
  fi
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN: %q ' "${CMD[@]}"
  echo
  exit 0
fi

"${CMD[@]}"
