#!/usr/bin/env bash
# Build the capture renderer of the retained NovPhy frames without root:
# Mesa 26.1.2 llvmpipe on LLVM 22.1.6 (the retained player logs report
# "llvmpipe (LLVM 22.1.6, 256 bits)" / "Mesa 26.1.2-arch1.1"). The server's
# system Mesa 25.2.8 / LLVM 20.1.2 renders rotated sprite edges differently.
#
# Output: ~/.cache/novphy-mesa/mesa-26.1.2-llvm22.1.6/lib/{libGLX_mesa.so.0,dri/swrast_dri.so,...}
# Use (capture_pipeline_v2 limits["engine_environment"]):
#   LD_LIBRARY_PATH=$PREFIX/lib LIBGL_DRIVERS_PATH=$PREFIX/lib/dri __GLX_VENDOR_LIBRARY_NAME=mesa
set -euo pipefail
ROOT="${NOVPHY_MESA_ROOT:-$HOME/.cache/novphy-mesa}"
ENV="$ROOT/env"
PREFIX="$ROOT/mesa-26.1.2-llvm22.1.6"
SOURCE_SHA256=bac2bca9121897a2b8162e79636b50ac998fca799c8e6cf914edd85962babdf0

mkdir -p "$ROOT"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if [[ ! -d "$ENV" ]]; then
  conda create -y -q -p "$ENV" -c conda-forge --override-channels python=3.11 llvmdev=22.1.6 llvm=22.1.6 \
    libllvm22=22.1.6 meson ninja pkg-config mako pyyaml packaging bison flex libdrm zlib zstd expat \
    xorg-libx11 xorg-libxext xorg-libxfixes xorg-libxxf86vm xorg-libxrandr xorg-libxshmfence libxcb \
    xorg-xorgproto libglvnd-devel gcc_linux-64 gxx_linux-64
fi
[[ -f mesa-26.1.2.tar.xz ]] || curl -sSLO https://archive.mesa3d.org/mesa-26.1.2.tar.xz
echo "$SOURCE_SHA256  mesa-26.1.2.tar.xz" | sha256sum -c -
rm -rf mesa-26.1.2 && tar -xf mesa-26.1.2.tar.xz
conda activate "$ENV"
cd mesa-26.1.2
export CFLAGS="-O2 -march=x86-64 -mtune=generic -pipe" CXXFLAGS="-O2 -march=x86-64 -mtune=generic -pipe"
export LDFLAGS="-Wl,-rpath,$ENV/lib -Wl,-rpath-link,$ENV/lib"
meson setup build --prefix="$PREFIX" --libdir=lib -Dbuildtype=plain -Dplatforms=x11 \
  -Dgallium-drivers=llvmpipe -Dvulkan-drivers= -Dglx=dri -Dglvnd=enabled -Degl=disabled -Dgbm=disabled \
  -Dgles1=disabled -Dgles2=disabled -Dllvm=enabled -Dshared-llvm=enabled -Dvalgrind=disabled \
  -Dlibunwind=disabled -Dzstd=enabled -Dbuild-tests=false -Dgallium-va=disabled -Dgallium-extra-hud=false \
  -Dlmsensors=disabled -Dgallium-rusticl=false -Dmicrosoft-clc=disabled -Dxlib-lease=disabled
ninja -C build -j "$(nproc)"
ninja -C build install
echo "installed $PREFIX"
