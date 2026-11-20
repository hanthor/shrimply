# Builds the prebuilt CUDA cubins for the flatpak sandbox: just the toolchain
# (rustup + CUDA from NVIDIA's repo) needed to run `make cuda-artifacts`, none
# of the top-level Dockerfile's runtime-bundling. Built via
# `make cuda-artifacts-image`.
FROM fedora:44
WORKDIR /src

RUN dnf install -y curl git make patchelf rustup && dnf clean all

RUN rustup-init -y --default-toolchain none --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# deps-fedora expects the RPMFusion repos to exist.
RUN dnf install -y \
        "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm" \
    && dnf clean all

COPY Makefile ./
RUN make deps-fedora DNF="dnf -y"

# CUDA toolkit from NVIDIA's own repo (Fedora's repos don't ship it).
# Pinned, not bare `cuda-toolkit`: this nvcc compiles the .cubin files vendored
# into the flatpak, so the toolkit version is part of this build's shipped
# output and must not float with whatever NVIDIA published last. Keeping the
# version in the Dockerfile text also means flatpak.yml's
# hashFiles(...cuda-artifacts.Dockerfile) moves the `cuda-cubins` cache key
# whenever it is bumped, so stale cubins can't survive a toolkit change.
# To bump: change this, and delete crates/render-cuda/prebuilt/ to force a
# regeneration (build.rs copies a vendored cubin in preference to running nvcc).
ARG CUDA_TOOLKIT_VERSION=13-4
RUN . /etc/os-release && \
    curl -fsSL -o /etc/yum.repos.d/cuda-fedora.repo \
        "https://developer.download.nvidia.com/compute/cuda/repos/fedora${VERSION_ID}/x86_64/cuda-fedora${VERSION_ID}.repo" && \
    dnf install -y "cuda-toolkit-${CUDA_TOOLKIT_VERSION}" && \
    dnf clean all

COPY . .

RUN rustup show

# Git submodules (slang) might be missing from the build context.
RUN test -e external/slang/CMakeLists.txt || { \
        echo "Git submodules are missing from the build context." >&2; \
        echo "Run 'git submodule update --init --recursive' before 'docker build'." >&2; \
        exit 1; \
    }

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_TOOLKIT_PATH=/usr/local/cuda

# No NVIDIA driver in the container; stub libcuda.so.1 so nvcc's link resolves.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# CUDA_HOST_CXX=g++: the Makefile default (g++-15) isn't installed here, only
# plain g++.
CMD ["make", "cuda-artifacts", "CUDA_HOST_CXX=g++"]
