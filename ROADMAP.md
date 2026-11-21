# Roadmap

Shrimply is pre-alpha. This roadmap states direction, not dates; items are
tracked in GitHub issues and may move.

## Now: portable foundation (epic #2)

Shrimply currently needs an NVIDIA GPU and driver to start. The decision
recorded in #132 is to **feature-gate CUDA behind an optional `cuda` cargo
feature**, enabled by default only where the NVIDIA build needs it. A runtime
hardware-abstraction layer is not part of that step.

1. Feature-gate CUDA out of the application crates (#11).
2. Load the CUDA driver with `dlopen` instead of linking `-lcuda` (#4).
3. Make the OptiX denoiser optional (#14).
4. Flatpak bundle that runs without the NVIDIA GL extension (#15), which is the
   check that a GPU-less machine can launch the app.

## Next: a second backend

Builds on the gate above; each item is a separate change.

- GPU backend abstraction (#5) and portable device memory (#6).
- Vulkan compositor, the SPIR-V sibling of `video-cuda` (#7).
- Portable decode (#8) and encode (#9).
- Preview interop without CUDA-GL sharing (#10).
- Runtime backend selection and capability reporting (#12).
- Portable stabilization (#13).

Features with no portable equivalent (OptiX denoising, NVIDIA Optical Flow,
NVENC) are absent on non-NVIDIA hardware, not reimplemented.

## Alongside: trust and contributor health

- CI that runs `cargo check`, `clippy` and `cargo test` on pull requests (#27)
  and Python tests for the compute server (#61).
- Timeouts, integrity checks and loopback-only binding for the compute server
  and model downloads (#17, #58, #63, #69, #71, #72, #76, #77).
- Build and architecture documentation for Linux and macOS (#33, #41, #82).

## Later

- Release versioning once the backend story is settled.
- Revisit multi-platform performance goals (#134) after a second backend
  exists to measure against.
