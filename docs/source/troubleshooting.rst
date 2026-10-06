Troubleshooting
===============

Building and running Shrimply can be complex due to its dependencies (CUDA,
FFmpeg, Docker, submodules). This guide covers common issues and solutions.

.. contents::
   :local:
   :depth: 2

Build Failures
--------------

**"CUDA not found" or "nvcc not found"**

Shrimply requires NVIDIA CUDA Toolkit and a compatible GPU driver. 

- Verify CUDA is installed: ``nvcc --version``
- Check GPU driver: ``nvidia-smi``
- Install CUDA: Follow `NVIDIA's installation guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>`__
- Add CUDA to PATH if needed:
  
  .. code-block:: console
  
     export PATH=/usr/local/cuda/bin:$PATH
     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

- Verify compute capability matches: Shrimply targets ``sm_86`` (RTX 30-series, RTX A series). Older GPUs may need code changes.

**"Slang submodule not initialized"**

Slang must be initialized before building.

.. code-block:: console

   git submodule update --init --recursive

If this fails with permission errors, check your GitHub SSH key setup and ensure you can clone submodules.

**"FFmpeg not found" or linking errors**

Shrimply requires FFmpeg development libraries. Install per your platform:

- **Fedora**: Included in ``make deps-fedora``
- **Ubuntu/Debian**: ``sudo apt install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libavresample-dev``
- **Arch**: ``sudo pacman -S ffmpeg``
- **macOS**: ``brew install ffmpeg``

If you install FFmpeg manually, ensure ``pkg-config`` can find it:

.. code-block:: console

   pkg-config --cflags --libs libavformat

**"Out of memory" during compilation**

Large Rust projects (including FFmpeg) can exhaust RAM. Solutions:

- Increase swap: ``sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile``
- Limit parallel jobs: ``export CARGO_BUILD_JOBS=1`` (trades speed for lower memory)
- Build in release mode first (uses more RAM but faster): ``make release``
- Close other applications

**"Linking failed" or undefined reference errors**

Ensure all dependencies are installed and in the library path:

.. code-block:: console

   make check    # Verifies all dependencies
   pkg-config --list-all | grep -E "cuda|ffmpeg|gtk"  # Verify packages are found

If packages are installed but not found:

- Set ``PKG_CONFIG_PATH``: ``export PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/local/lib/pkgconfig``
- Rebuild: ``cargo clean && make dev``

**"GNOME runtime not found" (Flatpak build)**

The Flatpak build needs GNOME SDK extensions.

.. code-block:: console

   make dist    # Installs everything

Or manually:

.. code-block:: console

   flatpak install flathub org.gnome.Platform//46 org.gnome.Sdk//46
   flatpak install flathub org.freedesktop.Sdk.Extension.rust-stable

**Docker build fails**

If using ``docker buildx build --target export --output type=local,dest=dist/stage .``:

- Ensure Docker is running: ``docker ps``
- Increase Docker memory if build hangs: Docker Desktop → Preferences → Resources → Memory
- Rebuild without cache: ``docker buildx build --no-cache --target export ...``
- Check disk space: ``docker system df``

Dependency Issues
-----------------

**"Submodule sync failed" or stale submodules**

If submodules are out of sync:

.. code-block:: console

   git submodule sync --recursive
   git submodule update --init --recursive

**Python environment errors (uv not found)**

Shrimply uses ``uv`` for Python environment management. Install it:

.. code-block:: console

   curl -LsSf https://astral.sh/uv/install.sh | sh

Then update your shell PATH or use ``~/.cargo/bin/uv`` directly.

**"Port already in use" (compute server)**

The local compute server defaults to port 8000. If it's already in use:

.. code-block:: console

   lsof -i :8000    # Find what's using the port
   # Either stop that process or configure Shrimply to use a different port

Runtime Issues
--------------

**"CUDA out of memory" when rendering**

Shrimply caches intermediate frames in VRAM. If you hit OOM:

- Reduce timeline resolution or preview quality
- Export at lower resolution first
- Close other GPU-using applications
- Use a different GPU if available: Set ``CUDA_VISIBLE_DEVICES=1`` for GPU 1, etc.

**"FFmpeg frame decode errors"**

If playback shows corruption or errors:

- Verify the media file is not corrupted: ``ffprobe -v error input.mp4``
- Try importing the media again
- Export the media with FFmpeg first: ``ffmpeg -i input.mp4 -c copy output.mp4`` (remux without re-encoding)

**"GTK warnings" or theme issues**

GTK 4 theming issues are usually cosmetic but can be verbose in logs.

- Set GSK renderer: ``GSK_RENDERER=gl shrimply`` (or ``cairo``, ``ngl``)
- Check GTK theme: ``gsettings get org.gnome.desktop.interface gtk-theme``

**"Manim scene not rendering"**

If Manim scenes fail to render:

- Check Python syntax: ``uv run --project server -- python3 -m py_compile scene.py``
- Verify dependencies: ``uv run --project server pip list | grep manim``
- Check scene class name matches filename convention
- Enable verbose logging: ``MANIM_LOG_LEVEL=DEBUG`` in the Shrimply environment

Flatpak-Specific Issues
-----------------------

**"Permission denied" when accessing files**

Flatpak sandboxes file access. Grant access to additional directories:

.. code-block:: console

   flatpak override --filesystem=/path/to/directory dev.shrimply.Shrimply

**"Cannot find GPU inside Flatpak"**

CUDA inside Flatpak requires additional setup. The development environment (``make dev``) bypasses the sandbox.

Reporting Issues
----------------

If you hit a problem not covered here:

1. Check the `GitHub issues <https://github.com/soirihiroka/shrimply/issues>`__
2. Collect logs: ``target/shrimply-dev.log`` (development) or journalctl (system)
3. Include:
   - Your OS and GPU
   - Output of ``make check``
   - Full error message and stack trace
   - Steps to reproduce

Platform-Specific Notes
-----------------------

**Fedora**

The Makefile targets Fedora. ``make deps-fedora`` installs most dependencies. CUDA requires additional NVIDIA repos.

**Ubuntu/Debian**

Most dependencies are available via apt. You may need to add the NVIDIA CUDA repository for ``nvcc``.

**Arch Linux**

Use pacman for most dependencies. CUDA is in AUR (``cuda``, ``cuda-tools``).

**macOS**

macOS support is experimental. Install dependencies via Homebrew (``brew install rustup ffmpeg``). CUDA/GPU support on macOS requires special handling (ROCm for AMD, or external GPU for NVIDIA).

**Windows**

Windows builds are not currently supported. Use Windows Subsystem for Linux (WSL) or a VM.

Reference
---------

- `make check` — Verifies all dependencies, formatting, and project integrity
- `target/shrimply-dev.log` — Development build log
- `docs/source/development.rst` — Full development setup guide
- `CONTRIBUTING.md` — Contribution terms and workflows
