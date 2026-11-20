Development
===========

Requirements
------------

The current development setup targets Fedora and uses the Rust toolchain in
``rust-toolchain.toml``. Install the native dependencies with:

.. code-block:: console

   $ make deps-fedora

Initialize the Slang submodule before building. Slang compiles the compositor
shaders to CUDA, and ``nvcc`` packages the current ``sm_86`` cubins.

Build and check
---------------

Use the Makefile for repository operations:

.. code-block:: console

   $ make dev
   $ make build
   $ make check
   $ make test
   $ make release

To build and run the optional Qt 6 launcher instead of the GTK launcher, use
``make dev-qt``. This produces the separate ``shrimply-qt`` development binary;
it does not replace the normal launcher or installation.
``make qt-build`` performs that debug build without launching it and writes the
binary to ``target/debug/shrimply-qt``.

``make check`` verifies native dependencies and CUDA artifacts, formatting,
source size, the selected Rust binaries, Clippy, the server and Manim Python
code, and this documentation site. The development launcher writes its log to
``target/shrimply-dev.log``.

Build the documentation on its own with:

.. code-block:: console

   $ make docs


Docker Build Environment [Experimental]
---------------------------------------

You can build Shrimply using the provided Dockerfile. 
This will generate the necesary binaries to run Shrimply on your machine.

Remember to pull the Git Submodules before building Shrimply

.. code-block:: console

   $ docker buildx build --target export --output type=local,dest=dist/stage .

Python environments
-------------------

Python dependencies are managed with uv. The documentation, local compute
server, and Manim worker each have their own ``pyproject.toml`` and committed
``uv.lock``. Use their Makefile targets or ``uv run --project`` from the
repository root; do not install their dependencies globally.

Server scripts
--------------

``server/scripts`` has two standalone developer tools, run the same way as
``make dev-server`` (``uv run --project server --locked <path> [args]``):

``benchmark_index_tts.py``
   Benchmarks the local IndexTTS Stable and Beta runtimes across model
   version, precision, and sampling mode. See ``--help`` for the full option
   set (voice, language, run count, kernel profiling, output directory).

.. code-block:: console

   $ uv run --project server --locked server/scripts/benchmark_index_tts.py --help

``validate_video_generation.py``
   Exercises the video-generation server API end to end against a running
   ``make dev-server`` instance, for one of three cases: ``base-t2va``,
   ``sketch-fl2va``, or ``base-ref2va``.

.. code-block:: console

   $ uv run --project server --locked server/scripts/validate_video_generation.py \
       base-t2va --output-directory /tmp/out --image ref.png --reference-video ref.mp4

Repository layout
-----------------

``crates/apps``
   Launcher and editor applications.

``crates/timeline`` and ``crates/project``
   Timeline behavior, project state, storage, and project importers.

``crates/preview``, ``crates/video``, and ``crates/export``
   Playback, decoding, compositing, visual effects, and export.

``crates/audio``
   Audio rendering, modifiers, transcription, text-to-speech, and lip sync.

``crates/3d``, ``crates/paint``, and ``crates/layered-image``
   Specialized content and rendering pipelines.

``crates/math`` and ``crates/core``
   Shared math and core data types.

``crates/mcp`` and ``crates/server-client``
   Live editor automation and compute-server communication.

``server``
   The uv-managed local AI compute server.

Contributing
------------

Read the repository's `contribution terms
<https://github.com/soirihiroka/shrimply/blob/main/CONTRIBUTING.md>`__ before
submitting a change.
