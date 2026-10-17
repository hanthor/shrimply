Server configuration
====================

The compute server reads its runtime configuration from environment variables.
This page describes values read by ``server/src/env.py``. Settings below apply
when the server is started directly; the checked-in Compose file uses mounted
``.docker`` directories for container persistence and passes the Git hash as a
build argument.

Prerequisites
-------------

The server project requires Python ``>=3.14,<3.15`` and ``uv``. Install ``uv``
using `Astral's official instructions
<https://docs.astral.sh/uv/getting-started/installation/>`__. Run commands in
this page from the ``server`` directory. The locked project environment starts
with:

.. code-block:: console

   $ uv run --locked src/main.py

Bind and access
---------------

The editor defaults to ``http://127.0.0.1:8787`` when connecting. The server
itself defaults to ``0.0.0.0:8787``, which listens on every interface. Use a
loopback bind when the server must remain local:

.. code-block:: console

   $ SHRIMPLY_SERVER_HOST=127.0.0.1 uv run --locked src/main.py

``SHRIMPLY_SERVER_SHARE=1`` additionally requests a temporary public
``gradio.live`` URL. The URL can invoke every compute endpoint. Do not enable
sharing on an untrusted network or treat it as authentication; stop the
server to remove the temporary URL.

``SHRIMPLY_SERVER_HOST``
   Address passed to the server listener. Default: ``0.0.0.0``. Set
   ``127.0.0.1`` for local-only access.

``SHRIMPLY_SERVER_PORT``
   Listener port. Default: ``8787``. The value must parse as a positive
   integer; invalid values stop startup.

``SHRIMPLY_SERVER_SHARE``
   Whether to request a temporary ``gradio.live`` URL. Default: ``0``. The
   only accepted values are ``0`` and ``1``.

``SHRIMPLY_SERVER_GIT_HASH``
   Optional version string reported by the server status endpoint. If unset,
   the server resolves the current Git commit when it starts. Container builds
   set this through the Docker build argument.

Worker lifecycle
----------------

``SHRIMPLY_MODEL_IDLE_TTL_SECONDS``
   Seconds before an idle model worker is unloaded. Default: ``600``. The
   value must be a positive integer. Lower values release resources sooner but
   make the next request reload the model more often.

Pneuma voice conversion
-----------------------

``SHRIMPLY_PNEUMA_MODEL_DIR``
   Directory containing Pneuma ``.safetensors`` or legacy ``.pth`` voice
   models. Default: ``models`` relative to the server's working directory.
   The :doc:`voice-conversion` page describes the local and Compose paths.

``SHRIMPLY_PNEUMA_DEVICE``
   PyTorch device string used by Pneuma inference. Default: ``cpu``. Use a
   device available to the process, such as ``cuda`` or ``cuda:0``; requesting
   an unavailable accelerator fails when Pneuma initializes.

Video-generation storage
------------------------

The following paths are used by video-generation workers. Relative defaults
are based on the server's current working directory. Set them to durable,
large-volume locations when model conversion, offload, or generated media
would exceed the working directory's available space.

``SHRIMPLY_VIDEO_GENERATION_CACHE``
   Temporary media and request staging root for video-generation workflows.
   Default: ``video-generation-cache``.

``MINIMAX_H3_CACHE``
   Optional model-download cache passed to the Hugging Face model loader for
   MiniMax H3. When unset, the loader uses its normal cache location.

``MINIMAX_H3_QUANTIZED_CACHE``
   Persistent root for converted MiniMax H3 quantized components. Default:
   ``quantized-cache``.

``MINIMAX_H3_DISK_OFFLOAD_CACHE``
   Root for MiniMax H3 components staged for disk-backed offload. Default:
   ``disk-offload-cache``.

``MINIMAX_H3_LORA_CACHE``
   Root for converted MiniMax H3 LoRA adapters. Default: ``lora-cache``.

``MINIMAX_H3_DECODE_OFFLOAD``
   Decoder placement for MiniMax H3. Default: ``auto``; the server chooses
   ``gpu`` when the available memory threshold allows it and otherwise uses
   ``disk``. Explicit values are ``auto``, ``gpu``, and ``disk``.

For example, select disk-backed H3 decoding when GPU memory is constrained:

.. code-block:: console

   $ MINIMAX_H3_DECODE_OFFLOAD=disk uv run --locked src/main.py

Compose paths
-------------

The checked-in ``server/compose.yaml`` mounts these persistent locations:

* ``server/.docker/cache`` for the ``uv`` cache;
* ``server/.docker/huggingface`` for Hugging Face downloads;
* ``server/.docker/pneuma/models`` for Pneuma models; and
* ``server/.docker/python`` for the managed Python installation.

Those mounts are separate from the direct-run defaults above. Keep the
container's mounted directories on storage with enough capacity for the model
families you enable.
