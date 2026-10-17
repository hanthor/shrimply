Effects and Animation
=====================

Animation
---------

Animatable properties support keyframes, continuous and discrete keyframe
graphs, and Rhai expressions. Use expressions when a value should be derived
from project state instead of authored as individual keyframes. See
:doc:`expressions` for the available values, functions, and examples.

Visual effects
--------------

Visual modifiers cover transforms, opacity, repetition, paths, text masks,
color correction, blur, distortion, stylization, keys, and masks. Shrimply
also includes model-backed segmentation and transparent-fill operations.

Modifiers are applied in ordered stages so 3D, vector, rasterization, and
raster operations can be composed without silently changing their data type.

Audio effects
-------------

Audio modifiers include gain, pan, pitch, denoise, equalization, filters,
noise gates, stereo width, tremolo, bit crushing, chorus, compression,
limiting, reverb, echo, distortion, voice change, and microphone-style
coloration.

3D scenes
---------

3D scene content can include shapes, text, imported objects, a ground plane,
point lights, and sun lights. Imported OBJ, GLB, and PLY content can be used in
the same scene pipeline.

3D rendering
------------

PBR scenes expose a **Denoiser** control in the inspector. It is enabled by
default and applies only to the PBR shading model; Toon and Depth scenes do not
use it.

The backend depends on the platform:

* NVIDIA/CUDA rendering uses the OptiX 9 AOV denoiser. Local development builds
  need the CUDA toolkit and the repository's ``external/optix-dev`` checkout;
  see :doc:`Development <../development>` for build prerequisites.
* macOS Metal rendering uses SVGF denoising. It cannot be combined with
  stochastic depth of field. Turn **Denoiser** off or set the camera's focus
  distance to ``0`` when that combination is needed.
