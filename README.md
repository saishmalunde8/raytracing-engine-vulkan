# Ray Tracing Engine — Vulkan (GPU)

A GPU implementation of a physically based path tracer, built on Vulkan compute.

This is the GPU half of a two-part project. The CPU engine is a complete,
importance-sampled path tracer with deterministic output; this repository takes that
renderer onto the GPU, where thousands of pixels are traced concurrently rather than
a handful of tiles across CPU threads.

> **Companion project** — the CPU engine lives at
> [raytracing-engine-cpu](https://github.com/saishmalunde8/raytracing-engine-cpu).
> That repository is canonical for all renderer code; this one carries a snapshot of
> it as a reference implementation.

---

## Current Status

**GPU compute path — working end to end.** A compute shader computes an 800×600 image
into a storage image every frame. Two pipeline barriers order that write against the
fragment shader's read in both directions — read-after-write within a frame, and
write-after-read across frames — and a three-vertex draw with no vertex buffer puts
the result on screen. The render resolution is fixed and independent of the window, so
resizing stretches the image rather than reallocating anything.

**The Vulkan machinery underneath is complete.** Instance and device creation with
portability flags, validation layers in debug builds, a combined graphics and compute
queue, swapchain selection and recreation on resize, render pass and framebuffers,
graphics and compute pipelines, descriptor sets, frames-in-flight synchronisation,
staged device-local uploads, image layout transitions, samplers, and push constants.

**Renderer port — not yet started.** What the compute shader computes is a gradient,
not a scene: there is no camera, no ray, no intersection and no material on the GPU.
The path tracer under `include/` and `src/` still runs unchanged, on the CPU.

This ordering is deliberate: the CPU renderer was brought up to date with importance
sampling *before* any GPU sampling code was written, so the port targets the current
algorithm rather than an obsolete one, and has a working reference to be validated
against.

Vertex buffers, index buffers, uniform buffers and a host-uploaded texture were each
built along the way and then deliberately removed, once a compute shader became the
image's source and a storage image replaced the uploaded one. They are in the commit
history rather than the working tree, which is where the reasoning for them lives too.

---

## Why Two Backends

The two implementations attack different limits.

Multithreading makes the renderer do the same work faster, and is bounded by core
count — roughly 3× on four cores. Importance sampling makes it do *less* work for the
same image, which on the CPU engine reached equal quality about 39× faster with no
new hardware. A GPU backend attacks the first limit again, far harder: thousands of
concurrent invocations instead of four.

Those gains multiply. The GPU port inherits an algorithm that already needs a fraction
of the samples, so the two improvements compound rather than compete.

---

## CPU Reference Implementation

`include/raytracer/`, `src/main.cpp` and `tools/` hold a snapshot of the CPU engine.
It exists here as a **correctness oracle**: render a scene on both backends and compare.

The exact upstream commit, the sync rules, and detailed notes on which parts of the CPU
code must not be translated literally to GPU are recorded in
[`REFERENCE_VERSION.md`](REFERENCE_VERSION.md).

![CPU reference render](docs/images/cpu-reference.png)

*Produced by the CPU reference implementation in this repository.*

### What carries over cleanly

The CPU engine was written in a way that happens to suit GPU execution:

- **Coordinate-derived RNG** — `pixel_sample_seed(i, j, sample)` derives randomness
  purely from a pixel's own coordinates, which is exactly what a shader invocation
  requires. A conventional per-thread sequential generator could not be ported at all.
- **Framebuffer-first output** — the renderer writes into a buffer and emits the image
  as a separate pass, matching how a compute shader writes to a storage image.
- **No mutable global state in the render path** — enforced by a build-time guard.
- **Independent, order-independent pixels** — no cross-tile dependencies.

### What has to change

Virtual dispatch, `shared_ptr`, recursion, the BVH's pointer-linked tree, per-bounce
heap allocation, `double` precision, and `std::mt19937` all need GPU-appropriate
replacements. Each is covered in `REFERENCE_VERSION.md`.

---

## Build & Run

### Vulkan backend

Requires the Vulkan SDK (with `VULKAN_SDK` set by its `setup-env.sh`) and GLFW.

```
cmake -B build
cmake --build build
./build/vulkan_backend
```

### CPU reference renderer

Built independently of the Vulkan target, so the two cannot break each other.

```
clang++ -std=c++17 -O2 -Iinclude src/main.cpp -o build/raytracer
./build/raytracer > output.ppm
```

### Sampler validation

Checks the direction samplers and probability densities against analytically known
results — useful for confirming a GPU sampler matches the reference before trusting a
full render.

```
clang++ -std=c++17 -O2 -Iinclude tools/sampling_check.cpp -o build/sampling_check
./build/sampling_check
```

---

## Repository Structure

```
backends/vulkan/ - Vulkan application and GPU backend
shaders/         - GLSL sources and compiled SPIR-V
include/         - CPU reference renderer (snapshot; see REFERENCE_VERSION.md)
src/             - CPU reference entry point and scenes
tools/           - Example scenes and sampler validation
assets/          - Runtime assets (e.g. textures)
docs/            - Documentation and render outputs
external/        - Third-party dependencies (stb_image)
```

---

## Learning Lineage

This project draws from the concepts and techniques presented in Peter Shirley’s
*Ray Tracing in One Weekend* series. The focus of this implementation is on deeply
engaging with the underlying rendering principles and organizing them into a
coherent, extensible system that can serve as a base for further exploration.
