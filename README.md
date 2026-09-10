# Ray Tracing Engine — Vulkan (GPU)

A GPU implementation of a physically based path tracer, built on Vulkan compute.

## This branch

`main` holds the **CPU reference engine** — a complete, importance-sampled path tracer
with deterministic output. It is kept here as a known-good baseline and as a
correctness oracle: render a scene on both backends and compare the results.

**The GPU work happens on [`feature/vulkan-backend`](../../tree/feature/vulkan-backend).**
That branch carries the Vulkan application, the shaders, and — once the port begins —
the changes needed to make the renderer GPU-portable. It will be merged here when the
port is complete, at which point this README is replaced.

## Where things live

| | |
|---|---|
| **GPU work in progress** | [`feature/vulkan-backend`](../../tree/feature/vulkan-backend) |
| **CPU engine — full documentation, renders, benchmarks** | [raytracing-engine-cpu](https://github.com/saishmalunde8/raytracing-engine-cpu) |
| **Snapshot details and porting notes** | [`REFERENCE_VERSION.md`](REFERENCE_VERSION.md) |

The CPU repository is canonical for all renderer code. Changes land there first and
are copied here as snapshots — anything edited directly on this branch will be lost at
the next sync.

## Build

```
clang++ -std=c++17 -O2 -Iinclude src/main.cpp -o build/raytracer
./build/raytracer > output.ppm
```

Sampler validation, useful for confirming a GPU sampler matches the reference:

```
clang++ -std=c++17 -O2 -Iinclude tools/sampling_check.cpp -o build/sampling_check
./build/sampling_check
```

## Learning Lineage

This project draws from the concepts and techniques presented in Peter Shirley’s
*Ray Tracing in One Weekend* series. The focus of this implementation is on deeply
engaging with the underlying rendering principles and organizing them into a
coherent, extensible system that can serve as a base for further exploration.
