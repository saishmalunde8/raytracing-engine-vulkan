# CPU Reference Implementation

The CPU renderer in this repository (`include/raytracer/`, `src/main.cpp`, `tools/`)
is a **snapshot**, copied from the CPU engine. It exists here as a correctness
oracle: render a scene on both backends and compare the results.

| | |
|---|---|
| **Upstream** | https://github.com/saishmalunde8/raytracing-engine-cpu |
| **Snapshot commit** | `1fdfc51c08dce3a5f8334a3a7f02f89eb530cd56` |
| **Snapshot date** | 2026-09-11 |

## Rules

**The CPU repository is canonical.** Renderer bug fixes and algorithm changes land
there first and are copied here afterwards — never the other way round. Anything
edited here directly will be lost at the next sync.

**Sync deliberately, not continuously.** Pull from upstream at real milestones, such
as before writing GPU sampling code or ahead of a correctness push. Update the commit
hash above whenever you do.

## What this snapshot contains

An importance-sampled path tracer: cosine-weighted material densities mixed with
direct light sampling, geometric sampling of spheres, quads and object lists,
orthonormal basis construction, and bitwise-reproducible output that is independent
of thread count.

`tools/sampling_check.cpp` validates the direction samplers and probability densities
against analytically known results — useful for confirming a GPU sampler matches the
reference before trusting a full render.

## Notes for the Vulkan port

The GPU implementation will not resemble this code structurally, and parts of it must
not be translated literally:

- **Virtual dispatch** — `hittable`, `material`, `texture` and `pdf` are polymorphic
  hierarchies. SPIR-V has no vtables; these become tagged unions or per-type buffers.
- **`shared_ptr`** — becomes indices into flat storage buffers.
- **Recursion** — `ray_color` recurses. This becomes an iterative loop carrying a
  throughput value.
- **Per-bounce heap allocation** — the mixture density in `camera.h` calls
  `make_shared` on every diffuse bounce. There is no heap on the GPU; these become
  stack values computed inline.
- **`double` precision** — the CPU path uses `double` throughout. GPUs are far slower
  in fp64; the port should use `float`.
- **`std::mt19937`** — roughly 2.5 KB of state per generator, fatal per-thread on a
  GPU. Replace with a small hash-based generator. The seeding scheme itself carries
  over unchanged: `pixel_sample_seed(i, j, sample)` already derives randomness purely
  from coordinates, which is exactly what a GPU shader requires.

Bitwise parity between the CPU and GPU backends is **not** an achievable goal —
floating-point rounding and transcendental precision differ across implementations.
What the shared seeding scheme does give is the same sample sequence, so both backends
converge toward the same image and can be compared within a tolerance.
