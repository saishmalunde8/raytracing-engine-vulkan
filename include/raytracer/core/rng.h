#ifndef RNG_H
#define RNG_H

#include <random>
#include <cstdint>
#include "vec3.h"

// Every random decision taken while rendering draws from an RNG instance that
// is passed down the call chain. Deterministic mode seeds one instance per
// pixel-sample; the default mode hands out a per-thread instance. There is no
// global sampling path, so a render cannot silently become irreproducible by
// reaching for ambient randomness.
struct RNG {
    std::mt19937 engine;
    std::uniform_real_distribution<double> dist;

    explicit RNG(uint32_t seed)
        : engine(seed), dist(0.0, 1.0) {}

    double next_double() {
        return dist(engine);
    }

    vec3 random_unit_vector() {
    auto a = next_double() * 2 * 3.1415926535897932385;
    auto z = next_double() * 2.0 - 1.0;
    auto r = std::sqrt(1 - z*z);
    return vec3(r*std::cos(a), r*std::sin(a), z);
    }

    // A direction in the +Z hemisphere, distributed with density cos(theta)/pi.
    // Directions near the pole come up more often than directions near the
    // horizon, which is where a diffuse surface actually sends most of its
    // light -- so samples get spent where they matter. Returned in basis
    // coordinates; rotate it onto a surface with onb::transform().
    //
    // Exactly two draws, no rejection loop, so the stream advances predictably.
    vec3 random_cosine_direction() {
    auto r1 = next_double();
    auto r2 = next_double();

    auto phi = 2 * 3.1415926535897932385 * r1;
    auto x = std::cos(phi) * std::sqrt(r2);
    auto y = std::sin(phi) * std::sqrt(r2);
    auto z = std::sqrt(1 - r2);

    return vec3(x, y, z);
    }
};

inline uint32_t pixel_sample_seed(int i, int j, int sample) {
    uint32_t seed = 0;
    seed ^= uint32_t(i) * 1973u;
    seed ^= uint32_t(j) * 9277u;
    seed ^= uint32_t(sample) * 26699u;
    seed += 1u;
    return seed;
}

// Sampling source for non-deterministic renders. One instance per worker
// thread, so tiles never share a stream and never contend for one.
inline RNG& thread_rng() {
    thread_local RNG rng(std::random_device{}());
    return rng;
}

#endif
