#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <cstdint>


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// Scene-construction randomness -- used while a scene is being assembled, never
// while it is being rendered. Sampling during a render draws from an RNG that is
// passed down the call chain instead (see core/rng.h).
//
// Scene construction runs once, on the main thread, before rendering begins, so
// a single generator is enough. It is seeded with a fixed value rather than from
// random_device: a scene built twice must be the same scene twice, or a render
// cannot be reproduced no matter how deterministic the sampling is.
inline constexpr uint32_t default_scene_seed = 1337u;

inline std::mt19937& scene_rng() {
    static std::mt19937 generator(default_scene_seed);
    return generator;
}

// Call before building a scene to select a different layout.
inline void seed_scene_rng(uint32_t seed) {
    scene_rng().seed(seed);
}

inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(scene_rng());
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return int(random_double(min, max+1));
}

// Common Headers

#include "raytracer/renderer/color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"


#endif