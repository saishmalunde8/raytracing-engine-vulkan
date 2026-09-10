// Validation for the direction samplers in core/. Standalone -- not part of the
// renderer build, and it links nothing from src/.
//
//   clang++ -std=c++17 -O2 -Iinclude tools/sampling_check.cpp -o build/sampling_check
//   ./build/sampling_check
//
// A direction sampler that is subtly wrong still produces a plausible-looking
// image, just one that is quietly lit incorrectly, and tracking that back from a
// finished render is painful. Each check below compares against a value that can
// be worked out on paper, so the sampler is measured rather than eyeballed.

#include "raytracer/core/rtweekend.h"
#include "raytracer/core/onb.h"
#include "raytracer/core/rng.h"
#include "raytracer/sampling/pdf.h"
#include "raytracer/materials/material.h"
#include "raytracer/geometry/sphere.h"
#include "raytracer/geometry/quad.h"
#include "raytracer/geometry/hittable_list.h"

#include <iomanip>
#include <string>

static int failures = 0;

static void check(const std::string& name, double got, double expected, double tol) {
    bool ok = std::fabs(got - expected) <= tol;
    if (!ok) failures++;
    std::cout << (ok ? "  pass  " : "  FAIL  ")
              << std::left << std::setw(46) << name
              << "got " << std::fixed << std::setprecision(6) << got
              << "   expected " << expected
              << "   tol " << tol << "\n";
}

static void check_true(const std::string& name, bool ok) {
    if (!ok) failures++;
    std::cout << (ok ? "  pass  " : "  FAIL  ") << name << "\n";
}

int main() {
    std::cout << "Direction sampler validation\n";
    std::cout << "----------------------------------------------------------------\n";

    const int N = 1000000;

    // 1. Cosine-weighted sampling, checked by integrating a function whose value
    //    over the hemisphere is known exactly.
    //
    //      integral of cos^3(theta) dw  over the hemisphere  =  pi/2
    //
    //    Sampling with density p = cos(theta)/pi, each sample contributes
    //    f/p = pi*cos^2(theta), so the mean of that must land on pi/2.
    //    A sampler with the wrong exponent or a missing square root misses this.
    {
        RNG rng(12345u);
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            double cos_theta = rng.random_cosine_direction().z();
            sum += pi * cos_theta * cos_theta;
        }
        check("cosine density integrates cos^3 to pi/2", sum / N, pi / 2, 0.005);
    }

    // 2. Every generated direction must be a unit vector in the +Z hemisphere.
    {
        RNG rng(999u);
        bool hemisphere = true, unit_length = true;
        for (int i = 0; i < N; i++) {
            vec3 d = rng.random_cosine_direction();
            if (d.z() < 0.0) hemisphere = false;
            if (std::fabs(d.length() - 1.0) > 1e-9) unit_length = false;
        }
        check_true("all cosine directions lie in the +Z hemisphere", hemisphere);
        check_true("all cosine directions are unit length", unit_length);
    }

    // 3. The basis must be orthonormal for any normal it is handed, including
    //    normals aligned with an axis, where a careless seed vector for the
    //    cross product collapses to zero.
    {
        RNG rng(7u);
        bool orthonormal = true, w_matches_normal = true;

        vec3 normals[] = {
            vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0),
            vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1),
            unit_vector(vec3(0.9, 0.1, 0.05))   // close to the x axis
        };

        auto verify = [&](const vec3& n) {
            onb basis(n);
            if (std::fabs(basis.u().length() - 1) > 1e-9) orthonormal = false;
            if (std::fabs(basis.v().length() - 1) > 1e-9) orthonormal = false;
            if (std::fabs(basis.w().length() - 1) > 1e-9) orthonormal = false;
            if (std::fabs(dot(basis.u(), basis.v())) > 1e-9) orthonormal = false;
            if (std::fabs(dot(basis.u(), basis.w())) > 1e-9) orthonormal = false;
            if (std::fabs(dot(basis.v(), basis.w())) > 1e-9) orthonormal = false;
            if (std::fabs(dot(basis.w(), unit_vector(n)) - 1) > 1e-9) w_matches_normal = false;
        };

        for (const auto& n : normals) verify(n);
        for (int i = 0; i < 10000; i++) verify(rng.random_unit_vector());

        check_true("basis is orthonormal for axis-aligned and random normals", orthonormal);
        check_true("basis w() points along the normal it was built from", w_matches_normal);
    }

    // 4. A cosine direction rotated onto a surface must end up on the outward
    //    side of that surface. If transform() were wrong, light would scatter
    //    into the geometry and surfaces would render far too dark.
    {
        RNG rng(2024u);
        bool correct_side = true;
        for (int i = 0; i < 100000; i++) {
            vec3 n = rng.random_unit_vector();
            onb basis(n);
            vec3 scattered = basis.transform(rng.random_cosine_direction());
            if (dot(scattered, n) < -1e-9) correct_side = false;
        }
        check_true("transformed directions stay on the normal's side", correct_side);
    }

    // 5. A density is only correct if generate() and value() agree with each
    //    other. Integrating a known function while sampling from the density and
    //    dividing by that same density's reported likelihood exercises both
    //    halves at once -- if either is wrong, the estimate misses.
    //
    //      integral of cos^3(theta) dw  over the hemisphere  =  pi/2
    //
    //    The normal here is deliberately not axis-aligned, so the basis
    //    transform inside generate() is under test too.
    {
        RNG rng(555u);
        vec3 n = unit_vector(vec3(0.3, 0.8, -0.5));
        cosine_pdf p(n);

        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            vec3 d = p.generate(rng);
            double c = dot(unit_vector(d), n);
            sum += (c * c * c) / p.value(d);
        }
        check("cosine_pdf integrates cos^3 to pi/2", sum / N, pi / 2, 0.005);
    }

    // 6. Same idea for the uniform density, but over the whole sphere rather
    //    than a hemisphere:
    //
    //      integral of cos^2(theta) dw  over the sphere  =  4*pi/3
    {
        RNG rng(4242u);
        sphere_pdf p;
        vec3 axis = unit_vector(vec3(-0.2, 0.6, 0.75));

        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            vec3 d = p.generate(rng);
            double c = dot(unit_vector(d), axis);
            sum += (c * c) / p.value(d);
        }
        check("sphere_pdf integrates cos^2 to 4pi/3", sum / N, 4 * pi / 3, 0.02);
    }

    // 7. A density must never report a negative likelihood, and must never
    //    report zero for a direction it generated itself -- the integrator
    //    divides by this value.
    {
        RNG rng(8080u);
        cosine_pdf cp(unit_vector(vec3(0.1, -0.9, 0.4)));
        sphere_pdf sp;
        bool positive_for_own_samples = true, never_negative = true;

        for (int i = 0; i < 100000; i++) {
            if (cp.value(cp.generate(rng)) <= 0.0) positive_for_own_samples = false;
            if (sp.value(sp.generate(rng)) <= 0.0) positive_for_own_samples = false;
            if (cp.value(rng.random_unit_vector()) < 0.0) never_negative = false;
        }
        check_true("densities are positive for directions they generate", positive_for_own_samples);
        check_true("densities never report a negative likelihood", never_negative);
    }

    // 8. Sampling through the virtual interface must stay reproducible -- a
    //    density that cached state between calls would break deterministic mode
    //    in a way no single-sample test would reveal.
    {
        vec3 n = unit_vector(vec3(1, 2, 3));
        cosine_pdf p(n);
        RNG a(31337u), b(31337u);

        bool identical = true;
        for (int i = 0; i < 10000; i++) {
            if ((p.generate(a) - p.generate(b)).length() != 0.0) identical = false;
        }
        check_true("equal seeds give equal sequences through the pdf interface", identical);
    }

    // ---- geometric sampling -------------------------------------------------
    //
    // A shape's sampler is right when 1/pdf_value averaged over its own samples
    // recovers the solid angle that shape covers from the sampling point. That
    // couples generate and evaluate the same way the density tests above do.

    auto white = make_shared<lambertian>(color(1,1,1));

    // 9. A sphere covers a cone, and that cone's solid angle is exact on paper:
    //        2*pi*(1 - sqrt(1 - r^2/d^2))
    {
        RNG rng(606u);
        point3 origin(0, 0, 0);
        sphere s(point3(0, 0, 10), 2.0, white);

        double expected = 2 * pi * (1 - std::sqrt(1 - (2.0*2.0)/(10.0*10.0)));

        double sum = 0.0;
        int misses = 0;
        for (int i = 0; i < N; i++) {
            vec3 d = s.random(origin, rng);
            double p = s.pdf_value(origin, d);
            if (p > 0) sum += 1.0 / p; else misses++;
        }
        check("sphere sampling recovers its solid angle", sum / N, expected, 0.001);
        check_true("sphere samples always hit the sphere", misses * 1000 < N);
    }

    // 10. A rectangle's solid angle has no tidy closed form, so it is measured a
    //     second, completely independent way: fire uniformly distributed
    //     directions and count how many land on the quad. The two estimates share
    //     no code, so agreement is meaningful.
    {
        point3 origin(0, 0, 0);
        quad q(point3(-1, -1, 2), vec3(2, 0, 0), vec3(0, 2, 0), white);

        RNG rng_a(707u);
        double sum = 0.0;
        int misses = 0;
        for (int i = 0; i < N; i++) {
            vec3 d = q.random(origin, rng_a);
            double p = q.pdf_value(origin, d);
            if (p > 0) sum += 1.0 / p; else misses++;
        }
        double via_pdf = sum / N;

        RNG rng_b(808u);
        int hits = 0;
        for (int i = 0; i < N; i++) {
            hit_record rec;
            if (q.hit(ray(origin, rng_b.random_unit_vector()), interval(0.001, infinity), rec))
                hits++;
        }
        double via_brute_force = 4 * pi * double(hits) / N;

        check("quad sampling matches brute-force solid angle", via_pdf, via_brute_force, 0.02);
        check_true("quad samples always hit the quad", misses * 1000 < N);
    }

    // 11. A list's density is the average of its members'. Two spheres placed at
    //     right angles cover disjoint cones, so the list must recover the sum of
    //     their individual solid angles.
    {
        RNG rng(909u);
        point3 origin(0, 0, 0);

        hittable_list lights;
        lights.add(make_shared<sphere>(point3(0, 0, 10), 2.0, white));
        lights.add(make_shared<sphere>(point3(0, 10, 0), 2.0, white));

        double one = 2 * pi * (1 - std::sqrt(1 - (2.0*2.0)/(10.0*10.0)));

        double sum = 0.0;
        int misses = 0;
        for (int i = 0; i < N; i++) {
            vec3 d = lights.random(origin, rng);
            double p = lights.pdf_value(origin, d);
            if (p > 0) sum += 1.0 / p; else misses++;
        }
        check("list sampling recovers both members' solid angle", sum / N, 2 * one, 0.002);
        check_true("list samples always hit a member", misses * 1000 < N);
    }

    // 12. An empty light list is a normal state -- four scenes have no emitters.
    //     It must report "cannot be sampled" rather than divide by zero or index
    //     off the end of an empty vector.
    {
        RNG rng(1010u);
        hittable_list empty;
        bool safe = (empty.pdf_value(point3(0,0,0), vec3(0,0,1)) == 0.0);
        vec3 d = empty.random(point3(0,0,0), rng);   // must not crash
        check_true("empty list reports zero density and does not crash", safe && d.length() > 0);
    }

    // 13. Wrapper and acceleration types keep the base-class defaults, which is
    //     what lets them compile untouched -- and is exactly why they must never
    //     be used as lights. Pinning the behaviour here so the rule is visible.
    {
        RNG rng(1111u);
        auto inner = make_shared<sphere>(point3(0, 0, 10), 2.0, white);
        translate moved(inner, vec3(1, 0, 0));
        rotate_y turned(inner, 30);

        bool defaults_hold =
            moved.pdf_value(point3(0,0,0), vec3(0,0,1)) == 0.0 &&
            turned.pdf_value(point3(0,0,0), vec3(0,0,1)) == 0.0;

        check_true("wrapped shapes report zero density (do not use as lights)", defaults_hold);
    }

    // 14. Mixing two densities must stay unbiased. A cosine density mixed with a
    //     uniform one covers the whole sphere, so integrating cos^2 through the
    //     mixture must still land on the full-sphere answer, 4*pi/3.
    {
        RNG rng(1212u);
        vec3 n = unit_vector(vec3(0.4, -0.6, 0.7));
        mixture_pdf mix(make_shared<cosine_pdf>(n), make_shared<sphere_pdf>());

        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            vec3 d = mix.generate(rng);
            double c = dot(unit_vector(d), n);
            sum += (c * c) / mix.value(d);
        }
        check("mixture_pdf integrates cos^2 to 4pi/3", sum / N, 4 * pi / 3, 0.02);
    }

    std::cout << "----------------------------------------------------------------\n";
    if (failures == 0) {
        std::cout << "ALL CHECKS PASSED\n";
        return 0;
    }
    std::cout << failures << " CHECK(S) FAILED\n";
    return 1;
}
