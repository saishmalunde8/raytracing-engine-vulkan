#ifndef PDF_H
#define PDF_H

#include "raytracer/core/rtweekend.h"
#include "raytracer/core/onb.h"
#include "raytracer/core/rng.h"
#include "raytracer/geometry/hittable.h"

// A probability density over directions.
//
// Two operations, and the split between them matters:
//
//   generate()  draws a direction. It is sampling, so it consumes the caller's
//               RNG and must be handed one.
//   value()     reports how likely this density was to produce a direction it
//               is given. It is pure arithmetic on a direction that already
//               exists -- no randomness -- so it takes no RNG.
//
// Keeping value() RNG-free means it can be called on a direction that some
// other density produced. That is what makes mixing densities possible: ask one
// to generate, then ask both how likely that direction was.
class pdf {
  public:
    virtual ~pdf() {}

    virtual double value(const vec3& direction) const = 0;
    virtual vec3 generate(RNG& rng) const = 0;
};

// Uniform over the whole sphere. Every direction equally likely, so the density
// is constant: one over the sphere's total solid angle.
class sphere_pdf : public pdf {
  public:
    sphere_pdf() {}

    double value(const vec3& direction) const override {
        return 1 / (4 * pi);
    }

    vec3 generate(RNG& rng) const override {
        return rng.random_unit_vector();
    }
};

// Cosine-weighted over the hemisphere around a normal. Favours directions near
// the normal, which is where a diffuse surface sends most of its light, so
// samples get spent where they carry the most energy.
class cosine_pdf : public pdf {
  public:
    cosine_pdf(const vec3& w) : uvw(w) {}

    double value(const vec3& direction) const override {
        auto cosine_theta = dot(unit_vector(direction), uvw.w());
        return std::fmax(0, cosine_theta / pi);
    }

    vec3 generate(RNG& rng) const override {
        return uvw.transform(rng.random_cosine_direction());
    }

  private:
    onb uvw;
};

// Aims at geometry rather than at a surface's own preference: directions drawn
// here point at the given object, which is how a shadowed surface finds a small
// bright light instead of waiting to stumble into it.
class hittable_pdf : public pdf {
  public:
    hittable_pdf(const hittable& objects, const point3& origin)
      : objects(objects), origin(origin) {}

    double value(const vec3& direction) const override {
        return objects.pdf_value(origin, direction);
    }

    vec3 generate(RNG& rng) const override {
        return objects.random(origin, rng);
    }

  private:
    const hittable& objects;
    point3 origin;
};

// Two densities, sampled half the time each.
//
// Neither one alone is enough. Aiming only at lights misses everything lit
// indirectly; following only the surface's own preference rarely finds a small
// light. Sampling from either but scoring against the average of both keeps the
// result unbiased while getting the variance reduction of both strategies --
// which is exactly why value() had to be callable on a direction this density
// did not generate.
class mixture_pdf : public pdf {
  public:
    mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1) {
        p[0] = p0;
        p[1] = p1;
    }

    double value(const vec3& direction) const override {
        return 0.5 * p[0]->value(direction) + 0.5 * p[1]->value(direction);
    }

    vec3 generate(RNG& rng) const override {
        if (rng.next_double() < 0.5)
            return p[0]->generate(rng);
        return p[1]->generate(rng);
    }

  private:
    shared_ptr<pdf> p[2];
};

#endif
