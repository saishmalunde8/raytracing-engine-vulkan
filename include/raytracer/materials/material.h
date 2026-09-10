#ifndef MATERIAL_H
#define MATERIAL_H

#include "raytracer/geometry/hittable.h"
#include "raytracer/textures/texture.h"
#include "raytracer/core/rng.h"
#include "raytracer/sampling/pdf.h"

// What a material reports back when a ray scatters off it.
//
// skip_pdf marks a scatter whose outgoing direction is not a choice but a
// consequence -- a mirror reflection, a refraction. There is no spread of
// possible directions to sample from, so those bypass density sampling entirely
// and hand the integrator a finished ray. Anything with a spread instead supplies
// a density in pdf_ptr for the integrator to sample and weight.
class scatter_record {
  public:
    color attenuation;
    shared_ptr<pdf> pdf_ptr;
    bool skip_pdf = true;
    ray skip_pdf_ray;
};

class material {
  public:
    virtual ~material() = default;

    // The incoming ray and hit record are passed so an emitter can decide
    // whether it is being looked at from the front or the back.
    virtual color emitted(const ray& r_in, const hit_record& rec,
                          double u, double v, const point3& p) const {
        return color(0,0,0);
    }

    // Scattering draws from the caller's RNG. Deterministic renders pass a
    // per-pixel-sample instance, everything else passes the thread's instance.
    virtual bool scatter(
        const ray& r_in, const hit_record& rec, scatter_record& srec, RNG& rng
    ) const {
        return false;
    }

    // How strongly this material actually scatters into `scattered`.
    //
    // Distinct from the density used to *pick* that direction. The integrator
    // divides one by the other, and the two only cancel when the sampling
    // density happens to match the material exactly. Separating them is what
    // makes it legal to sample from somewhere else entirely -- a light, say.
    virtual double scattering_pdf(
        const ray& r_in, const hit_record& rec, const ray& scattered
    ) const {
        return 0;
    }
};

class lambertian : public material {
  public:
    lambertian(const color& albedo) : tex(make_shared<solid_color>(albedo)) {}
    lambertian(shared_ptr<texture> tex) : tex(tex) {}

    // Reports a density instead of a direction. The integrator does the picking,
    // which is what lets it mix this density with one aimed at a light.
    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec, RNG& rng)
    const override {
        srec.attenuation = tex->value(rec.u, rec.v, rec.p);
        srec.pdf_ptr = make_shared<cosine_pdf>(rec.normal);
        srec.skip_pdf = false;
        return true;
    }

    double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    const override {
        auto cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
        return cos_theta < 0 ? 0 : cos_theta/pi;
    }

    private:
        shared_ptr<texture> tex;
};

class metal : public material {
  public:
    metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec, RNG& rng)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + fuzz * rng.random_unit_vector();
        srec.attenuation = albedo;
        srec.skip_pdf = true;
        srec.skip_pdf_ray = ray(rec.p, reflected, r_in.time());
        return (dot(srec.skip_pdf_ray.direction(), rec.normal) > 0);
    }

  private:
    color albedo;
    double fuzz;
};

class perlin_metal : public material {
  public:
    perlin_metal(shared_ptr<texture> noise_tex, color base_albedo = color(0.8,0.85,0.88), double min_fuzz = 0.0, double max_fuzz = 0.12)
      : noise_tex(noise_tex), base_albedo(base_albedo), min_fuzz(min_fuzz), max_fuzz(max_fuzz) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec, RNG& rng)
    const override {
        // Sample the noise texture at the hit point
        color nval = noise_tex->value(rec.u, rec.v, rec.p);

        // Convert noise to scalar in [0,1]
        double nn = (nval.x() + nval.y() + nval.z()) / 3.0;
        nn = interval(0.0, 1.0).clamp(nn);

        // Map noise to albedo tint and fuzz (roughness)
        // Slight tint variation around base_albedo
        color albedo = base_albedo * (0.85 + 0.3 * nn);

        double fuzz = min_fuzz + (max_fuzz - min_fuzz) * nn;
        if (fuzz < 0) fuzz = 0;
        if (fuzz > 1) fuzz = 1;

        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + fuzz * rng.random_unit_vector();
        srec.attenuation = albedo;
        srec.skip_pdf = true;
        srec.skip_pdf_ray = ray(rec.p, reflected, r_in.time());
        return (dot(srec.skip_pdf_ray.direction(), rec.normal) > 0);
    }

  private:
    shared_ptr<texture> noise_tex;
    color base_albedo;
    double min_fuzz;
    double max_fuzz;
};

class dielectric : public material {
  public:
    dielectric(double refraction_index) : refraction_index(refraction_index) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec, RNG& rng)
    const override {
        srec.attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());

        double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > rng.next_double())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        srec.skip_pdf = true;
        srec.skip_pdf_ray = ray(rec.p, direction, r_in.time());
        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;

    static double reflectance(double cosine, double refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
};

class diffuse_light : public material {
  public:
    diffuse_light(shared_ptr<texture> tex) : tex(tex) {}
    diffuse_light(const color& emit) : tex(make_shared<solid_color>(emit)) {}

    // Emits from its front face only.
    //
    // Light sampling aims rays at a light's front face. If the same light also
    // emitted backwards, its energy would be counted twice -- once by an aimed
    // ray, once by an ordinary bounce arriving from behind -- and the scene
    // would render brighter than it should.
    color emitted(const ray& r_in, const hit_record& rec,
                  double u, double v, const point3& p) const override {
        if (!rec.front_face)
            return color(0,0,0);
        return tex->value(u, v, p);
    }

  private:
    shared_ptr<texture> tex;
};

class isotropic : public material {
  public:
    isotropic(const color& albedo) : tex(make_shared<solid_color>(albedo)) {}
    isotropic(shared_ptr<texture> tex) : tex(tex) {}

    // Scatters with no preferred direction, so its density is uniform over the
    // whole sphere rather than a hemisphere around a normal.
    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec, RNG& rng)
    const override {
        srec.attenuation = tex->value(rec.u, rec.v, rec.p);
        srec.pdf_ptr = make_shared<sphere_pdf>();
        srec.skip_pdf = false;
        return true;
    }

    double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    const override {
        return 1 / (4 * pi);
    }

  private:
    shared_ptr<texture> tex;
};

#endif
