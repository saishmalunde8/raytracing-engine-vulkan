// Example scenes for the render gallery.
//
// Kept separate from src/main.cpp so the renderer's own scene list stays focused
// on the ones that exercise the engine. These exist to be looked at.
//
//   clang++ -std=c++17 -O2 -Iinclude tools/gallery.cpp -o build/gallery
//   ./build/gallery <scene> [width] [spp] > out.ppm
//   ./build/gallery list

#include "raytracer/core/rtweekend.h"
#include "raytracer/acceleration/bvh.h"
#include "raytracer/camera/camera.h"
#include "raytracer/geometry/constant_medium.h"
#include "raytracer/geometry/hittable.h"
#include "raytracer/geometry/hittable_list.h"
#include "raytracer/materials/material.h"
#include "raytracer/geometry/quad.h"
#include "raytracer/geometry/sphere.h"
#include "raytracer/textures/texture.h"

#include <string>
#include <cstring>

// ---------------------------------------------------------------- shared bits

// A camera preset every scene starts from, so differences between renders come
// from the scene rather than from forgotten settings.
static camera base_camera(int width, int spp, double aspect = 16.0/9.0) {
    camera cam;
    cam.aspect_ratio       = aspect;
    cam.image_width        = width;
    cam.samples_per_pixel  = spp;
    cam.max_depth          = 50;
    cam.background         = color(0,0,0);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 1.0;
    cam.deterministic      = true;
    cam.vup                = vec3(0,1,0);
    cam.defocus_angle      = 0;
    return cam;
}

// =============================================================== 1. HERO
//
// The dusk scene, unchanged from src/main.cpp. Edit freely here -- this copy is
// for experimenting with, the one in main.cpp is the shipped renderer's.
static void hero(int width, int spp) {
    hittable_list world;
    hittable_list lights;

    auto checker = make_shared<checker_texture>(0.32, color(0.15, 0.25, 0.35), color(0.85, 0.88, 0.92));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

    auto difflight = make_shared<diffuse_light>(color(1.0, 0.85, 0.6)*10);
    auto sun = make_shared<sphere>(point3(-15, 9, -15), 7, difflight);
    world.add(sun);
    lights.add(sun);

    auto star_light = make_shared<diffuse_light>(color(1.0, 1.0, 1.0) * 2.0);
    double star_radius = 0.08;  // VERY small
    double star_height = 30.0;  // far away

    for (int i = 0; i < 60; i++) {

        double theta = random_double(0, 2 * pi);
        double phi   = random_double(0.25 * pi, 0.5 * pi);
        // only upper sky, not horizon

        double x = star_height * sin(phi) * cos(theta);
        double y = star_height * cos(phi);
        double z = star_height * sin(phi) * sin(theta);

        world.add(make_shared<sphere>(
            point3(x, y, z),
            star_radius,
            star_light
        ));
    }

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    auto center2 = center + vec3(0, random_double(0,.5), 0);
                    world.add(make_shared<sphere>(center, center2, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1, material1));

    auto material2 = make_shared<lambertian>(color(0.7, 0.5, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    world = hittable_list(make_shared<bvh_node>(world));

    auto cam = base_camera(width, spp);
    cam.max_depth         = 10;
    cam.background        = color(0,0,0.01);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 0.85;

    cam.vfov     = 21;
    cam.lookfrom = point3(13,2,-3);
    cam.lookat   = point3(0,0,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    cam.render(world, lights);
}

// ======================================================= 2-4. LIGHT SIZE STUDY
//
// The same Cornell box with the ceiling light shrunk twice. Emission is scaled
// by the inverse of the area, so total power stays roughly constant and the
// three renders are comparable -- what changes is how hard the light is to find.
static void cornell_light(int width, int spp, double scale) {
    hittable_list world;
    hittable_list lights;

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));

    double du = 130 * scale, dv = 105 * scale;
    double cx = 343 - 130*0.5, cz = 332 - 105*0.5;   // centre of the original
    auto light = make_shared<diffuse_light>(color(15,15,15) / (scale*scale));

    world.add(make_shared<quad>(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green));
    world.add(make_shared<quad>(point3(0,0,0),   vec3(0,555,0), vec3(0,0,555), red));
    world.add(make_shared<quad>(point3(0,0,0),   vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white));
    world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white));

    auto light_quad = make_shared<quad>(
        point3(cx + du*0.5, 554, cz + dv*0.5), vec3(-du,0,0), vec3(0,0,-dv), light);
    world.add(light_quad);
    lights.add(light_quad);

    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));
    world.add(box1);

    shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130,0,65));
    world.add(box2);

    auto cam = base_camera(width, spp, 1.0);
    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.render(world, lights);
}

// ============================================================== 5. MATERIALS
//
// One row, identical lighting, so the materials can be compared directly.
static void materials(int width, int spp) {
    hittable_list world;
    hittable_list lights;

    world.add(make_shared<quad>(point3(-30,0,-12), vec3(60,0,0), vec3(0,0,24),
              make_shared<lambertian>(color(0.55,0.55,0.58))));

    auto noise = make_shared<noise_texture>(6.0, noise_mode::turbulence, 7);

    shared_ptr<material> mats[] = {
        make_shared<lambertian>(color(0.75, 0.30, 0.28)),
        make_shared<metal>(color(0.86, 0.88, 0.90), 0.0),
        make_shared<metal>(color(0.86, 0.86, 0.80), 0.38),
        make_shared<dielectric>(1.5),
        make_shared<perlin_metal>(noise, color(0.82,0.84,0.88), 0.01, 0.30),
    };
    for (int i = 0; i < 5; i++)
        world.add(make_shared<sphere>(point3(-8.5 + i*3.4, 1.3, 0), 1.3, mats[i]));

    // Sixth slot: isotropic, which only exists inside a medium.
    auto smoke_bound = make_shared<sphere>(point3(8.5, 1.3, 0), 1.3,
                                           make_shared<dielectric>(1.5));
    world.add(make_shared<constant_medium>(smoke_bound, 0.9, color(0.85,0.85,0.9)));

    auto lamp = make_shared<quad>(point3(-9,9,-5), vec3(20,0,0), vec3(0,0,8),
                                  make_shared<diffuse_light>(color(6,6,6)));
    world.add(lamp);
    lights.add(lamp);

    auto cam = base_camera(width, spp, 3.2);
    cam.vfov     = 16;
    cam.lookfrom = point3(0, 3.0, 26);
    cam.lookat   = point3(0, 1.2, 0);
    cam.render(world, lights);
}

// =============================================================== 6. TEXTURES
static void textures(int width, int spp) {
    hittable_list world;
    hittable_list lights;

    world.add(make_shared<quad>(point3(-30,0,-12), vec3(60,0,0), vec3(0,0,24),
              make_shared<lambertian>(color(0.5,0.5,0.52))));

    shared_ptr<texture> texs[] = {
        make_shared<checker_texture>(0.6, color(.12,.22,.30), color(.88,.90,.92)),
        make_shared<image_texture>("assets/textures/earthmap.jpg"),
        make_shared<noise_texture>(3.0, noise_mode::plain),
        make_shared<noise_texture>(3.0, noise_mode::turbulence, 7),
        make_shared<noise_texture>(3.0, noise_mode::marble, 7),
        make_shared<noise_texture>(3.0, noise_mode::wood, 7),
    };
    for (int i = 0; i < 6; i++)
        world.add(make_shared<sphere>(point3(-8.5 + i*3.4, 1.3, 0), 1.3,
                  make_shared<lambertian>(texs[i])));

    auto lamp = make_shared<quad>(point3(-11,9,-5), vec3(24,0,0), vec3(0,0,8),
                                  make_shared<diffuse_light>(color(6,6,6)));
    world.add(lamp);
    lights.add(lamp);

    auto cam = base_camera(width, spp, 3.2);
    cam.vfov     = 16;
    cam.lookfrom = point3(0, 3.0, 26);
    cam.lookat   = point3(0, 1.2, 0);
    cam.render(world, lights);
}

// ========================================================== 9. PERLIN METAL
//
// The material that exists in no reference implementation: roughness driven by
// a noise texture, so a single surface is polished in places and matte in
// others.
static void perlinmetal(int width, int spp) {
    hittable_list world;
    hittable_list lights;

    auto floor_tex = make_shared<checker_texture>(1.1,
                     color(0.05,0.05,0.06), color(0.92,0.90,0.86));
    world.add(make_shared<quad>(point3(-30,0,-16), vec3(60,0,0), vec3(0,0,32),
              make_shared<lambertian>(floor_tex)));

    // Something with edges for the varying roughness to smear. Against a smooth
    // background, polished and rough are indistinguishable.
    world.add(make_shared<sphere>(point3(-9.5, 2.2, -7.0), 2.2,
              make_shared<lambertian>(color(0.85, 0.22, 0.18))));
    world.add(make_shared<sphere>(point3( 9.5, 2.2, -7.0), 2.2,
              make_shared<lambertian>(color(0.15, 0.45, 0.85))));
    world.add(make_shared<sphere>(point3( 0.0, 2.6, -9.5), 2.6,
              make_shared<lambertian>(color(0.95, 0.80, 0.25))));

    double scales[]   = { 0.7, 1.4, 2.6, 4.5 };
    double maxfuzz[]  = { 0.05, 0.22, 0.45, 0.75 };
    for (int i = 0; i < 4; i++) {
        auto tex = make_shared<noise_texture>(scales[i], noise_mode::marble, 7);
        world.add(make_shared<sphere>(point3(-6.6 + i*4.4, 1.6, 0), 1.6,
                  make_shared<perlin_metal>(tex, color(0.88,0.82,0.70),
                                            0.0, maxfuzz[i])));
    }

    auto key = make_shared<quad>(point3(-5,10,-8), vec3(10,0,0), vec3(0,0,6),
               make_shared<diffuse_light>(color(1.0,0.88,0.70) * 26));
    world.add(key);
    lights.add(key);

    auto rim = make_shared<sphere>(point3(-13, 5, 7), 1.6,
               make_shared<diffuse_light>(color(0.55,0.70,1.0) * 30));
    world.add(rim);
    lights.add(rim);

    auto cam = base_camera(width, spp, 2.4);
    cam.background         = color(0.05, 0.06, 0.09);
    cam.g_use_sky_gradient = true;
    cam.g_sky_strength     = 0.45;   // an environment for the metal to reflect
    cam.vfov     = 26;
    cam.lookfrom = point3(0.5, 3.4, 16.0);
    cam.lookat   = point3(0, 1.6, 0);
    cam.defocus_angle = 0.3;
    cam.focus_dist    = 16.0;
    cam.render(world, lights);
}

// ========================================================== 10. MULTI LIGHT
//
// Several small coloured emitters. Overlapping coloured shadows, and the light
// list averaging across all of them.
static void multilight(int width, int spp) {
    hittable_list world;
    hittable_list lights;

    world.add(make_shared<quad>(point3(-24,0,-16), vec3(48,0,0), vec3(0,0,32),
              make_shared<lambertian>(color(0.80,0.80,0.82))));
    world.add(make_shared<quad>(point3(-24,0,-16), vec3(48,0,0), vec3(0,20,0),
              make_shared<lambertian>(color(0.78,0.78,0.80))));

    struct L { point3 p; color c; };
    L set[] = {
        { point3(-6.5, 6.0, 4.0), color(1.00, 0.22, 0.28) },
        { point3( 0.0, 7.2, 5.0), color(0.25, 1.00, 0.45) },
        { point3( 6.5, 6.0, 4.0), color(0.30, 0.45, 1.00) },
    };
    for (auto& L : set) {
        auto e = make_shared<sphere>(L.p, 0.5,
                 make_shared<diffuse_light>(L.c * 42));
        world.add(e);
        lights.add(e);
    }

    world.add(make_shared<sphere>(point3(-2.6, 1.7, 0), 1.7,
              make_shared<lambertian>(color(0.86,0.86,0.86))));
    world.add(make_shared<sphere>(point3( 2.6, 1.7, 0), 1.7,
              make_shared<lambertian>(color(0.86,0.86,0.86))));
    shared_ptr<hittable> col = box(point3(0,0,0), point3(1.4,5.0,1.4),
              make_shared<lambertian>(color(0.84,0.84,0.84)));
    col = make_shared<rotate_y>(col, 22);
    col = make_shared<translate>(col, vec3(-0.7, 0, -4.0));
    world.add(col);

    auto cam = base_camera(width, spp);
    cam.vfov     = 40;
    cam.lookfrom = point3(0, 5.0, 15.0);
    cam.lookat   = point3(0, 2.0, 0);
    cam.render(world, lights);
}

// =============================================================== 11. BOKEH
static void bokeh(int width, int spp) {
    seed_scene_rng(404u);
    hittable_list world;
    hittable_list lights;

    world.add(make_shared<sphere>(point3(0,-1000,0), 1000,
              make_shared<lambertian>(color(0.16,0.17,0.20))));

    for (int i = 0; i < 150; i++) {
        double x = random_double(-14, 14);
        double z = random_double(-26, 2);
        double r = random_double(0.16, 0.42);
        world.add(make_shared<sphere>(point3(x, r, z), r,
                  make_shared<metal>(color(0.88,0.86,0.82), 0.02)));
    }
    world.add(make_shared<sphere>(point3(0, 1.1, 3.0), 1.1,
              make_shared<dielectric>(1.5)));
    world.add(make_shared<sphere>(point3(-2.4, 0.8, 2.2), 0.8,
              make_shared<lambertian>(color(0.72,0.36,0.30))));

    auto key = make_shared<sphere>(point3(-8, 9, -14), 2.2,
               make_shared<diffuse_light>(color(1.0, 0.88, 0.72) * 26));
    world.add(key);
    lights.add(key);

    auto cam = base_camera(width, spp);
    cam.background    = color(0.015, 0.018, 0.030);
    cam.vfov          = 26;
    cam.lookfrom      = point3(0, 1.5, 11.5);
    cam.lookat        = point3(0, 1.0, 3.0);
    cam.defocus_angle = 3.2;
    cam.focus_dist    = 8.6;
    cam.render(world, lights);
}

// =============================================================== 12. GLASS
static void glass(int width, int spp) {
    hittable_list world;
    hittable_list lights;

    auto checker = make_shared<checker_texture>(0.7,
                   color(.16,.18,.22), color(.80,.82,.85));
    world.add(make_shared<quad>(point3(-24,0,-18), vec3(48,0,0), vec3(0,0,36),
              make_shared<lambertian>(checker)));

    world.add(make_shared<sphere>(point3(-3.4, 1.5, 0), 1.5, make_shared<dielectric>(1.5)));
    // hollow shell: a negative radius flips the surface normal inward
    world.add(make_shared<sphere>(point3( 0.0, 1.5, 0), 1.5, make_shared<dielectric>(1.5)));
    world.add(make_shared<sphere>(point3( 0.0, 1.5, 0), -1.2, make_shared<dielectric>(1.5)));
    world.add(make_shared<sphere>(point3( 3.4, 1.5, 0), 1.5, make_shared<dielectric>(2.4)));

    auto key = make_shared<quad>(point3(-3,9,-4), vec3(6,0,0), vec3(0,0,4),
               make_shared<diffuse_light>(color(1,1,1) * 22));
    world.add(key);
    lights.add(key);

    auto cam = base_camera(width, spp, 2.2);
    cam.vfov     = 28;
    cam.lookfrom = point3(0, 3.4, 13.0);
    cam.lookat   = point3(0, 1.3, 0);
    cam.render(world, lights);
}

// ============================================================= 13. GOD RAYS
//
// Fog plus a bright source behind slatted geometry. Light becomes visible in the
// air between the slats.
static void godrays(int width, int spp) {
    hittable_list world;
    hittable_list lights;

    auto stone = make_shared<lambertian>(color(0.30, 0.28, 0.26));

    // An enclosed room, so no ray escapes to the flat background and the fog has
    // a boundary that is never itself visible.
    world.add(make_shared<quad>(point3(-16,0,-22),  vec3(32,0,0), vec3(0,0,34), stone)); // floor
    world.add(make_shared<quad>(point3(-16,14,-22), vec3(32,0,0), vec3(0,0,34), stone)); // ceiling
    world.add(make_shared<quad>(point3(-16,0,-22),  vec3(0,0,34), vec3(0,14,0), stone)); // left
    world.add(make_shared<quad>(point3( 16,0,-22),  vec3(0,0,34), vec3(0,14,0), stone)); // right
    world.add(make_shared<quad>(point3(-16,0,12),   vec3(32,0,0), vec3(0,14,0), stone)); // behind

    // The far wall is a bright window. A broad source behind the slats is what
    // produces distinct shafts -- a small point source just makes a glow.
    auto window_mat = make_shared<diffuse_light>(color(1.0, 0.93, 0.78) * 14);
    auto window = make_shared<quad>(point3(-16,0,-22), vec3(32,0,0), vec3(0,14,0), window_mat);
    world.add(window);
    lights.add(window);

    // Slats between the window and the camera, carving the light into bands.
    for (int i = -5; i <= 5; i++) {
        shared_ptr<hittable> slat = box(point3(0,0,0), point3(0.8, 14.0, 0.8), stone);
        slat = make_shared<translate>(slat, vec3(i * 2.9 - 0.4, 0, -13.0));
        world.add(slat);
    }

    world.add(make_shared<sphere>(point3(-2.6, 1.5, -1.0), 1.5,
              make_shared<lambertian>(color(0.55,0.52,0.48))));
    world.add(make_shared<sphere>(point3(2.6, 1.2, 1.6), 1.2,
              make_shared<metal>(color(0.80,0.78,0.74), 0.10)));

    auto scene = make_shared<hittable_list>();
    scene->add(make_shared<bvh_node>(world));

    // Fog boundary large enough to enclose the whole room, so its own edge never
    // appears in frame.
    auto fog_bound = make_shared<sphere>(point3(0, 6, -5), 60,
                                         make_shared<dielectric>(1.5));
    scene->add(make_shared<constant_medium>(fog_bound, 0.013, color(0.94,0.90,0.84)));

    auto cam = base_camera(width, spp);
    cam.background = color(0.008, 0.010, 0.016);
    cam.vfov       = 46;
    cam.lookfrom   = point3(6.5, 4.6, 9.0);
    cam.lookat     = point3(-0.5, 4.0, -12.0);
    cam.render(*scene, lights);
}

// ============================================================= 14. MIRRORS
static void mirrors(int width, int spp) {
    hittable_list world;
    hittable_list lights;

    world.add(make_shared<quad>(point3(-20,0,-20), vec3(40,0,0), vec3(0,0,40),
              make_shared<lambertian>(color(0.24,0.24,0.26))));

    auto chrome = make_shared<metal>(color(0.95,0.95,0.96), 0.0);
    world.add(make_shared<sphere>(point3(-4.6, 3.4, 0), 3.4, chrome));
    world.add(make_shared<sphere>(point3( 4.6, 3.4, 0), 3.4, chrome));

    auto core = make_shared<sphere>(point3(0, 2.0, 0), 0.55,
                make_shared<diffuse_light>(color(1.0, 0.55, 0.25) * 34));
    world.add(core);
    lights.add(core);

    auto fill = make_shared<quad>(point3(-6,12,-5), vec3(12,0,0), vec3(0,0,6),
                make_shared<diffuse_light>(color(0.55,0.65,0.95) * 3.2));
    world.add(fill);
    lights.add(fill);

    auto cam = base_camera(width, spp);
    cam.background = color(0.01,0.01,0.02);
    cam.vfov       = 34;
    cam.lookfrom   = point3(0, 5.2, 15.5);
    cam.lookat     = point3(0, 2.6, 0);
    cam.render(world, lights);
}

// ============================================================== 15. NESTED
//
// A glass shell holding smoke, and a glass sphere holding a denser core.
static void nested(int width, int spp) {
    hittable_list world;
    hittable_list lights;

    world.add(make_shared<quad>(point3(-24,0,-18), vec3(48,0,0), vec3(0,0,36),
              make_shared<lambertian>(color(0.62,0.62,0.66))));

    // glass shell containing smoke
    auto shell = make_shared<sphere>(point3(-3.2, 2.0, 0), 2.0,
                 make_shared<dielectric>(1.5));
    world.add(shell);
    world.add(make_shared<constant_medium>(shell, 0.42, color(0.22,0.28,0.42)));

    // glass sphere with a dense bright core
    auto outer = make_shared<sphere>(point3(3.2, 2.0, 0), 2.0,
                 make_shared<dielectric>(1.5));
    world.add(outer);
    auto inner = make_shared<sphere>(point3(3.2, 2.0, 0), 0.9,
                 make_shared<dielectric>(1.5));
    world.add(inner);
    world.add(make_shared<constant_medium>(inner, 1.4, color(0.85,0.45,0.25)));

    auto key = make_shared<quad>(point3(-4,10,-4), vec3(8,0,0), vec3(0,0,5),
               make_shared<diffuse_light>(color(1,1,1) * 18));
    world.add(key);
    lights.add(key);

    auto cam = base_camera(width, spp, 2.0);
    cam.vfov     = 30;
    cam.lookfrom = point3(0, 3.6, 13.5);
    cam.lookat   = point3(0, 1.9, 0);
    cam.render(world, lights);
}

// ------------------------------------------------------------------ dispatch

int main(int argc, char** argv) {
    std::string scene = (argc > 1) ? argv[1] : "list";
    int width = (argc > 2) ? std::atoi(argv[2]) : 700;
    int spp   = (argc > 3) ? std::atoi(argv[3]) : 100;

    if      (scene == "hero")         hero(width, spp);
    else if (scene == "light_full")   cornell_light(width, spp, 1.00);
    else if (scene == "light_quarter")cornell_light(width, spp, 0.50);
    else if (scene == "light_pinhole")cornell_light(width, spp, 0.20);
    else if (scene == "materials")    materials(width, spp);
    else if (scene == "textures")     textures(width, spp);
    else if (scene == "perlinmetal")  perlinmetal(width, spp);
    else if (scene == "multilight")   multilight(width, spp);
    else if (scene == "bokeh")        bokeh(width, spp);
    else if (scene == "glass")        glass(width, spp);
    else if (scene == "godrays")      godrays(width, spp);
    else if (scene == "mirrors")      mirrors(width, spp);
    else if (scene == "nested")       nested(width, spp);
    else {
        std::clog << "usage: gallery <scene> [width] [spp] > out.ppm\n\nscenes:\n"
                  << "  hero           dusk scene (same as main.cpp)\n"
                  << "  light_full     cornell box, full-size light\n"
                  << "  light_quarter  cornell box, half-edge light\n"
                  << "  light_pinhole  cornell box, near-pinhole light\n"
                  << "  materials      lambertian / metal / rough / glass / perlin_metal / isotropic\n"
                  << "  textures       checker / image / perlin / turbulence / marble / wood\n"
                  << "  perlinmetal    noise-driven roughness study\n"
                  << "  multilight     three coloured emitters\n"
                  << "  bokeh          shallow focus, specular highlights\n"
                  << "  glass          dielectrics, including a hollow shell\n"
                  << "  godrays        light shafts through slats\n"
                  << "  mirrors        facing chrome spheres\n"
                  << "  nested         media inside glass\n";
        return 1;
    }
    return 0;
}
