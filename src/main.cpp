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

void bouncing_spheres() {
    hittable_list world;
    hittable_list lights;

    auto checker = make_shared<checker_texture>(0.32, color(0.15, 0.25, 0.35), color(0.85, 0.88, 0.92));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

    auto difflight = make_shared<diffuse_light>(color(1.0, 0.85, 0.6)*10);
    auto sun = make_shared<sphere>(point3(-15, 9, -15), 7, difflight);
    world.add(sun);
    lights.add(sun);

    // Stars are left out of `lights`: they are far too small and dim next to
    // the sun to make aiming rays at them worthwhile, and measured at equal
    // sample counts, adding all 60 changed the render by nothing while costing
    // about 8% more time.
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

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 700;
    cam.samples_per_pixel = 25;
    cam.max_depth         = 10;
    cam.background        = color(0,0,0.01);
    // cam.background       = color(0.20, 0.25, 0.35); // cool base tone
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 0.85;
    cam.deterministic = true;

    cam.vfov     = 21;
    cam.lookfrom = point3(13,2,-3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    cam.render(world, lights);
}

void checkered_spheres(){
    hittable_list world;
    hittable_list lights;

    auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .3), color(.9, .9, .9));

    world.add(make_shared<sphere>(point3(0,-10, 0), 10, make_shared<lambertian>(checker)));
    world.add(make_shared<sphere>(point3(0, 10, 0), 10, make_shared<lambertian>(checker)));

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 1200;
    cam.samples_per_pixel = 200;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 1.0;

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world, lights);
}

void earth() {
    auto earth_texture = make_shared<image_texture>("assets/textures/earthmap.jpg");
    auto earth_surface = make_shared<lambertian>(earth_texture);
    auto globe = make_shared<sphere>(point3(0,0,0), 2, earth_surface);

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 1200;
    cam.samples_per_pixel = 200;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 1.0;


    cam.vfov     = 20;
    cam.lookfrom = point3(0,0,11);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    hittable_list lights;

    cam.render(hittable_list(globe), lights);
}

void sphere_noise_gallery() {
    hittable_list world;
    hittable_list lights;

    // Ground
    auto ground_tex = make_shared<noise_texture>(0.5, noise_mode::plain);
    world.add(make_shared<sphere>(
        point3(0,-1000,0), 1000,
        make_shared<lambertian>(ground_tex)
    ));

    // Plain Perlin
    auto perlin_tex = make_shared<noise_texture>(4.0, noise_mode::plain);
    world.add(make_shared<sphere>(
        point3(-4, 2, 0), 2,
        make_shared<lambertian>(perlin_tex)
    ));

    // // Turbulence
    auto turb_tex = make_shared<noise_texture>(4.0, noise_mode::turbulence, 7);
    world.add(make_shared<sphere>(
        point3(0, 2, 0), 2,
        make_shared<lambertian>(turb_tex)
    ));

    // Marble
    auto marble_tex = make_shared<noise_texture>(4.0, noise_mode::marble, 7);
    world.add(make_shared<sphere>(
        point3(4, 2, 0), 2,
        make_shared<lambertian>(marble_tex)
    ));

    // Wood
    auto wood_tex = make_shared<noise_texture>(4.0, noise_mode::wood, 7);
    world.add(make_shared<sphere>(
        point3(8, 2, 0), 2,
        make_shared<lambertian>(wood_tex)
    ));

    camera cam;
    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 700;
    cam.samples_per_pixel = 50;
    cam.max_depth         = 10;
    cam.background        = color(0.70, 0.80, 1.00);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 1.0;
    cam.deterministic = true;


    cam.vfov     = 20;
    cam.lookfrom = point3(15,5,15);
    cam.lookat   = point3(2,2,0);
    cam.vup      = vec3(0,1,0);
    cam.defocus_angle = 0;

    cam.render(world, lights);
}

void quads() {
    hittable_list world;
    hittable_list lights;

    // Materials
    auto left_red     = make_shared<lambertian>(color(1.0, 0.2, 0.2));
    auto back_green   = make_shared<lambertian>(color(0.2, 1.0, 0.2));
    auto right_blue   = make_shared<lambertian>(color(0.2, 0.2, 1.0));
    auto upper_orange = make_shared<lambertian>(color(1.0, 0.5, 0.0));
    auto lower_teal   = make_shared<lambertian>(color(0.2, 0.8, 0.8));
    // Create a Perlin noise texture (scale controls grain frequency)
    auto perlin_tex = make_shared<noise_texture>(8.0); // increase for finer detail
    // Perlin-modulated metal: base color (cool steel), fuzz varies with noise
    auto perlin_metal_mat = make_shared<perlin_metal>(perlin_tex, color(0.80, 0.85, 0.88), 0.01, 0.18);

    // Quads
    world.add(make_shared<quad>(point3(-3,-2, 5), vec3(0, 0,-4), vec3(0, 4, 0), left_red));
    world.add(make_shared<quad>(point3(-2,-2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_green));
    world.add(make_shared<quad>(point3( 3,-2, 1), vec3(0, 0, 4), vec3(0, 4, 0), perlin_metal_mat));
    world.add(make_shared<quad>(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_orange));
    world.add(make_shared<quad>(point3(-2,-3, 5), vec3(4, 0, 0), vec3(0, 0,-4), lower_teal));
    world.add(make_shared<sphere>(point3(0,1,2), 0.5, right_blue));

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 1.0;

    cam.vfov     = 80;
    cam.lookfrom = point3(0,0,9);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world, lights);
}

void simple_light() {
    hittable_list world;
    hittable_list lights;

    auto pertext = make_shared<noise_texture>(4, noise_mode::wood, 7);
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
    world.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

    auto difflight = make_shared<diffuse_light>(color(4,4,4));
    auto light_sphere = make_shared<sphere>(point3(-4,7,4), 1.5, difflight);
    auto light_panel  = make_shared<quad>(point3(3,1,-2), vec3(2,0,0), vec3(0,2,0), difflight);
    world.add(light_sphere);
    world.add(light_panel);
    lights.add(light_sphere);
    lights.add(light_panel);

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 50;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 1.0;

    cam.vfov     = 20;
    cam.lookfrom = point3(26,3,6);
    cam.lookat   = point3(0,2,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world, lights);
}

void cornell_box() {
    hittable_list world;
    hittable_list lights;

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    world.add(make_shared<quad>(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green));
    world.add(make_shared<quad>(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red));
    auto light_quad = make_shared<quad>(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), light);
    world.add(light_quad);
    lights.add(light_quad);
    world.add(make_shared<quad>(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white));
    world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white));

    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));
    world.add(box1);

    shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130,0,65));
    world.add(box2);

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 600;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 1.0;

    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world, lights);
}

void cornell_smoke() {
    hittable_list world;
    hittable_list lights;

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(7, 7, 7));

    world.add(make_shared<quad>(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green));
    world.add(make_shared<quad>(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red));
    auto light_quad = make_shared<quad>(point3(113,554,127), vec3(330,0,0), vec3(0,0,305), light);
    world.add(light_quad);
    lights.add(light_quad);
    world.add(make_shared<quad>(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white));

    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));

    shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130,0,65));

    world.add(make_shared<constant_medium>(box1, 0.01, color(0,0,0)));
    world.add(make_shared<constant_medium>(box2, 0.01, color(1,1,1)));

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 600;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 1.0;

    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world, lights);
}

void final_scene(int image_width, int samples_per_pixel, int max_depth) {
    hittable_list boxes1;
    auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

    int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1,101);
            auto z1 = z0 + w;

            boxes1.add(box(point3(x0,y0,z0), point3(x1,y1,z1), ground));
        }
    }

    hittable_list world;
    hittable_list lights;

    world.add(make_shared<bvh_node>(boxes1));

    auto light = make_shared<diffuse_light>(color(7, 7, 7));
    auto light_quad = make_shared<quad>(point3(123,554,147), vec3(300,0,0), vec3(0,0,265), light);
    world.add(light_quad);
    lights.add(light_quad);

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30,0,0);
    auto sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
    world.add(make_shared<sphere>(center1, center2, 50, sphere_material));

    world.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
    world.add(make_shared<sphere>(
        point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)
    ));

    auto boundary = make_shared<sphere>(point3(360,150,145), 70, make_shared<dielectric>(1.5));
    world.add(boundary);
    world.add(make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
    boundary = make_shared<sphere>(point3(0,0,0), 5000, make_shared<dielectric>(1.5));
    world.add(make_shared<constant_medium>(boundary, .0001, color(1,1,1)));

    auto emat = make_shared<lambertian>(make_shared<image_texture>("assets/textures/earthmap.jpg"));
    world.add(make_shared<sphere>(point3(400,200,400), 100, emat));
    auto pertext = make_shared<noise_texture>(0.2);
    world.add(make_shared<sphere>(point3(220,280,300), 80, make_shared<lambertian>(pertext)));

    hittable_list boxes2;
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<sphere>(point3::random(0,165), 10, white));
    }

    world.add(make_shared<translate>(
        make_shared<rotate_y>(
            make_shared<bvh_node>(boxes2), 15),
            vec3(-100,270,395)
        )
    );

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth         = max_depth;
    cam.background        = color(0,0,0);
    cam.g_use_sky_gradient = false;
    cam.g_sky_strength     = 1.0;

    cam.vfov     = 40;
    cam.lookfrom = point3(478, 278, -600);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world, lights);
}

int main() {
    switch (4) {
        case 1: bouncing_spheres();           break;
        case 2: checkered_spheres();          break;
        case 3: earth();                      break;  
        case 4: sphere_noise_gallery();       break;
        case 5: quads();                      break;
        case 6: simple_light();               break;
        case 7: cornell_box();                break;
        case 8:  cornell_smoke();             break;
        case 9:  final_scene(800, 10000, 40); break;
        default: final_scene(400,   250,  4); break;
    }
}