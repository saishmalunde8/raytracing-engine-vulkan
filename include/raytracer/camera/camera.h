#include <atomic>
#include <thread>
#include <chrono>

#ifndef CAMERA_H
#define CAMERA_H

#include "raytracer/geometry/hittable.h"
#include "raytracer/geometry/hittable_list.h"
#include "raytracer/sampling/pdf.h"
#include "raytracer/materials/material.h"
#include "raytracer/renderer/color.h"
#include "raytracer/core/rtweekend.h"
#include "raytracer/core/rng.h"

class camera {
    public:

        double aspect_ratio      = 1.0;  // Ratio of image width over height
        int    image_width       = 100;  // Rendered image width in pixel count
        int    samples_per_pixel = 10;   // Count of random samples for each pixel
        int    max_depth         = 10;   // Maximum number of ray bounces into scene
        color  background = color(0.70, 0.80, 1.00); // Scene background color
        bool   g_use_sky_gradient = true;
        double g_sky_strength     = 1.0; // 0 = background only, 1 = full sky
        bool deterministic = false;

        // Upper bound on worker threads. The renderer still never asks for more
        // than the hardware reports, so this only ever caps that number. The
        // default matches what the renderer has always used.
        //
        // Deterministic renders are unaffected by this: seeding is per pixel and
        // per sample, so the same scene produces the same image at any thread
        // count. Adjustable so that property can actually be checked, and so the
        // single-threaded benchmark figure can be reproduced.
        unsigned int max_threads = 4;

        double vfov     = 90;  // Vertical view angle (field of view)
        point3 lookfrom = point3(0,0,0);   // Point camera is looking from
        point3 lookat   = point3(0,0,-1);  // Point camera is looking at
        vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

        double defocus_angle = 0;  // Variation angle of rays through each pixel
        double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus
      
        struct Tile {
            int x0;
            int x1;
            int y0;
            int y1;
        };

        // `lights` holds the shapes worth aiming rays at. Like `world` it is
        // read-only for the whole render, so worker threads share it without
        // synchronisation. An empty list is valid and means "no lights to aim at".
        void render(const hittable& world, const hittable_list& lights) {
            initialize();
    // ------------------------------------------------------------------------------------                  create framebuffer
            // 1. Create framebuffer (width × height pixels)
            std::vector<color> framebuffer(image_width * image_height);
    // ------------------------------------------------------------------------------------                  create tiles and store them in an array
            constexpr int TILE_SIZE = 16;

            std::vector<Tile> tiles;
    
            // Generate tiles covering the entire image
            for (int y = 0; y < image_height; y += TILE_SIZE) {
                for (int x = 0; x < image_width; x += TILE_SIZE) {

                    Tile tile;
                    tile.x0 = x;
                    tile.y0 = y;

                    // Clamp tile bounds to image size
                    tile.x1 = std::min(x + TILE_SIZE, image_width);
                    tile.y1 = std::min(y + TILE_SIZE, image_height);

                    tiles.push_back(tile);
                }
            }
    // ------------------------------------------------------------------------------------                  create atomic counter and create worker function
            std::atomic<size_t> next_tile_index{0};

            auto worker = [&]() {
                while (true) {
                    size_t tile_index = next_tile_index.fetch_add(1);

                    if (tile_index >= tiles.size())
                        break;

                    const Tile& tile = tiles[tile_index];

                    render_region(
                        tile.x0, tile.x1,
                        tile.y0, tile.y1,
                        world,
                        lights,
                        framebuffer
                    );
                }
            };
    // ------------------------------------------------------------------------------------                 find out number of threads
            unsigned int hw_threads = std::thread::hardware_concurrency();
            if (hw_threads == 0) hw_threads = 4;

            unsigned int thread_count = std::min(hw_threads, max_threads);
    // ------------------------------------------------------------------------------------                 start render clock
            auto render_start = std::chrono::high_resolution_clock::now();
    // ------------------------------------------------------------------------------------                 create threads array and run threads
            std::vector<std::thread> threads;   
            threads.reserve(thread_count);

            for (unsigned int t = 0; t < thread_count; t++) {
                threads.emplace_back(worker);
            }
    // ------------------------------------------------------------------------------------                 start progress bar
            const size_t total_tiles = tiles.size();
    
            while (true) {
                size_t done = next_tile_index.load();
                if (done >= total_tiles)
                    break;

                double percent = 100.0 * done / total_tiles;

                std::clog << "\rRendering: "
                        << static_cast<int>(percent) << "% "
                        << std::flush;

                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
    // ------------------------------------------------------------------------------------                 join threads
            for (auto& t : threads) {
                t.join();
            }
    // ------------------------------------------------------------------------------------                 stop render clock and add render benchmark
            std::clog << "\rRendering: 100%        \n";     

            auto render_end = std::chrono::high_resolution_clock::now();

            auto elapsed = std::chrono::duration<double>(render_end - render_start).count();

            double total_samples = double(image_width) * double(image_height) * double(samples_per_pixel);

            double samples_per_second = total_samples / elapsed;

            std::clog << "\n=== Render Benchmark ===\n";
            std::clog << "Resolution: " << image_width << " x " << image_height << "\n";
            std::clog << "Samples per pixel: " << samples_per_pixel << "\n";
            std::clog << "Total samples: " << total_samples << "\n";
            std::clog << "Threads: " << thread_count << "\n";
            std::clog << "Render time: " << elapsed << " seconds\n";
            std::clog << "Samples/sec: " << samples_per_second << "\n";
    // ------------------------------------------------------------------------------------                  output framebuffer to image ppm
            // 4. Output framebuffer AFTER rendering
            std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

            for (int j = 0; j < image_height; j++) {
                for (int i = 0; i < image_width; i++) {
                    int index = j * image_width + i;
                    write_color(std::cout, framebuffer[index]);
                }
            }
        }
    // ------------------------------------------------------------
  
    private:
        int    image_height;   // Rendered image height
        double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
        point3 center;         // Camera center
        point3 pixel00_loc;    // Location of pixel 0, 0
        vec3   pixel_delta_u;  // Offset to pixel to the right
        vec3   pixel_delta_v;  // Offset to pixel below
        vec3   u, v, w;              // Camera frame basis vectors
        vec3   defocus_disk_u;       // Defocus disk horizontal radius
        vec3   defocus_disk_v;       // Defocus disk vertical radius

        void render_region(
            int x0, int x1,
            int y0, int y1,
            const hittable& world,
            const hittable_list& lights,
            std::vector<color>& framebuffer) {
            for (int j = y0; j < y1; j++) {
                for (int i = x0; i < x1; i++) {
                    color pixel_color(0,0,0);
                    for (int sample = 0; sample < samples_per_pixel; sample++) {
                        if (deterministic) {
                            // Seeded per pixel and per sample, so the result does not
                            // depend on tile order, scheduling, or thread count.
                            RNG rng(pixel_sample_seed(i, j, sample));
                            pixel_color += sample_pixel(i, j, world, lights, rng);
                        } else {
                            pixel_color += sample_pixel(i, j, world, lights, thread_rng());
                        }
                    }

                    int index = j * image_width + i;
                    framebuffer[index] = pixel_samples_scale * pixel_color;
                }
            }
        }

        // The entire per-sample path. Both seeding strategies run through this,
        // so there is only one implementation of a sample to keep correct.
        color sample_pixel(int i, int j, const hittable& world, const hittable_list& lights,
                           RNG& rng) const {
            ray r = get_ray(i, j, rng);
            return ray_color(r, max_depth, world, lights, rng);
        }

        void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;
    
            pixel_samples_scale = 1.0 / samples_per_pixel;

            center = lookfrom;
    
            // Determine viewport dimensions.
            auto theta = degrees_to_radians(vfov);
            auto h = std::tan(theta/2);
            auto viewport_height = 2 * h * focus_dist;
            auto viewport_width = viewport_height * (double(image_width)/image_height);
    
            // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            // Calculate the vectors across the horizontal and down the vertical viewport edges.
            vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
            vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge
    
            // Calculate the horizontal and vertical delta vectors from pixel to pixel.
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;
    
            // Calculate the location of the upper left pixel.
            auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

            // Calculate the camera defocus disk basis vectors.
            auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }
    
        // Construct a camera ray originating from the defocus disk and directed at a
        // randomly sampled point around the pixel location i, j. The three draws
        // below happen in a fixed order -- pixel offset, then defocus disk, then
        // shutter time -- because that order defines the random stream a
        // deterministic render reproduces.
        ray get_ray(int i, int j, RNG& rng) const {

            auto offset = sample_square(rng);

            auto pixel_sample = pixel00_loc
                            + ((i + offset.x()) * pixel_delta_u)
                            + ((j + offset.y()) * pixel_delta_v);

            auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(rng);

            auto ray_direction = pixel_sample - ray_origin;
            auto ray_time = rng.next_double();

            return ray(ray_origin, ray_direction, ray_time);
        }

        vec3 sample_square(RNG& rng) const {
            // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
            return vec3(rng.next_double() - 0.5,
                        rng.next_double() - 0.5,
                        0);
        }

        // Returns a random point in the camera defocus disk, sampled in polar form
        // so it consumes exactly two draws. A rejection-sampled disk would consume a
        // variable number and desynchronise the stream.
        point3 defocus_disk_sample(RNG& rng) const {
            auto r = std::sqrt(rng.next_double());
            auto theta = 2 * pi * rng.next_double();
            auto x = r * std::cos(theta);
            auto y = r * std::sin(theta);
            return center + x * defocus_disk_u + y * defocus_disk_v;
        }

        color ray_color(const ray& r, int depth, const hittable& world,
                        const hittable_list& lights, RNG& rng) const {
            if (depth <= 0)
                return color(0,0,0);

            hit_record rec;
            
            if (world.hit(r, interval(0.001, infinity), rec)) {

                scatter_record srec;

                color color_from_emission =
                    rec.mat->emitted(r, rec, rec.u, rec.v, rec.p);

                if (!rec.mat->scatter(r, rec, srec, rng))
                    return color_from_emission;

                // Specular. The outgoing direction is determined, not chosen, so
                // there is no density to weight against -- follow it directly.
                if (srec.skip_pdf)
                    return color_from_emission
                        + srec.attenuation
                            * ray_color(srec.skip_pdf_ray, depth - 1, world, lights, rng);

                // Everything else: pick a direction, then correct for having
                // picked it that way. The weight is how much the material
                // scatters this way, over how likely we were to choose it.
                //
                // With lights in the scene, half of all directions are aimed
                // straight at one rather than left to the material's own
                // preference -- which is how a surface finds a small bright light
                // instead of waiting to stumble into it. Both halves are scored
                // against the average of the two densities, and that averaging is
                // what stops the aiming from biasing the result.
                shared_ptr<pdf> sampling_pdf;

                if (lights.objects.empty()) {
                    // Nothing to aim at. Mixing with an unsampleable list would
                    // send half of all rays off in a fixed placeholder direction.
                    sampling_pdf = srec.pdf_ptr;
                } else {
                    sampling_pdf = make_shared<mixture_pdf>(
                        make_shared<hittable_pdf>(lights, rec.p),
                        srec.pdf_ptr);
                }

                ray scattered = ray(rec.p, sampling_pdf->generate(rng), r.time());
                auto pdf_value = sampling_pdf->value(scattered.direction());

                double scattering_pdf = rec.mat->scattering_pdf(r, rec, scattered);

                // Two directions carry no energy and must not reach the division
                // below.
                //
                // scattering_pdf <= 0 means the material does not scatter this
                // way at all -- an aimed ray pointing at a light that sits behind
                // the surface, most often. Its contribution is genuinely zero.
                //
                // A vanishing pdf_value means the sampler could not meaningfully
                // have produced this direction, usually a hit test failing at a
                // grazing angle. Dividing by it turns one sample into an enormous
                // weight, and that pixel stays blown out no matter how many more
                // samples are thrown at it.
                //
                // Together these are also the 0/0 case, which yields NaN. The
                // second test is written as a negated `>` so a NaN pdf, which
                // compares false against everything, falls through it too.
                //
                // Returning here also skips the recursive trace, so a ray that
                // could not have contributed costs nothing to reject.
                if (scattering_pdf <= 0 || !(pdf_value > 1e-8))
                    return color_from_emission;

                color sample_color = ray_color(scattered, depth - 1, world, lights, rng);
                color color_from_scatter =
                    (srec.attenuation * scattering_pdf * sample_color) / pdf_value;

                return color_from_emission + color_from_scatter;
            }

            // ---------- MISS (background + gradient sky) ----------

            color base_bg = background;

            if (!g_use_sky_gradient)
                return base_bg;

            vec3 unit_direction = unit_vector(r.direction());
            double t = interval(0.0, 1.0).clamp(0.35 * (unit_direction.y() + 1.0));

            color horizon = color(1.00, 0.68, 0.45);
            color zenith  = color(0.45, 0.55, 0.75);

            color sky = (1.0 - t) * horizon + t * zenith;

            // Blend the gradient over the flat background rather than replacing
            // it, so g_sky_strength can dial the sky from "background only" to
            // "full dusk" -- a night sky wants a hint of horizon glow, not the
            // whole gradient at full strength.
            return (1.0 - g_sky_strength) * base_bg + g_sky_strength * sky;
        }

// --------------------------
            // if (world.hit(r, interval(0.001, infinity), rec)) {
            //     ray scattered;
            //     color attenuation;
            //     if (rec.mat->scatter(r, rec, attenuation, scattered))
            //         return attenuation * ray_color(scattered, depth-1, world);
            //     return color(0,0,0);
            // }

            // vec3 unit_direction = unit_vector(r.direction());
            // auto a = 0.5*(unit_direction.y() + 1.0);

            // color color_1 = color(1.0, 1.0, 1.0);
            // color color_2 = color(0.5, 0.7, 1.0);
        
            // return (1.0 - a) * color_1 + a * 0.8 * color_2;
};

#endif