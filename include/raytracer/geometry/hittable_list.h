#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "raytracer/acceleration/aabb.h"
#include "hittable.h"

#include <memory>
#include <vector>

using std::make_shared;
using std::shared_ptr;

class hittable_list : public hittable {
  public:
    std::vector<shared_ptr<hittable>> objects;

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    void add(shared_ptr<hittable> object) {
        objects.push_back(object);
        bbox = aabb(bbox, object->bounding_box());
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto& object : objects) {
            if ((*object).hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
    
    aabb bounding_box() const override { return bbox; }

    // Sampling a list means picking one member at random and sampling that, so
    // the list's density is the average of its members' densities. Averaging is
    // what keeps it a valid density: each member is chosen 1/n of the time, so
    // each contributes 1/n of its own likelihood.
    //
    // Both operations guard the empty case. An empty light list is normal --
    // four of the scenes have no emitters at all -- and must report "cannot be
    // sampled" rather than divide by zero.
    double pdf_value(const point3& origin, const vec3& direction) const override {
        if (objects.empty()) return 0.0;

        auto weight = 1.0 / objects.size();
        auto sum = 0.0;

        for (const auto& object : objects)
            sum += weight * object->pdf_value(origin, direction);

        return sum;
    }

    vec3 random(const point3& origin, RNG& rng) const override {
        if (objects.empty()) return vec3(1, 0, 0);

        auto int_size = int(objects.size());
        return objects[int(rng.next_double() * int_size)]->random(origin, rng);
    }

  private:
    aabb bbox;
};

#endif