#ifndef ONB_H
#define ONB_H

#include "vec3.h"

// An orthonormal basis built around a surface normal.
//
// Direction sampling is far easier to reason about in a coordinate frame where
// the normal is +Z: a cosine-weighted hemisphere sample, for instance, is two
// draws and a square root. This class builds such a frame for an arbitrary
// normal, so a direction can be generated in the simple frame and then rotated
// into world space with transform().
class onb {
  public:
    onb(const vec3& n) {
        axis[2] = unit_vector(n);

        // Any vector not parallel to the normal works as a seed for the cross
        // product. Picking the axis the normal leans on least keeps the two
        // vectors comfortably apart, so the cross product stays well defined.
        vec3 a = (std::fabs(axis[2].x()) > 0.9) ? vec3(0,1,0) : vec3(1,0,0);

        axis[1] = unit_vector(cross(axis[2], a));
        axis[0] = cross(axis[2], axis[1]);
    }

    const vec3& u() const { return axis[0]; }
    const vec3& v() const { return axis[1]; }
    const vec3& w() const { return axis[2]; }

    // Transform a direction from basis coordinates into world space.
    vec3 transform(const vec3& v) const {
        return (v[0] * axis[0]) + (v[1] * axis[1]) + (v[2] * axis[2]);
    }

  private:
    vec3 axis[3];
};

#endif
