#version 450

// Phase 2, step 4a -- vulkan-tutorial.com "Shader modules".
//
// The tutorial's triangle, with its three vertices hardcoded here. No vertex
// buffer exists yet (that is Phase 3), so gl_VertexIndex -- 0, 1, 2 for the
// three invocations of a three-vertex draw -- picks one position each.
vec2 positions[3] = vec2[](
    vec2( 0.0, -0.5),
    vec2( 0.5,  0.5),
    vec2(-0.5,  0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

// Must pair with the fragment shader's `in` at the same location: that
// matching is the entire interface between the two stages.
layout(location = 0) out vec3 fragColor;

void main() {
    // Clip coordinates. Vulkan's Y axis points DOWN, unlike OpenGL's, so the
    // -0.5 vertex above is the TOP of the screen. Vulkan's depth range is
    // also [0,1] rather than [-1,1]. Neither matters for a flat triangle;
    // both matter when the camera maths arrives in Phase 6.
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
