#version 450

// Interpolated across the triangle by the rasterizer: each fragment receives
// a blend of the three vertex colours weighted by its position within the
// triangle. Location 0 pairs with the vertex shader's `out`.
layout(location = 0) in vec3 fragColor;

// Location 0 here means something different -- it is a framebuffer ATTACHMENT
// index, not a stage interface slot. It binds to the render pass's colour
// attachment in step 6a.
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
