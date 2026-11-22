#version 450

// Phase 5, step 6 -- sampling the compute shader's output.
//
// Unchanged in substance from step 3: one interpolated coordinate in, one
// texture() out. What changed is on the other side of the descriptor -- the
// image this samples was written by a compute dispatch rather than uploaded
// from the host, and nothing in this file can tell the difference. That is the
// point of a descriptor.

// Location 0, matching shader.vert's `out`. Interpolated across the triangle
// by the rasterizer, free and fixed-function.
layout(location = 0) in vec2 fragTexCoord;

// Binding 0 now, renumbered from 1 because the uniform block that held
// binding 0 went with the quad. The number must match the C++ side's
// VkDescriptorSetLayoutBinding::binding, in another language and another file,
// and nothing checks that it does.
//
// sampler2D, not image2D. The SAME VkImage is bound to the compute shader as a
// writable image2D and to this shader as a readable sampler2D, through two
// different descriptor types in two different set layouts. That is legal
// because the image's usage flags declared both STORAGE and SAMPLED, and
// because VK_IMAGE_LAYOUT_GENERAL is the one layout that permits both.
layout(binding = 0) uniform sampler2D srcImage;

layout(location = 0) out vec4 outColor;

void main() {
    // The sampler's rules still apply, and now they do visible work: the
    // storage image is 800x600 and the window is whatever size it is, so
    // VK_FILTER_LINEAR is interpolating between texels on almost every pixel.
    // That is deliberate -- render resolution is decoupled from window size,
    // which is what a path tracer wants, and the filter is what makes the
    // mismatch look like a rescale rather than a grid.
    outColor = texture(srcImage, fragTexCoord);
}
