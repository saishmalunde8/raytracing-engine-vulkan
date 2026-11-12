#version 450

// Phase 5, step 3 -- vulkan-tutorial.com "Combined image sampler".
//
// The first time this shader reaches for memory. Until now it was handed an
// interpolated colour and did arithmetic on it; now it DEREFERENCES a
// descriptor, exactly as shader.vert has done for the uniform block since
// Phase 4.
//
// Interpolated across the triangle by the rasterizer: each fragment receives a
// blend of the corner values weighted by its position within the triangle.
// Location 1 pairs with the vertex shader's `out` at location 1.
layout(location = 1) in vec2 fragTexCoord;

// sampler2D is ONE thing here and TWO on the C++ side: binding 1 holds a
// COMBINED_IMAGE_SAMPLER, which is an image view and a sampler in a single
// descriptor. The separate form exists -- texture2D plus sampler, joined as
// sampler2D(tex, samp) -- and avoids N*M descriptors for N images and M
// samplers, at a possible cost on hardware that fuses the two.
//
// This 1 is a DESCRIPTOR BINDING, and the 1 on fragTexCoord above is a stage
// interface slot. Different namespaces, and the file now has all four in play
// at once: vertex attribute location, stage interface location, descriptor
// binding, and the vertex buffer's binding slot.
//
// Nothing here says which image or which sampler. That is the point of a
// descriptor: the shader names a slot, and what the slot resolves to was
// decided on the host by vkUpdateDescriptorSets.
layout(binding = 1) uniform sampler2D texSampler;

// Location 0 here means something different again -- it is a framebuffer
// ATTACHMENT index, not a stage interface slot.
layout(location = 0) out vec4 outColor;

void main() {
    // The texture alone, where the tutorial offers multiplying by the
    // interpolated vertex colour. One variable at a time: the colours on
    // screen are then exactly the texture's, so "is this the right image, at
    // the right scale" is answerable by eye. A tinted checkerboard is not.
    //
    // texture() does the sampling the VkSampler describes -- LINEAR blending
    // of the four nearest texels, REPEAT outside [0,1]. The coordinates stay
    // inside [0,1] here, so the address mode never comes up; the filtering
    // does, and softens every square boundary.
    outColor = texture(texSampler, fragTexCoord);
}
