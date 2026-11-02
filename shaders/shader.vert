#version 450

// Phase 4, step 2 -- vulkan-tutorial.com "Descriptor pool and sets".
//
// The first time this shader REACHES for memory instead of being handed it.
//
// The two `in` variables below arrive the way they have since Phase 3: a
// fixed-function fetch ran before this shader did, walked the bound vertex
// buffer with the stride and offsets baked into the pipeline, and handed this
// invocation its attributes. Nothing here asked for them. That is the one case
// in Vulkan where no descriptor is involved, and it is misleading because it
// is the first case anyone meets.
//
// The uniform block is the other kind. Nothing hands it over; this shader
// dereferences binding 0 of descriptor set 0, which resolves to real memory
// only because the C++ side allocated a set against a matching layout, wrote a
// buffer's address into it, and bound it before the draw. Every resource the
// path tracer needs -- BVH nodes, primitives, materials, the camera, the
// accumulation and output images -- arrives exactly this way.
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

// binding = 0 is the ENTIRE contract with the C++ side, as `location` is for
// the attributes above. The block name UniformBufferObject and the instance
// name ubo are matched against nothing at all -- the C++ struct of the same
// name is never handed to Vulkan, only its size is. Agreement between the two
// layouts is on us, and a mismatch is silent: the bytes are simply read as
// something else.
//
// A third namespace, too: this 0 is a descriptor binding, unrelated to vertex
// attribute location 0 and to the vertex buffer's binding slot 0.
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
} ubo;

// A DIFFERENT location space from the attributes above: this 0 is a vertex-to-
// fragment interface slot, not vertex attribute slot 0. It pairs with the
// fragment shader's `in` at the same location, and that pairing is the entire
// interface between the two stages.
layout(location = 0) out vec3 fragColor;

void main() {
    // Clip coordinates, now by way of a matrix that came from host memory this
    // frame rather than straight from the vertex buffer. ubo.model is identity
    // in this step, so the result is bit-for-bit what the previous step
    // produced -- which is the checkpoint, and a demanding one: an unwritten
    // buffer would be a ZERO matrix, collapsing all four corners onto the
    // origin and drawing nothing at all.
    //
    // Vulkan's Y axis points DOWN, unlike OpenGL's, so the vertex at y = -0.5
    // is the TOP of the screen, and the depth range is [0,1] rather than
    // [-1,1]. Neither matters for a flat quad under an identity transform;
    // both matter once the camera maths arrives in Phase 6.
    gl_Position = ubo.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
