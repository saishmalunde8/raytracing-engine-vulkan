#version 450

// Phase 3, step 2b -- vulkan-tutorial.com "Vertex input description".
//
// The three hardcoded vertices are gone. Each invocation is now HANDED one
// vertex's attributes, fetched from the bound vertex buffer by fixed-function
// hardware that ran before this shader did, using the stride and offsets the
// pipeline was given at creation. gl_VertexIndex is no longer needed: there
// is nothing left here to index into.
//
// These location numbers are the entire contract with the C++ side. They must
// match the VkVertexInputAttributeDescription entries exactly. The names
// inPosition and inColor are matched against nothing at all.
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

// A DIFFERENT location space from the two above: this 0 is a vertex-to-
// fragment interface slot, not vertex attribute slot 0. It pairs with the
// fragment shader's `in` at the same location, and that pairing is the entire
// interface between the two stages.
layout(location = 0) out vec3 fragColor;

void main() {
    // Clip coordinates. Vulkan's Y axis points DOWN, unlike OpenGL's, so the
    // vertex at y = -0.5 is the TOP of the screen. Vulkan's depth range is
    // also [0,1] rather than [-1,1]. Neither matters for a flat triangle;
    // both matter when the camera maths arrives in Phase 6.
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
