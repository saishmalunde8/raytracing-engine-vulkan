#version 450

// Phase 5, step 3 -- vulkan-tutorial.com "Combined image sampler".
//
// Three `in` variables now, and the third does nothing here but travel. A
// vertex shader cannot sample a texture usefully for this job: the texture is
// read PER FRAGMENT, and all this stage can do is hand each corner's
// coordinate to the rasterizer, which interpolates it across the triangle the
// same way it has interpolated colour since Phase 3. Interpolation is the
// whole mechanism -- four corner coordinates become a coordinate for every
// pixel, for free, in fixed-function hardware.
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

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

// A DIFFERENT location space from the attributes above: these are vertex-to-
// fragment interface slots. They pair with the fragment shader's `in` at the
// same location, and that pairing is the entire interface between the stages.
//
// Nothing consumes fragColor as of this step -- the fragment shader outputs
// the texture alone, so the vertex colours are deliberately ignored. An
// unmatched vertex output is legal and simply discarded. It stays because
// removing it means editing the Vertex struct and the attribute descriptions,
// and step 7 deletes the vertex buffer outright.
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    // Clip coordinates, by way of a matrix that came from host memory this
    // frame rather than straight from the vertex buffer.
    //
    // Vulkan's Y axis points DOWN, unlike OpenGL's, so the vertex at y = -0.5
    // is the TOP of the screen, and the depth range is [0,1] rather than
    // [-1,1]. The same downward Y is why texture coordinate (0,0) is the
    // top-left here, which is what lets the corner at y = -0.5 take v = 0
    // with no flip anywhere in the pipeline.
    gl_Position = ubo.model * vec4(inPosition, 0.0, 1.0);

    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
