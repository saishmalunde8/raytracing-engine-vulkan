#version 450

// Phase 5, step 6 -- no tutorial page. The fullscreen triangle.
//
// This shader has NO INPUTS. No `in` declarations, no uniform block, no
// descriptor of any kind. The pipeline declares zero vertex bindings and zero
// vertex attributes, there is no vertex buffer bound, and the draw call is
// vkCmdDraw(3, 1, 0, 0).
//
// What makes that work is gl_VertexIndex. A draw call decides how many times
// the vertex shader runs, and gl_VertexIndex is simply the loop counter handed
// to each invocation. Nothing anywhere requires that counter to address
// memory -- using it to index a vertex buffer is a convention, not a rule.
// Here it indexes a three-element constant instead.
//
// This is the shape a path tracer wants. The image is computed by a compute
// shader; the graphics pipeline exists only to get it onto the screen, and
// geometry has no part in that.

// Location 0 now, renumbered from 1 because fragColor went with the quad's
// vertex colours. The number has to match shader.frag's `in`, and nothing
// outside the pipeline-creation check verifies that it does.
layout(location = 0) out vec2 fragTexCoord;

void main() {
    // A TRIANGLE BIGGER THAN THE SCREEN. These three points cover the whole
    // [-1,1] clip square with one primitive:
    //
    //      (-1,-1) *-------*  (1,-1)       the visible square is the
    //              |      /|                top-left quarter of the
    //              |     / |                triangle; the hardware clips
    //       (-1,1) *----*  |                the rest away for free
    //              |  /    |
    //              | /     |
    //              |/      |
    //      (-1,3)  *-------*  (3,-1) ... (the hypotenuse runs between them)
    //
    // A quad would need two triangles, six indices or an index buffer, and a
    // diagonal seam straight across the middle of the image -- where the
    // rasterizer's derivatives are discontinuous. One triangle has no seam.
    //
    // The winding is clockwise in Vulkan's y-down framebuffer space, matching
    // the quad's and surviving the VK_CULL_MODE_BACK_BIT the rasterizer still
    // has set. Reverse these three and the window goes blank with a perfectly
    // clean validation log.
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 position = positions[gl_VertexIndex];

    // Straight through. No model matrix, because there is nothing to
    // transform: these coordinates are already clip space, which is what the
    // rotating quad's matrix was producing from its [-0.5, 0.5] positions.
    gl_Position = vec4(position, 0.0, 1.0);

    // Clip space [-1,1] to texture space [0,1]. At the three corners this
    // gives (0,0), (2,0) and (0,2); interpolated across the visible square it
    // spans exactly [0,1], and the off-screen excess is clipped with
    // everything else.
    //
    // NO FLIP, and this is where step 3's deferred question gets answered.
    // Vulkan's clip space has y pointing DOWN and its texture space has (0,0)
    // at the TOP-LEFT, so the two agree: clip (-1,-1) is the top-left corner
    // of the window and maps to texel (0,0) of the image. The compute shader
    // wrote texel (0,0) from invocation (0,0), where red and green are both
    // zero -- so the dark corner lands top-left. Flip the v here and the
    // gradient renders upside down with nothing reported.
    fragTexCoord = (position + 1.0) * 0.5;
}
