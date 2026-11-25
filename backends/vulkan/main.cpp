// Vulkan backend entry point.
//
// Phase 6, step 2 -- the camera uniform and ray generation.
//
// Step 1 gave the shader random numbers and nothing to spend them on. This
// step spends them: every invocation builds a CAMERA RAY for its pixel,
// jittered within the pixel and leaving from a random point on the lens. It
// is the first object here a path tracer would recognise.
//
// Nothing is intersected yet. There is no sphere and no material, and the
// window shows each ray's direction as a colour -- a smooth two-axis
// gradient, which is a ray field and not a scene.
//
// WHERE THE HOST/DEVICE LINE FALLS, which is most of what porting a renderer
// to a GPU consists of. camera::initialize() in camera.h turns six scene
// parameters -- vfov, lookfrom, lookat, vup, defocus_angle, focus_dist --
// into six vectors. camera::get_ray() uses only those six vectors. So
// initialize() stays on the host and runs once, its results are uploaded, and
// get_ray() becomes the shader. Nothing about the field of view or the basis
// is recomputed 480,000 times a frame, and the uniform block is small enough
// to be a uniform block because of where that line was drawn.
//
// THE ARITHMETIC IS IN DOUBLE UNTIL THE UPLOAD. The CPU computes the basis in
// double, so this does too, and the single narrowing to float happens as the
// values enter the block the shader reads. Computing in float instead would
// round every intermediate, and the two backends would then differ by more
// than one rounding with nothing to say which was closer to right.
//
// THE DIRECTION IS NOT NORMALISED, and that is faithfulness rather than an
// oversight. The CPU returns pixel_sample - ray_origin as it stands, and
// sphere::hit consumes it with the full quadratic, where a is dot(d, d)
// rather than 1. Normalising would still intersect correctly but would
// rescale every t, so the shadow epsilon and the ray extents would mean
// different distances on the two backends. The display normalises a local
// copy and throws it away.
//
// THE CHECKPOINT CHANGES CHARACTER HERE, and that is the step's real lesson.
// Step 1's generator was integer arithmetic and a power-of-two scale, so the
// host could predict every bit and the checkpoint was exact. A camera ray
// involves divisions, a square root and a sine, and Vulkan permits a couple
// of units in the last place on those -- so exactness is gone and will not
// come back for the rest of the port.
//
// A tolerance alone would be a weak replacement, because a tolerance only
// says the two sides are close and says nothing about whether either is
// right. So the tolerance is backed by INVARIANTS -- statements that hold
// exactly in geometry, that a merely-close ray field fails:
//
//   PLANARITY. Every ray direction must have a dot product with w of exactly
//   -focus_dist, because the viewport is a plane perpendicular to w at that
//   distance and the lens offset lies in the u-v plane. One number per pixel,
//   and it pins the basis orthogonality, the focus distance and the defocus
//   disk's orientation simultaneously. A camera that was skewed, mis-scaled
//   or had its disk built from the wrong vectors fails it.
//
//   SYMMETRY. Pixel-centre directions for mirrored pairs must sum to exactly
//   -2 * focus_dist * w. This catches a flipped viewport axis, which
//   planarity cannot: an upside-down camera is still perfectly planar.
//
// Neither is visible. A direction-as-colour gradient looks correct for a
// camera that is skewed, inverted, or pointing somewhere else entirely.
//
// Eight traps:
//
//   1. CONSTANTS THAT MUST AGREE ACROSS THE HOST/SHADER BOUNDARY, WITH
//      NOTHING CHECKING THAT THEY DO. WORKGROUP_SIZE against local_size_x/y
//      was the first; the hash constants added a pair of algorithms written
//      twice; this step adds getRay, written twice, and the std140 offsets.
//      The offsets are the one case with a real mitigation -- the
//      static_asserts on CameraUniform turn a silent mismatch into a compile
//      error -- and that is the pattern to reach for whenever the boundary
//      can be stated as a number.
//   2. FORMAT FEATURES ARE NOT GUARANTEED. What a VkFormat may be used for
//      varies by device and by tiling, and must be queried.
//      R32G32B32A32_SFLOAT is probed ahead of Phase 7 needing it, and it
//      reports no STORAGE_IMAGE_ATOMIC -- as R8G8B8A8_UNORM does -- so
//      accumulation cannot use atomics on this device in any format.
//   3. VkClearColorValue IS float32 REGARDLESS OF THE IMAGE'S BIT DEPTH.
//      The union has an integer member and it is the wrong one for a UNORM
//      image: 0-255 bytes go in as 0.0-1.0 floats.
//   4. RENDER RESOLUTION IS NOT WINDOW SIZE. The storage image is fixed at
//      800x600 and stretched across whatever the window is. It is also what
//      the camera's aspect ratio is derived from, which is the right source:
//      the viewport has to match the pixel grid being sampled, not the window
//      the result is displayed in.
//   5. A MISSING BARRIER IS STILL NOT VISUALLY VERIFIABLE. On unified-memory
//      hardware with a single queue, removing either barrier around the
//      dispatch would very probably still look correct, because the hazard it
//      guards is one this device happens not to expose. Correct by accident
//      on the machine it was written on. No readback can prove it -- the
//      reasoning in recordStorageImageBarrier is the proof, and the access
//      masks are what a reviewer checks.
//   6. THE NAIVE uint-TO-FLOAT CONVERSION RETURNS 1.0, and [0,1) is a
//      contract the samplers depend on. float(u) * (1.0/4294967296.0) gives
//      exactly 1.0 for every u from 2^32 - 128 upward, about one draw in
//      2^25. Downstream, 1.0 reaches sqrt(1 - r2) as exactly zero and a
//      1/(1-r) elsewhere as infinity. The top 24 bits remove the case
//      instead of narrowing it.
//   7. HOW MANY DRAWS A RAY CONSUMES IS NOT CONSTANT. get_ray takes two for
//      the jitter, two for the defocus disk ONLY IF defocus_angle > 0, and
//      one for the shutter time always -- five at this camera, three at a
//      camera with no lens. Two ways to get this wrong, both silent: drawing
//      the defocus pair unconditionally desynchronises every camera without a
//      lens, and skipping the shutter draw because nothing reads it yet
//      shifts every later draw by one. The stream POSITION is shared
//      semantics with the CPU backend, so a draw is taken whether or not
//      there is anything to spend it on.
//   8. NEITHER GLSL NOR C++ DEFINES THE EVALUATION ORDER OF AN EXPRESSION'S
//      OPERANDS, so vec2(nextFloat() - 0.5, nextFloat() - 0.5) does not say
//      which draw becomes x. Both values are random and both get used, so a
//      compiler evaluating right to left transposes the jitter and nothing
//      looks wrong -- the antialiasing is still correct, off a different
//      stream. Step 1's four-draw probe had this shape and its bit-exact
//      comparison passed, which means only that this toolchain happens to go
//      left to right. Every draw is now sequenced into a named local on both
//      sides. The CPU's sample_square still has the original form; it is in
//      include/, which is synced, so it is flagged rather than fixed.
//
// Next: sphere intersection and the sky on a miss, which turns this ray field
// into the first image with an object in it.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Angles in radians everywhere, before glm is included. GLM's historical
// default was degrees and some builds still accept both, so a glm::rotate call
// given 45.0f can silently mean 45 radians instead -- a factor of 57 that
// renders as something plausible rather than as an error. Forcing it is how
// that ambiguity stops existing. Phase 4 is the first phase with any angles in
// it at all; glm::rotate itself arrives with the rotation in step 3, and the
// gtc/matrix_transform.hpp header it needs comes with it.
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

// glm::rotate and the rest of the transform constructors -- glm keeps the base
// types and the matrix builders in separate headers. Declared here rather than
// with the camera in Phase 6 because the rotation below is the first caller.
#include <glm/gtc/matrix_transform.hpp>

// Both glm headers are UNUSED as of step 6: the Vertex struct and the uniform
// block that needed vec2, vec3 and mat4 went with the quad. They stay for two
// reasons -- Phase 6's camera brings every one of those types straight back,
// and removing them would leave the glm dependency in CMakeLists.txt pointing
// at nothing, in a file whose rule is append-only.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// Two is enough for the CPU to stay one frame ahead of the GPU and hide the
// gap. More only adds input-to-photon latency and memory, not throughput.
// Unrelated to the swapchain's image count, which happens to be three.
const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// Device-level extensions, distinct from the instance-level ones.
//
// VK_KHR_portability_subset is the second half of the macOS fix. The instance
// extension in step 2 said "show me incomplete drivers"; this one says "I
// acknowledge this specific device is incomplete". The spec requires enabling
// it on any device that advertises it, and that is easy to miss -- omitting
// it trips validation errors later rather than here.
const std::vector<const char*> deviceExtensions = {
    // Presenting is a per-device capability, so the swapchain is a DEVICE
    // extension -- unlike the instance-level ones enabled in Phase 1.
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    "VK_KHR_portability_subset"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// vkCreateDebugUtilsMessengerEXT belongs to an extension, so the loader does
// not export it and it cannot be called directly. It has to be looked up by
// name at runtime and called through a function pointer.
VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// Any uint32_t is a valid queue family index, including 0, so there is no
// number free to mean "not found". optional carries that separately.
//
// Deviation from the conventional setup, per the roadmap: the usual search is
// for a graphics-only family. This backend submits compute work too, and on
// Apple Silicon one family serves both, so both bits are required here rather
// than revisiting this in Phase 3.
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsAndComputeFamily;

    // Kept separate from the one above because nothing guarantees they are
    // the same family. They happen to be on Apple Silicon; assuming that in
    // the type would be assuming the hardware.
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsAndComputeFamily.has_value() &&
               presentFamily.has_value();
    }
};

// Everything vkCreateSwapchainKHR will need in the next two steps. Each of
// these is a property of a device/surface *pair* rather than of either alone,
// which is why all three queries below take the surface.
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;   // image count and extent bounds
    std::vector<VkSurfaceFormatKHR> formats; // (format, colorSpace) pairs
    std::vector<VkPresentModeKHR> presentModes;
};

// No Vertex struct, no vertex array, no index array and no uniform block as of
// step 6. Each of them existed to feed the rotating quad, and the fullscreen
// triangle feeds itself: three positions from a constant indexed by
// gl_VertexIndex, and nothing to transform them by. The hardware's whole
// vertex-fetch path is simply unused.
//
// What they taught is in the history and not repeated here: stride and
// offsetof propagating a struct change into the read pattern, indices as an
// unordered pool with the post-transform cache behind it, and std140 alignment
// waiting for Phase 6's camera to put a vec3 next to a mat4.

// Phase 6 step 2. THE HOST SIDE OF THE SHADER'S UNIFORM BLOCK, and the
// alignment here is the whole point.
//
// std140 gives a vec3 a base alignment of 16 bytes and a size of 12. C++
// gives glm::vec3 an alignment of 4 and a size of 12. So without alignas the
// members would pack at 0, 12, 24, 36, 48, 60, 72 while the shader reads them
// at 0, 16, 32, 48, 64, 80, 92 -- every vector after the first one wrong, and
// nothing anywhere would say so. The camera would simply point somewhere
// else, which looks exactly like a camera that was configured wrong.
//
// THE FLOAT HAS NO alignas, AND THAT IS NOT AN OVERSIGHT. Giving it one for
// consistency is the mistake this struct exists to demonstrate. A float's
// base alignment in std140 is 4, so it is placed at the next multiple of 4
// after defocusDiskV's twelve bytes end -- offset 92, inside the padding that
// brought defocusDiskV up to 16. Adding alignas(16) would move it to 96 and
// the shader would read defocusDiskV's padding as the defocus angle: four
// bytes of whatever, deciding whether the lens exists.
//
// The static_asserts below are the real mitigation. Every offset the shader
// uses is stated here as a compile-time claim, so the consistent-looking
// version of this struct does not compile rather than rendering a subtly
// wrong image. They were checked against the SPIR-V the shader compiles to,
// not against a reading of the specification.
struct CameraUniform {
    alignas(16) glm::vec3 center;
    alignas(16) glm::vec3 pixel00Loc;
    alignas(16) glm::vec3 pixelDeltaU;
    alignas(16) glm::vec3 pixelDeltaV;
    alignas(16) glm::vec3 defocusDiskU;
    alignas(16) glm::vec3 defocusDiskV;
    float defocusAngle;
};

static_assert(offsetof(CameraUniform, center) == 0,
              "CameraUniform::center must sit at std140 offset 0");
static_assert(offsetof(CameraUniform, pixel00Loc) == 16,
              "CameraUniform::pixel00Loc must sit at std140 offset 16");
static_assert(offsetof(CameraUniform, pixelDeltaU) == 32,
              "CameraUniform::pixelDeltaU must sit at std140 offset 32");
static_assert(offsetof(CameraUniform, pixelDeltaV) == 48,
              "CameraUniform::pixelDeltaV must sit at std140 offset 48");
static_assert(offsetof(CameraUniform, defocusDiskU) == 64,
              "CameraUniform::defocusDiskU must sit at std140 offset 64");
static_assert(offsetof(CameraUniform, defocusDiskV) == 80,
              "CameraUniform::defocusDiskV must sit at std140 offset 80");

// The one that catches the plausible mistake. 92, not 96: the float belongs
// in the preceding vec3's padding.
static_assert(offsetof(CameraUniform, defocusAngle) == 92,
              "CameraUniform::defocusAngle must sit at std140 offset 92 -- in "
              "defocusDiskV's padding. An alignas(16) here would move it to "
              "96 and the shader would read padding as the defocus angle.");

static_assert(sizeof(CameraUniform) == 96,
              "CameraUniform must be 96 bytes: 92 used, rounded up to the "
              "16-byte alignment std140 gives the block");

class vulkan_app {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();

        // THE CHECKPOINT THAT SPEAKS FOR THE PER-FRAME PATH. mainLoop ends
        // in vkDeviceWaitIdle, so every submission has completed and both the
        // image and the probe buffer hold the result of the last recorded
        // frame -- recording immediately precedes submission in drawFrame, and
        // submissions on one queue complete in order.
        //
        // The probe needs no invalidate before it is read: its memory is
        // HOST_COHERENT, so the GPU's writes are visible to the host as soon
        // as the submission that made them completes.
        verifyCameraRays(lastPushedTime, "after the main loop");

        cleanup();
    }

private:
    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;

    // Created from the instance, not the device: the window exists before any
    // GPU has been chosen, and from the next step onward whether a GPU is
    // suitable depends on its ability to present to this surface.
    VkSurfaceKHR surface;

    // Not created and not destroyed -- this is a handle to hardware that
    // already exists. It is released implicitly with the instance, which is
    // why it never appears in cleanup().
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    // Created, owned and destroyed by us, unlike the physical device above.
    VkDevice device;

    // Queues are created with the device and die with it, so there is no
    // destroy call for these -- they are only handles to ones that already
    // exist. On this GPU both names are expected to resolve to the same queue.
    VkQueue graphicsAndComputeQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;

    // Owned by the swapchain, like the physical device is owned by the
    // instance: destroyed implicitly with it, never by us.
    std::vector<VkImage> swapChainImages;

    // Kept because the image views below, the render pass (6a), framebuffers
    // (7) and viewport (5) all need them again.
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    // Ours, unlike swapChainImages: created by us, so destroyed by us.
    std::vector<VkImageView> swapChainImageViews;

    VkRenderPass renderPass;

    // The shaders' uniform interface -- empty for now. Unlike the shader
    // modules, this outlives pipeline creation, so it is freed in cleanup().
    VkPipelineLayout pipelineLayout;

    VkPipeline graphicsPipeline;

    // One per swapchain image: vkAcquireNextImageKHR decides which image is
    // free at draw time, so every one of them needs a framebuffer waiting.
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkCommandPool commandPool;

    // One per frame in flight: the CPU cannot re-record a buffer the GPU is
    // still reading. Freed with the pool, so never in cleanup() by itself.
    std::vector<VkCommandBuffer> commandBuffers;

    // The SHAPE of the resource slots a shader may read -- not the resources.
    // One schema, created once, outlived by nothing: the pipeline layout built
    // from it is the only thing that refers to it.
    VkDescriptorSetLayout descriptorSetLayout;

    // The allocator the sets come out of, sized once at creation and unable to
    // grow -- the same bargain as the command pool, including that its sets are
    // freed WITH it and never individually.
    VkDescriptorPool descriptorPool;

    // ONE set, where step 3 had one per frame in flight. Those existed because
    // each pointed at that frame's uniform buffer; with the uniform buffers
    // gone they would be two identical descriptors naming one image that never
    // changes. The "group by update frequency" rule from step 3, giving the
    // opposite answer on the same code.
    //
    // No destroy call for this; see descriptorPool above.
    VkDescriptorSet descriptorSet;

    // All that survives of the checkerboard texture steps 1 to 3 built. A
    // sampler references no image and never did -- it is filtering and
    // addressing rules and nothing else -- which is precisely why deleting the
    // texture out from under it in step 6 left it intact and immediately
    // reusable for the storage image.
    //
    // The name is kept rather than churned. It samples an image; WHICH image is
    // the descriptor's business, not its own.
    VkSampler textureSampler;

    // Phase 5 step 4. The image a compute shader writes in step 5 and the
    // fullscreen triangle samples in step 6. Three things make it unlike the
    // texture above: its usage declares STORAGE, it rests in GENERAL rather
    // than SHADER_READ_ONLY_OPTIMAL, and no host data is ever uploaded into
    // it -- the GPU is the only thing that will ever produce its contents.
    VkImage storageImage;
    VkDeviceMemory storageImageMemory;

    // Created in this step rather than a later one because an image is
    // unbindable without a view, and step 5's descriptor wants one ready.
    // Nothing about a view says "storage": the usage flag on the IMAGE
    // decided that, and this identical view will serve step 7's sampled read.
    VkImageView storageImageView;

    // Render resolution, deliberately NOT the window size. See trap 4 at the
    // top of the file.
    static const uint32_t RENDER_WIDTH = 800;
    static const uint32_t RENDER_HEIGHT = 600;

    // What the image is cleared to before anything else writes it, and the
    // reason this step has a checkpoint. See THE SENTINEL at the top.
    static constexpr std::array<uint8_t, 4> STORAGE_SENTINEL = {0, 0, 255,
                                                                255};

    // Phase 5 step 5. The compute path, and every object on it is separate
    // from the graphics one. A pipeline layout names its set layouts, so
    // binding 0 in the compute layout below has nothing whatever to do with
    // binding 0 in descriptorSetLayout above -- they are different
    // namespaces, reached through different pipeline layouts.
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;

    // One set, not one per frame in flight. The storage image is written once
    // at startup in this step and the set only ever points at the same image,
    // so there is nothing per-frame about it -- which is the "group by update
    // frequency" rule from step 3 landing on the opposite answer.
    VkDescriptorSet computeDescriptorSet;

    // MUST MATCH local_size_x/y in shader.comp, and nothing checks that it
    // does. The shader bakes its workgroup size into the SPIR-V; this is what
    // the host uses to work out how many groups to ask for. The two numbers
    // live in different languages, in different files, compiled by different
    // compilers, and a disagreement is silent -- see trap 1 at the top.
    static const uint32_t WORKGROUP_SIZE = 8;

    // Phase 6 step 1, widened in step 2. The probe buffer: what the shader
    // produced, at full precision, for the host checkpoint to compare against
    // its own reference. Not a render target and not on the display path.
    //
    // Step 1 held four raw draws per pixel. It now holds the finished ray --
    // two vec4 entries, origin-and-time then direction -- because the ray is
    // computed from those draws and so is strictly stronger evidence: a wrong
    // draw cannot produce a right ray.
    VkBuffer rayProbeBuffer;
    VkDeviceMemory rayProbeMemory;

    // Mapped at creation and left mapped until cleanup. A persistent mapping
    // costs nothing on this hardware, and map/read/unmap around each
    // checkpoint would be three calls to achieve the same thing.
    void* rayProbeMapped = nullptr;

    // MUST MATCH the probe writes in shader.comp. See trap 1 at the top.
    static const uint32_t PROBE_VEC4S_PER_PIXEL = 2;

    // MUST MATCH the number of nextFloat() calls getRay makes, which is 5 at
    // this camera and 3 at a camera with no lens -- see trap 7. Step 1 set
    // this to 4 with a note guessing at jitter plus a scatter direction; that
    // guess was wrong in a way reading get_ray settled. The jitter is two,
    // the defocus disk is two more and conditional, and the shutter time is a
    // fifth that nothing reads yet.
    static const uint32_t RAY_DRAWS_WITH_LENS = 5;

    // What the probe holds before any dispatch writes it, serving the same
    // purpose as STORAGE_SENTINEL above: it distinguishes "the shader never
    // wrote this" from "the shader wrote the wrong thing". Outside [0,1), so
    // it fails the range test too rather than only the comparison.
    static constexpr float RAY_PROBE_SENTINEL = -1.0f;

    // Buckets for the chi-square over [0,1). Sixteen leaves 120,000 values
    // expected in each, which is enough for the statistic to be tight.
    static const int RNG_BUCKETS = 16;

    // Phase 6 step 2. The camera, and these six are the INPUTS the CPU's
    // camera class takes. Copied from the first scene in src/main.cpp so that
    // the end-of-phase comparison has a camera both backends can be given,
    // with one deviation: the aspect ratio comes from the render target
    // instead of the scene's 16/9, because RENDER_WIDTH and RENDER_HEIGHT are
    // 800x600. Driving the CPU to the same 800x600 is a matter of setting
    // aspect_ratio to 4/3 and image_width to 800, since it derives its height
    // as int(image_width / aspect_ratio).
    //
    // doubles, matching the CPU exactly, and narrowed to float only when the
    // uniform is filled. The arithmetic that produces the six vectors runs
    // once per program, so there is nothing to gain by doing it in float and
    // something to lose: the uniform would then hold the float-rounded
    // results of float-rounded intermediates.
    static constexpr double CAMERA_VFOV = 21.0;
    static constexpr double CAMERA_DEFOCUS_ANGLE = 0.6;
    static constexpr double CAMERA_FOCUS_DIST = 10.0;

    VkBuffer cameraUniformBuffer;
    VkDeviceMemory cameraUniformMemory;
    void* cameraUniformMapped = nullptr;

    // Kept after the upload, because the checkpoint needs them. These are the
    // exact float values the shader reads, so the host reference starts from
    // the same numbers rather than from the doubles they were rounded from --
    // which isolates "did the shader do the arithmetic right" from "is
    // narrowing to float lossy", two questions that would otherwise be mixed
    // into one tolerance.
    CameraUniform cameraValues{};

    // The orthonormal basis, in double. NOT uploaded: the shader derives
    // every ray from the six vectors above and never needs the basis itself.
    // The checkpoint does need it -- the symmetry and squareness tests below
    // are stated in terms of w -- so it stays on the host, where it is free.
    glm::dvec3 cameraU{};
    glm::dvec3 cameraV{};
    glm::dvec3 cameraW{};

    // Semaphores order one GPU operation against another and never block the
    // CPU -- their state cannot even be read from the host. A fence is the
    // opposite tool: it is how the CPU waits for the GPU to finish.
    //
    // Per frame in flight, which is safe because the fence for a slot
    // guarantees the submit that waited on its semaphore has completed.
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkFence> inFlightFences;

    // Which in-flight slot the next frame uses. Cycles 0, 1, 0, 1...
    uint32_t currentFrame = 0;

    // Deliberate deviation: the conventional setup uses ONE render-finished
    // semaphore. That violates VUID-vkQueueSubmit-pSignalSemaphores-00067, and
    // current validation layers catch it. A present's wait on a binary
    // semaphore is only known to have completed once that image is
    // re-acquired, so reusing one semaphore across three images can signal it
    // while the presentation engine still holds it. One per swapchain image
    // fixes it, because image i coming round again is exactly the proof its
    // last present finished.
    //
    // Step 10 deliberately leaves this one alone. The usual treatment converts
    // it to MAX_FRAMES_IN_FLIGHT indexed by currentFrame at that point, which
    // reintroduces exactly the violation above -- frames in flight and image
    // count are independent numbers, and this one belongs to the image.
    //
    // Belonging to the image also means sharing its lifetime: these are
    // rebuilt with the swapchain in recreateSwapChain(), not kept alongside
    // the per-frame objects.
    std::vector<VkSemaphore> renderFinishedSemaphores;

    // Set from the GLFW callback, consumed after present. Needed because a
    // driver is permitted to report VK_SUCCESS from a mismatched swapchain
    // rather than OUT_OF_DATE, leaving the return codes silent on a resize.
    bool framebufferResized = false;

    // Only so the frame rate can be reported. FIFO is vsync, so the average
    // should settle on the display's refresh interval rather than running
    // free -- which is the cheapest confirmation that frames really present.
    uint32_t frameCounter = 0;
    std::chrono::steady_clock::time_point frameWindowStart;

    // t = 0 for the animation. Back after step 6 deleted it, because step 6
    // had nothing left to animate; set at the end of initVulkan so that the
    // several hundred milliseconds of device and pipeline setup do not count
    // as elapsed time and the first frame starts at a shift of zero.
    std::chrono::steady_clock::time_point startTime;

    // The time value pushed for the LAST RECORDED frame, kept so that the
    // post-loop checkpoint can compare the image against the host formula at
    // exactly that time. Recording always immediately precedes submission in
    // drawFrame, and submissions on one queue complete in order, so after
    // vkDeviceWaitIdle the image holds the result of this value and no other.
    //
    // A float rather than a double on purpose: it is the same float that was
    // pushed, so the shift derived from it on the host is bit-identical to the
    // one the shader derived.
    float lastPushedTime = 0.0f;

    // Printed once each on the first recorded frame rather than every frame.
    // The description is derived from the arguments the barrier was actually
    // recorded with, which is the opposite of a hand-maintained parallel list.
    int barriersLogged = 0;

    void initWindow() {
        glfwInit();

        // GLFW was written for OpenGL and creates an OpenGL context by
        // default. Vulkan has no context -- we manage every resource
        // ourselves -- so that default has to be switched off.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // The GLFW_RESIZABLE hint that used to sit here is gone: resizing is
        // GLFW's default, and the recreation logic it was waiting for now
        // exists.
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

        // GLFW callbacks are plain C function pointers, so they cannot be
        // non-static member functions -- there is no `this` to pass. The
        // window user pointer is the standard bridge: stash the instance here,
        // fetch it back inside the static callback. Phase 7's keyboard and
        // mouse callbacks will need exactly the same arrangement.
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    // Only records that a resize happened. It fires from inside
    // glfwPollEvents, and rebuilding a swapchain from there would mean
    // rebuilding it in the middle of a frame.
    static void framebufferResizeCallback(GLFWwindow* window, int width,
                                          int height) {
        auto app =
            reinterpret_cast<vulkan_app*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderFinishedSemaphores();
        createRenderPass();

        // Before the pipeline: the pipeline layout is built from this, and the
        // pipeline from that.
        createDescriptorSetLayout();
        createGraphicsPipeline();

        // The compute path's own set layout and pipeline, independent of
        // everything above. Neither touches the render pass, the framebuffers
        // or the swapchain, which is why a compute pipeline needs nothing
        // recreated when the window resizes.
        createComputeDescriptorSetLayout();
        createComputePipeline();

        createFramebuffers();
        createCommandPool();

        // The sampler outlived the texture it was built for in step 2. It
        // names no image, so there is nothing for it to have lost.
        createTextureSampler();

        // After the command pool, which its transition and its clear submit
        // through. The only image left in the program.
        createStorageImage();

        // Before the compute descriptor set, which points at it. Nothing else
        // depends on it: it is not on the display path and no pipeline is
        // built from it.
        createRayProbeBuffer();

        // Before the compute descriptor set, which points at it. Its
        // arithmetic needs nothing from the device, so it could run anywhere;
        // it goes here so that the printed camera sits next to the buffer it
        // was written into.
        createCameraUniformBuffer();

        createDescriptorPool();
        createDescriptorSets();

        // From the same pool as the graphics set. Then ONE dispatch at
        // t = 0, which is now the startup self-test rather than the only
        // dispatch there is -- the per-frame path in recordCommandBuffer
        // immediately overwrites what it produced.
        createComputeDescriptorSet();
        dispatchCompute(0.0f);

        createCommandBuffers();
        createSyncObjects();

        // Last, so that the several hundred milliseconds spent above do not
        // count as elapsed animation time and the first frame starts at a
        // shift of zero.
        startTime = std::chrono::steady_clock::now();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error(
                "validation layers requested, but not available!");
        }

        // Optional metadata. Drivers occasionally use it to apply
        // per-application workarounds.
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Raytracing Engine Vulkan Backend";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // macOS/MoltenVK. Since SDK 1.3.216 MoltenVK is reported as a
        // portability driver -- one that does not implement all of Vulkan,
        // because Metal cannot. Vulkan will not hand one over unless we opt
        // in with both this flag and the matching extension, added in
        // getRequiredExtensions(). Without them vkCreateInstance fails with
        // VK_ERROR_INCOMPATIBLE_DRIVER.
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // The real debug messenger cannot exist yet -- it is created from the
        // instance we are about to make. Chaining a create-info here via
        // pNext gives the layers a temporary one covering vkCreateInstance
        // and vkDestroyInstance, which would otherwise report nothing.
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount =
                static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext =
                (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        listAvailableExtensions();
    }

    // The layer ships with the SDK rather than the driver, so it may simply
    // not be installed. Same two-call enumeration pattern as extensions.
    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    std::vector<const char*> getRequiredExtensions() {
        // Ask GLFW which extensions this platform needs to present to a
        // window, rather than hardcoding them.
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions =
            glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(
            glfwExtensions, glfwExtensions + glfwExtensionCount);

        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

        // Required dependency of the VK_KHR_portability_subset device
        // extension enabled in createLogicalDevice(). Without it
        // vkCreateDevice reports
        // VUID-vkCreateDevice-ppEnabledExtensionNames-01387.
        extensions.push_back(
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    void populateDebugMessengerCreateInfo(
        VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType =
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(
                instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    // GLFW hides the platform-specific call behind one function. On macOS it
    // takes the CAMetalLayer from the NSWindow's content view and calls
    // vkCreateMetalSurfaceEXT; on Windows the same line becomes
    // vkCreateWin32SurfaceKHR. The extensions this needs -- VK_KHR_surface and
    // VK_EXT_metal_surface -- are already enabled: they are exactly what
    // glfwGetRequiredInstanceExtensions() has been returning since step 2.
    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        std::cout << "Window surface created" << std::endl;
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        std::cout << "Selected GPU: " << properties.deviceName << std::endl;

        // maxImageCount of 0 is not "none available" -- it is the spec's way
        // of saying there is no upper limit beyond memory.
        SwapChainSupportDetails support = querySwapChainSupport(physicalDevice);
        std::cout << "Swapchain support for this surface:\n"
                  << "\tsurface formats: " << support.formats.size() << "\n"
                  << "\tpresent modes:   " << support.presentModes.size()
                  << "\n"
                  << "\timage count:     "
                  << support.capabilities.minImageCount << " min, "
                  << support.capabilities.maxImageCount << " max"
                  << (support.capabilities.maxImageCount == 0
                          ? " (0 = no limit)" : "")
                  << std::endl;
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        // vkCreateDevice rejects the same family index appearing twice in
        // pQueueCreateInfos (VUID-VkDeviceCreateInfo-queueFamilyIndex-02802),
        // so the set is not tidiness -- it is what stops a validation error on
        // hardware where the two roles share a family, as here.
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsAndComputeFamily.value(),
            indices.presentFamily.value()
        };

        // Required even with a single queue: relative scheduling priority
        // in [0.0, 1.0], used when queues contend for the GPU.
        float queuePriority = 1.0f;

        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // No optional hardware features requested yet. Vulkan will not let
        // us use a feature we did not ask for here, even if the hardware
        // supports it -- knowing this up front is what lets the driver
        // specialise.
        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount =
            static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // Older guidance sets device-level layers here, for compatibility
        // with implementations predating the instance-level ones. That advice
        // is now out of date: the current spec requires enabledLayerCount to
        // be 0 (VUID-VkDeviceCreateInfo-enabledLayerCount-12384). Device
        // layers were removed, not just deprecated; the instance-level layers
        // cover device calls.
        createInfo.enabledLayerCount = 0;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        // The queues already exist -- they were created alongside the device
        // from queueCreateInfos above. This only fetches handles to them.
        vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0,
                         &graphicsAndComputeQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0,
                         &presentQueue);

        std::cout << "Logical device created from "
                  << queueCreateInfos.size() << " queue create info(s):\n"
                  << "\tgraphics+compute family "
                  << indices.graphicsAndComputeFamily.value() << "\n"
                  << "\tpresent family          "
                  << indices.presentFamily.value() << "\n"
                  << "\tsame VkQueue handle:    "
                  << (graphicsAndComputeQueue == presentQueue ? "yes" : "no")
                  << std::endl;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        // Must stay inside the guard: calling the surface queries on a device
        // that does not support VK_KHR_swapchain is undefined, not a graceful
        // failure. And supporting the extension is not the same as having
        // anything usable for *this* surface -- hence the second check.
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport =
                querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() &&
                                !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported &&
               swapChainAdequate;
    }

    // vkEnumerateDeviceExtensionProperties, not the Instance version used in
    // Phase 1 -- same two-call count-then-fill shape, different scope. Erasing
    // from a set of the required names leaves it empty only if all were found.
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                             nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                             availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                                 deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                                  &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                             nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                                 details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                                  &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                device, surface, &presentModeCount,
                details.presentModes.data());
        }

        return details;
    }

    // Chosen deliberately rather than just taking formats[0]. An _SRGB format
    // makes the hardware do the linear<->sRGB conversion on read and write.
    // That matters here: a path tracer accumulates linear radiance, and
    // something has to gamma-encode it before display. Letting the swapchain
    // do it is free -- but then the shader must NOT also gamma-correct in
    // Phase 6, or the correction lands twice and the image washes out.
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
                availableFormat.colorSpace ==
                    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        // Safe only because step 2a rejects any device with no formats.
        return availableFormats[0];
    }

    // FIFO is the one mode the spec guarantees every implementation offers,
    // which is what makes it the fallback rather than a preference. MAILBOX
    // gives the same no-tearing promise without blocking the submitting
    // thread -- lower latency, more power.
    VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // A currentExtent of UINT32_MAX is the window system saying "pick
        // whatever you like"; any other value is a size we must match.
        if (capabilities.currentExtent.width !=
            std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        // GLFW measures windows in screen coordinates; Vulkan works in pixels.
        // On a Retina display they differ by 2x, so glfwGetWindowSize here
        // would build a swapchain a quarter of the needed size -- the classic
        // "image in the corner" bug. glfwGetFramebufferSize returns pixels.
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width,
                                        capabilities.minImageExtent.width,
                                        capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height,
                                         capabilities.minImageExtent.height,
                                         capabilities.maxImageExtent.height);

        return actualExtent;
    }

    static const char* presentModeName(VkPresentModeKHR mode) {
        switch (mode) {
            case VK_PRESENT_MODE_IMMEDIATE_KHR:    return "IMMEDIATE";
            case VK_PRESENT_MODE_MAILBOX_KHR:      return "MAILBOX";
            case VK_PRESENT_MODE_FIFO_KHR:         return "FIFO";
            case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED";
            default:                               return "other";
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport =
            querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat =
            chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode =
            chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        // One more than the minimum: sitting at exactly minImageCount can mean
        // waiting on the driver's internal work before an image is available.
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        // maxImageCount of 0 means "no limit", so it must not be read as a
        // ceiling of zero. Here max is 3 and min+1 is 3 -- exactly at the top.
        if (swapChainSupport.capabilities.maxImageCount > 0 &&
            imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;

        // Always 1 unless rendering stereoscopic 3D, where an image would
        // carry one layer per eye.
        createInfo.imageArrayLayers = 1;

        // We draw straight into these images. This stays COLOR_ATTACHMENT
        // rather than becoming a transfer destination, because the Phase 5
        // compute output lands in a separate storage image that a fullscreen
        // pass samples.
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {
            indices.graphicsAndComputeFamily.value(),
            indices.presentFamily.value()
        };

        // EXCLUSIVE: one queue family owns an image at a time and ownership
        // must be handed over explicitly -- the fast path, and what we take
        // here since step 1b showed both roles are family 0. CONCURRENT drops
        // the transfer requirement at a cost. Both branches exist so this stays
        // correct on hardware where the families differ.
        if (indices.graphicsAndComputeFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        // No rotation or flip -- hand back the transform the surface already
        // reports. Mobile drivers use this field for screen orientation.
        createInfo.preTransform =
            swapChainSupport.capabilities.currentTransform;

        // Ignore alpha when compositing against other windows on the desktop.
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode = presentMode;

        // Lets the implementation skip work on pixels hidden behind another
        // window. Only unsafe if we needed to read those pixels back.
        createInfo.clipped = VK_TRUE;

        // Stays null even on recreation. recreateSwapChain() destroys the old
        // swapchain BEFORE building the new one, so there is nothing to hand
        // over. Using this properly would mean keeping the old swapchain alive
        // through creation, which would let presentation continue from it
        // during a resize instead of pausing.
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        // The field above is a *minimum*; the implementation is free to make
        // more, so the real count is asked for rather than assumed.
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                                swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

        std::cout << "Swapchain created:\n"
                  << "\timages:       " << swapChainImages.size()
                  << " (requested at least " << createInfo.minImageCount
                  << ")\n"
                  << "\tformat:       " << swapChainImageFormat
                  << (swapChainImageFormat == VK_FORMAT_B8G8R8A8_SRGB
                          ? " (B8G8R8A8_SRGB)" : "")
                  << "\n\tpresent mode: " << presentModeName(presentMode)
                  << "\n\tsharing mode: "
                  << (createInfo.imageSharingMode ==
                              VK_SHARING_MODE_EXCLUSIVE
                          ? "EXCLUSIVE" : "CONCURRENT")
                  << "\n\textent:       " << swapChainExtent.width << "x"
                  << swapChainExtent.height << std::endl;
    }

    // One view per swapchain image. Only the loop is left here: Phase 5 step 2
    // moved the view itself into createImageView below, because the texture
    // needs byte-for-byte the same construction and the storage image will be
    // a third caller.
    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            // The format the swapchain was created with, which is why step 2c
            // kept that around.
            swapChainImageViews[i] =
                createImageView(swapChainImages[i], swapChainImageFormat);
        }

        std::cout << "Image views created: " << swapChainImageViews.size()
                  << " for " << swapChainImages.size()
                  << " swapchain image(s)" << std::endl;
    }

    // Nothing in Vulkan binds a VkImage. A shader or a framebuffer binds a
    // VIEW, which is an interpretation layered over the image's storage: how
    // to read it, which part of it, and with what channel remapping. The
    // image can outlive any particular reading of it, and can be read several
    // ways at once.
    VkImageView createImageView(VkImage image, VkFormat format) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = image;

        // How to read it: a plain 2-D texture. This is the field that would
        // say CUBE or 2D_ARRAY for the same underlying memory.
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = format;

        // The swizzle. IDENTITY throughout means no channel remapping; this is
        // where you would broadcast red to every channel for a monochrome view
        // without touching the image underneath.
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        // Which part of the image the view covers: colour data rather than
        // depth or stencil, no mipmaps, and one array layer. For a swapchain
        // image that has to agree with the imageArrayLayers = 1 set in step
        // 2c; for the texture, with the mipLevels = 1 and arrayLayers = 1 set
        // in createImage.
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &createInfo, nullptr, &imageView) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create image view!");
        }

        return imageView;
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};

        // Must match the swapchain's format: step 7 backs these attachments
        // with the swapchain images themselves.
        colorAttachment.format = swapChainImageFormat;

        // One sample, agreeing with the multisampling-off choice in step 5.
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

        // What happens to the contents on entry and exit. On a tile-based GPU
        // these are not bookkeeping: CLEAR lets the driver skip reading the
        // tile in from main memory at all, and DONT_CARE on store would let it
        // skip writing back. That bandwidth saving is most of why Apple
        // Silicon is fast. STORE is required here only because we want to
        // look at the result afterwards.
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // No stencil buffer in play.
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        // Images have a LAYOUT: the same pixels are arranged differently in
        // memory depending on whether they are being rendered into, sampled,
        // presented or copied. Other APIs perform these transitions
        // invisibly; Vulkan makes them fields you fill in.
        //
        // UNDEFINED says the previous contents do not matter -- safe only
        // because loadOp clears, and it lets the driver discard them rather
        // than preserve them. PRESENT_SRC_KHR is where the image has to end
        // up, since the swapchain displays it next.
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};

        // An index into pAttachments below -- and the same number the fragment
        // shader writes through with layout(location = 0) out vec4. Those two
        // numberings are one numbering; this is where they meet.
        colorAttachmentRef.attachment = 0;

        // The layout to hold DURING the subpass. So the driver runs
        // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL on entry and
        // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR on exit by itself.
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};

        // Spelled out because a subpass could bind compute instead.
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        // The pass performs the UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        // transition declared above on its own, but nothing so far says WHEN.
        // By default it may start as soon as the command buffer begins
        // executing, which can be before vkAcquireNextImageKHR has signalled
        // that the image is actually ours to write. This pins the ordering.
        VkSubpassDependency dependency{};

        // EXTERNAL stands for the implicit operations before the render pass.
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;

        // Wait on the colour attachment output stage of whatever preceded us,
        // and make our own colour writes the thing that waits. Paired with
        // step 9b waiting on imageAvailableSemaphore at this same stage, that
        // is what keeps the layout transition behind the acquire.
        dependency.srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr,
                               &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        std::cout << "Render pass created:\n"
                  << "\tattachments: " << renderPassInfo.attachmentCount
                  << " colour, format " << swapChainImageFormat << "\n"
                  << "\tsubpasses:   " << renderPassInfo.subpassCount
                  << ", graphics bind point\n"
                  << "\tload/store:  CLEAR on entry, STORE on exit\n"
                  << "\tlayouts:     UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL"
                     " -> PRESENT_SRC_KHR" << std::endl;
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;

            // A compatibility reference, exactly as in the pipeline: any
            // render pass with matching attachment formats and counts works.
            framebufferInfo.renderPass = renderPass;

            // Positional, and this is where a chain built over four steps
            // finally lands on an image: the fragment shader's
            // layout(location = 0) out, the subpass's
            // VkAttachmentReference.attachment = 0, the render pass's
            // pAttachments[0], and attachments[0] here are one numbering.
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;

            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;

            // The third place this number has to agree: the swapchain's
            // imageArrayLayers in 2c, the views' layerCount in 3, and here.
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr,
                                    &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }

        std::cout << "Framebuffers created: " << swapChainFramebuffers.size()
                  << " for " << swapChainImageViews.size()
                  << " image view(s), at " << swapChainExtent.width << "x"
                  << swapChainExtent.height << std::endl;
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices =
            findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

        // Lets one buffer be reset and re-recorded on its own; without it the
        // whole pool has to be reset at once. We re-record every frame.
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        // A pool belongs to exactly one queue family, because its commands are
        // encoded for that family's capabilities -- a buffer from a graphics
        // pool cannot be submitted to a transfer-only queue.
        //
        // The reason pools exist as a separate object at all is that they are
        // NOT thread-safe: two threads cannot record from one pool at the same
        // time. One pool per thread is the pattern that makes parallel command
        // recording possible.
        poolInfo.queueFamilyIndex =
            queueFamilyIndices.graphicsAndComputeFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }

        std::cout << "Command pool created on queue family "
                  << poolInfo.queueFamilyIndex << std::endl;
    }

    // Only the flags this backend will ever ask for. Which memory type an
    // allocation landed on is the single observable fact a buffer produces,
    // and it is unreadable as a hex mask.
    static std::string memoryPropertyString(VkMemoryPropertyFlags flags) {
        std::string out;
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            out += "DEVICE_LOCAL|";
        }
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            out += "HOST_VISIBLE|";
        }
        if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
            out += "HOST_COHERENT|";
        }
        if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
            out += "HOST_CACHED|";
        }
        if (out.empty()) {
            return "(none)";
        }
        out.pop_back();
        return out;
    }

    static std::string bufferUsageString(VkBufferUsageFlags flags) {
        std::string out;
        if (flags & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
            out += "TRANSFER_SRC|";
        }
        if (flags & VK_BUFFER_USAGE_TRANSFER_DST_BIT) {
            out += "TRANSFER_DST|";
        }
        if (flags & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
            out += "VERTEX_BUFFER|";
        }
        if (flags & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) {
            out += "INDEX_BUFFER|";
        }
        if (flags & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
            out += "UNIFORM_BUFFER|";
        }
        if (out.empty()) {
            return "(none)";
        }
        out.pop_back();
        return out;
    }

    // Two independent constraints have to be satisfied at once, which is why
    // this takes two arguments.
    //
    // typeFilter is memoryTypeBits straight out of the buffer's requirements:
    // bit i set means "memory type i is capable of backing this particular
    // buffer". That is the hardware's constraint. properties is what we need
    // from the memory -- mappable, device-local -- which is ours. The first
    // type satisfying both wins.
    uint32_t findMemoryType(uint32_t typeFilter,
                            VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            bool capable = typeFilter & (1 << i);

            // ALL the requested bits, not any of them: a type offering
            // HOST_VISIBLE when DEVICE_LOCAL was also asked for is no match.
            bool suitable =
                (memProperties.memoryTypes[i].propertyFlags & properties) ==
                properties;

            if (capable && suitable) {
                return i;
            }
        }

        throw std::runtime_error("failed to find a suitable memory type!");
    }

    // Everything two buffers have in common. The three leading parameters are
    // exactly what differs between a staging buffer and the real one in step
    // 3b; buffer and bufferMemory are outputs. label exists only so each
    // allocation can report which memory type it landed on, which is the one
    // observable thing this sequence produces.
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory,
                      const std::string& label) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;

        // What the buffer is FOR, which is checked against how it is later
        // used. Not a hint.
        bufferInfo.usage = usage;

        // Same reasoning as the swapchain's: only the graphics+compute family
        // touches these, so no cross-family ownership transfer is needed.
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer: " + label);
        }

        // Nothing has been allocated yet -- buffer is a description with no
        // storage behind it. Only the driver knows which memory can back this
        // particular buffer, so it is asked rather than assumed.
        //
        // size here can exceed the requested size: alignment padding is the
        // driver's business. Allocating the requested size instead of
        // memRequirements.size is a real bug that happens to work on
        // forgiving hardware.
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, properties);

        // One allocation per buffer. The device limit maxMemoryAllocationCount
        // is real and small, so this does not scale -- engines make a few
        // large allocations and sub-allocate at offsets instead. Kept naive
        // deliberately; the BVH upload is where it bites.
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to allocate memory for: " +
                                     label);
        }

        // The final 0 is the offset into the allocation. It is non-zero only
        // when several buffers share one allocation, which is the scheme the
        // comment above describes and this code does not use.
        vkBindBufferMemory(device, buffer, bufferMemory, 0);

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        const VkMemoryType& chosen =
            memProperties.memoryTypes[allocInfo.memoryTypeIndex];

        std::cout << label << ": " << size << " B requested"
                  << "\n\tusage:     " << bufferUsageString(usage)
                  << "\n\tallocated: " << memRequirements.size
                  << " B, alignment " << memRequirements.alignment << " B"
                  << "\n\tcan use:   memory type mask 0x" << std::hex
                  << memRequirements.memoryTypeBits << std::dec
                  << "\n\tchose:     type " << allocInfo.memoryTypeIndex
                  << " on heap " << chosen.heapIndex << ", "
                  << memoryPropertyString(chosen.propertyFlags) << std::endl;
    }

    // The envelope every one-off GPU operation outside the render loop needs.
    // It was inline in copyBuffer back when a buffer-to-buffer copy was the
    // only such operation; by step 1 of Phase 5 there were five, so it moved
    // out here, and copyBuffer itself is gone as of step 6. Only the commands
    // recorded between the two calls ever differ.
    //
    // Three callers remain: transitionImageLayout, readbackImage and
    // dispatchCompute -- and as of step 7 that last one is only the startup
    // self-test. The per-frame dispatch records into the frame's own command
    // buffer instead, where vkQueueWaitIdle would mean blocking the CPU on a
    // fully idle GPU every frame and throwing away frames in flight
    // altogether. Two pipeline barriers do the ordering there.
    VkCommandBuffer beginSingleTimeCommands() {
        // Reuses the render loop's pool. A separate short-lived pool created
        // with TRANSIENT_BIT would let the driver optimise for throwaway
        // buffers, which is the obvious refinement and this does not take.
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        // Recorded once and thrown away, which the driver may exploit. The
        // per-frame buffers in the render loop cannot say this: they are
        // re-recorded every frame.
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        // No semaphores and no fence. Any family supporting graphics or
        // compute implicitly supports transfer, so family 0's queue serves.
        // Hardware with a dedicated transfer-only family can upload faster;
        // that is not chased here.
        vkQueueSubmit(graphicsAndComputeQueue, 1, &submitInfo, VK_NULL_HANDLE);

        // The blunt instrument: blocks the CPU until the whole queue drains.
        // It is what makes freeing the staging buffer safe in the caller --
        // doing that while the copy was still in flight would be a
        // use-after-free the GPU commits, not the CPU. A fence would let many
        // copies be submitted and waited on together, which is what will
        // matter when uploading the BVH rather than one 60-byte triangle.
        vkQueueWaitIdle(graphicsAndComputeQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    static std::string pipelineStageString(VkPipelineStageFlags stages) {
        std::string out;
        auto add = [&out](const char* name) {
            if (!out.empty()) {
                out += "|";
            }
            out += name;
        };
        if (stages & VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) add("TOP_OF_PIPE");
        if (stages & VK_PIPELINE_STAGE_TRANSFER_BIT) add("TRANSFER");
        if (stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) add("COMPUTE_SHADER");
        if (stages & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
            add("FRAGMENT_SHADER");
        return out.empty() ? "none" : out;
    }

    static std::string accessFlagsString(VkAccessFlags access) {
        std::string out;
        auto add = [&out](const char* name) {
            if (!out.empty()) {
                out += "|";
            }
            out += name;
        };
        if (access & VK_ACCESS_SHADER_READ_BIT) add("SHADER_READ");
        if (access & VK_ACCESS_SHADER_WRITE_BIT) add("SHADER_WRITE");
        if (access & VK_ACCESS_TRANSFER_READ_BIT) add("TRANSFER_READ");
        if (access & VK_ACCESS_TRANSFER_WRITE_BIT) add("TRANSFER_WRITE");
        return out.empty() ? "0 (none)" : out;
    }

    static std::string imageLayoutString(VkImageLayout layout) {
        switch (layout) {
            case VK_IMAGE_LAYOUT_UNDEFINED:
                return "UNDEFINED";
            case VK_IMAGE_LAYOUT_GENERAL:
                return "GENERAL";
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return "TRANSFER_DST_OPTIMAL";
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                return "TRANSFER_SRC_OPTIMAL";
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return "SHADER_READ_ONLY_OPTIMAL";
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                return "PRESENT_SRC_KHR";
            default:
                return "layout " + std::to_string(static_cast<int>(layout));
        }
    }

    // The image counterpart of createBuffer, and deliberately the same shape:
    // the leading parameters are what differs between one image and the next,
    // image and imageMemory are outputs, label exists so the allocation can
    // report which memory type it landed on.
    void createImage(uint32_t width, uint32_t height, VkFormat format,
                     VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImage& image,
                     VkDeviceMemory& imageMemory, const std::string& label) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;

        // depth is 1 for a 2-D image, not 0. Vulkan models 1-D, 2-D and 3-D
        // images with one extent, and an extent of zero in any dimension the
        // imageType uses is invalid -- so the flat image is one texel deep
        // rather than zero texels deep.
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;

        // No mip chain and no array. Mipmap generation is one of the topics
        // this project skips: a path tracer samples along rays rather than
        // across a surface at a known screen-space derivative, so the level of
        // detail a mip chain selects has nothing to select from.
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;

        imageInfo.format = format;
        imageInfo.tiling = tiling;

        // UNDEFINED, not PREINITIALIZED. The difference is a claim about
        // whether data is already in there: PREINITIALIZED promises the host
        // wrote some through a linear mapping and must be preserved, which is
        // only meaningful at TILING_LINEAR and is false here. UNDEFINED says
        // the contents are garbage and may be discarded, which is what lets
        // the first transition below be free.
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        // What it is FOR, checked against how it is used, exactly as with a
        // buffer. An image sampled by a shader that was never created with
        // SAMPLED_BIT is a validation error, not a slow path.
        imageInfo.usage = usage;

        // Same reasoning as the swapchain's and the buffers': one
        // graphics+compute family touches everything, so no ownership
        // transfer is needed.
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        // One sample per texel. Multisampling is another skipped topic --
        // the path tracer antialiases by jittering rays inside the pixel,
        // which is a better answer to the same problem and costs no extra
        // image memory.
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image: " + label);
        }

        // Same two-step as a buffer: the handle above is a description with no
        // storage behind it, and only the driver knows what memory can back
        // it. The size it reports is reliably LARGER than width*height*4 for
        // a tiled image, because the swizzle pads -- which is the most direct
        // evidence that TILING_OPTIMAL is not row-major.
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to allocate memory for: " +
                                     label);
        }

        vkBindImageMemory(device, image, imageMemory, 0);

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        const VkMemoryType& chosen =
            memProperties.memoryTypes[allocInfo.memoryTypeIndex];

        std::cout << label << ": " << width << "x" << height << ", "
                  << (static_cast<size_t>(width) * height * 4)
                  << " B of pixel data"
                  << "\n\ttiling:    "
                  << (tiling == VK_IMAGE_TILING_OPTIMAL ? "OPTIMAL"
                                                        : "LINEAR")
                  << "\n\tallocated: " << memRequirements.size
                  << " B, alignment " << memRequirements.alignment << " B"
                  << "\n\tchose:     type " << allocInfo.memoryTypeIndex
                  << " on heap " << chosen.heapIndex << ", "
                  << memoryPropertyString(chosen.propertyFlags) << std::endl;
    }

    // A layout transition is GPU work -- the driver may genuinely rearrange
    // bytes -- so it is a recorded command, not a host call. The vehicle is a
    // pipeline barrier, which is doing two separable jobs at once: changing
    // the layout, and establishing a dependency between what came before and
    // what comes after.
    //
    // No format parameter, which the conventional signature has and does not
    // use at this point. It is there for depth images, whose aspectMask has to
    // name depth and possibly stencil instead of colour; depth buffering is a
    // skipped topic, so taking an argument that is ignored would be a worse
    // comment than this one.
    void transitionImageLayout(VkImage image, VkImageLayout oldLayout,
                               VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        // Used only by a QUEUE FAMILY OWNERSHIP transfer, which this is not.
        // The sentinel is required: leaving these zeroed would name family 0
        // and request a transfer to itself.
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        // The four fields that are the actual content of the barrier. Stages
        // say WHEN the dependency applies, access masks say WHAT KIND of
        // memory access is being ordered. Wrong values here very often work
        // on desktop hardware anyway, which is precisely what makes this the
        // defect that ships.
        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        // As of step 6 only three of the seven rows below are REACHABLE:
        // UNDEFINED -> GENERAL, and the GENERAL <-> TRANSFER_SRC_OPTIMAL pair
        // readbackImage uses. The four naming SHADER_READ_ONLY_OPTIMAL went
        // idle with the checkerboard texture and are kept on purpose, because
        // the value of this table is the COMPARISON between rows -- two
        // transitions ending in the same TRANSFER_SRC_OPTIMAL from two
        // different starting layouts, sharing not one mask bit between them.

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            // Nothing wrote this image before now, so there is nothing to
            // wait for and nothing to flush: access 0 at the top of the pipe.
            // The copy that follows is what must be held back until the
            // rearrangement is done, hence TRANSFER on the far side.
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            // The real one. The copy's writes must complete AND become
            // visible to shader reads; naming the fragment stage is what lets
            // everything earlier in the pipeline carry on regardless.
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            // Verification only, and the direction that proves a layout is a
            // statement about the present rather than a property of the image.
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                   newLayout == VK_IMAGE_LAYOUT_GENERAL) {
            // Step 4's storage image reaching its resting layout, and the
            // only transition it ever gets. Nothing wrote it, so the source
            // side is empty as before; the destination side names BOTH
            // consumers, because the next thing to touch this image is a
            // clear (transfer) and everything after that is compute.
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask =
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL &&
                   newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            // Verification only, as with the texture's pair -- and worth
            // comparing against it. The DESTINATION layout is identical to
            // the texture's readback transition and the masks share not one
            // bit, which is as plainly as it can be shown that the masks are
            // not derivable from the layout names. They describe what the
            // surrounding code does, and here that is a compute write.
            barrier.srcAccessMask =
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                          VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_GENERAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask =
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        } else {
            // Throwing beats guessing. An unlisted pair is a transition
            // nobody reasoned about, and the masks are not derivable from the
            // layouts -- they depend on what the surrounding code does.
            throw std::runtime_error("unsupported layout transition: " +
                                     imageLayoutString(oldLayout) + " -> " +
                                     imageLayoutString(newLayout));
        }

        // The one command. The two zero/nullptr pairs after the stages are
        // memory barriers and buffer barriers -- the same call orders all
        // three kinds of resource, and this uses only the image form.
        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);

        endSingleTimeCommands(commandBuffer);
    }

    // Pulls an image's pixels back to the host, leaving it in the layout it
    // was found in. Written as a helper on its first use because step 5 needs
    // the identical operation on the storage image, where it is the whole
    // checkpoint: proving a compute dispatch wrote the right values at the
    // right coordinates is not something the screen can show.
    std::vector<uint8_t> readbackImage(VkImage image, uint32_t width,
                                       uint32_t height,
                                       VkImageLayout currentLayout) {
        VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * 4;

        VkBuffer dstBuffer;
        VkDeviceMemory dstMemory;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     dstBuffer, dstMemory, "Readback buffer");

        transitionImageLayout(image, currentLayout,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};

        // The inverse of the buffer-to-image copy that filled the texture
        // until step 6 deleted it, and it UNDOES the swizzle: what
        // lands in the buffer is row-major again, which is the only reason
        // comparing it against the generated pixels means anything.
        vkCmdCopyImageToBuffer(commandBuffer, image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuffer,
                               1, &region);

        endSingleTimeCommands(commandBuffer);

        // Back where it came from, so a caller cannot be broken by having been
        // measured.
        transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              currentLayout);

        std::vector<uint8_t> out(static_cast<size_t>(size));
        void* mapped;
        vkMapMemory(device, dstMemory, 0, size, 0, &mapped);
        memcpy(out.data(), mapped, static_cast<size_t>(size));
        vkUnmapMemory(device, dstMemory);

        vkDestroyBuffer(device, dstBuffer, nullptr);
        vkFreeMemory(device, dstMemory, nullptr);

        return out;
    }

    void createTextureSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

        // Magnification is what happens here: a 256x256 texture stretched
        // across a quad that covers most of an 800x600 window means several
        // screen pixels per texel. LINEAR blends the four nearest texels;
        // NEAREST would take one and leave the square boundaries perfectly
        // sharp. Both are defensible for a checkerboard -- LINEAR is chosen so
        // that step 3 shows filtering happening rather than hiding it, and
        // because the artefact it removes is the one a path tracer's output
        // benefits from when render resolution and window size differ.
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;

        // What happens outside [0,1]. REPEAT tiles, which is the useful
        // default and is invisible here because step 3's coordinates stay in
        // range. The alternatives are MIRRORED_REPEAT, CLAMP_TO_EDGE and
        // CLAMP_TO_BORDER, and only the last needs borderColor below to mean
        // anything. W is set for symmetry; a 2-D sampler never consults it.
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        // Deliberate deviation -- the conventional setup enables this. See the
        // file header: anisotropy fixes blur on a surface seen at a grazing
        // angle, this renderer ends up displaying an image face-on at 1:1, and
        // VK_TRUE here without the samplerAnisotropy device feature (which
        // createLogicalDevice does not enable) is a validation error rather
        // than a quality setting. maxAnisotropy is ignored while this is
        // FALSE; it is set to the floor rather than left zero so the struct
        // reads as a decision instead of an oversight.
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;

        // Only consulted by CLAMP_TO_BORDER, which is not in use. Without an
        // extension this field takes a handful of enum values -- transparent,
        // opaque black, opaque white -- and not an arbitrary colour.
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

        // VK_FALSE means texture coordinates run [0,1) regardless of size,
        // which is what lets a shader sample a texture without knowing its
        // resolution. VK_TRUE would make them [0,width) x [0,height) and tie
        // the shader to this image. Worth knowing because step 7 sidesteps
        // samplers entirely and addresses the storage image by integer pixel
        // -- that is imageLoad, not a sampler, and the distinction is easy to
        // blur.
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        // Percentage-closer filtering for shadow maps: compare the sampled
        // value against a reference and return the fraction that passes,
        // rather than the value. Nothing here wants it.
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

        // Inert with one mip level, which is what createImage made. Set
        // explicitly so that adding a mip chain later is a visible change to
        // these lines rather than a silent change in behaviour.
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        // Note what is NOT in this struct: any mention of an image. That is
        // the whole point of a sampler being its own object. The two are
        // paired later, in a descriptor, and either can be swapped without
        // rebuilding the other.
        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }

        // So the declined feature is visible rather than merely absent.
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        std::cout << "Texture sampler created: LINEAR mag/min, REPEAT address"
                  << "\n\tanisotropy:  disabled (hardware supports up to "
                  << properties.limits.maxSamplerAnisotropy
                  << "x, and samplerAnisotropy is not an enabled device "
                     "feature)"
                  << "\n\treferences:  no image -- a sampler is rules only"
                  << std::endl;
    }

    void createStorageImage() {
        // A FORMAT FEATURE: what one particular VkFormat is permitted to be
        // used for, which varies by tiling. See trap 2 at the top of the
        // file for why a guaranteed format is queried anyway.
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(
            physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProperties);

        VkFormatFeatureFlags features = formatProperties.optimalTilingFeatures;
        bool storageSupported =
            (features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
        bool sampledSupported =
            (features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
        bool atomicSupported =
            (features & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) != 0;

        // ATOMIC is printed because it is the one of the three a path tracer
        // may actually miss: accumulating samples from several invocations
        // into one texel without a race wants atomics, and no float format is
        // required to support them.
        std::cout << "Format R8G8B8A8_UNORM at OPTIMAL tiling: STORAGE_IMAGE "
                  << (storageSupported ? "yes" : "no") << ", SAMPLED_IMAGE "
                  << (sampledSupported ? "yes" : "no")
                  << ", STORAGE_IMAGE_ATOMIC "
                  << (atomicSupported ? "yes" : "no") << std::endl;

        if (!storageSupported) {
            throw std::runtime_error(
                "R8G8B8A8_UNORM cannot be used as a storage image");
        }

        // PROBED AHEAD OF BEING NEEDED, which is the cheap half of trap 2.
        // Phase 7 accumulates samples into a float image, and whether that
        // can use atomics is decided by this query. Asking now costs one call
        // while format properties are already being printed; finding out in
        // Phase 7 costs a redesign. Nothing here uses the answer yet, and the
        // image below is still R8G8B8A8_UNORM.
        VkFormatProperties hdrProperties;
        vkGetPhysicalDeviceFormatProperties(
            physicalDevice, VK_FORMAT_R32G32B32A32_SFLOAT, &hdrProperties);
        VkFormatFeatureFlags hdrFeatures = hdrProperties.optimalTilingFeatures;

        std::cout << "Format R32G32B32A32_SFLOAT at OPTIMAL tiling: "
                     "STORAGE_IMAGE "
                  << ((hdrFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
                          ? "yes"
                          : "no")
                  << ", SAMPLED_IMAGE "
                  << ((hdrFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                          ? "yes"
                          : "no")
                  << ", STORAGE_IMAGE_ATOMIC "
                  << ((hdrFeatures &
                       VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT)
                          ? "yes"
                          : "no")
                  << "  (unused in this step -- Phase 7 accumulates here)"
                  << std::endl;

        // Four usage flags, and each one's reason is worth stating because a
        // missing flag is not caught here -- it is caught at the command that
        // needed it, which may be steps away:
        //
        //   STORAGE      compute writes it with imageStore (step 5)
        //   SAMPLED      the fullscreen triangle reads it (step 7)
        //   TRANSFER_SRC readbackImage copies out of it (the checkpoints)
        //   TRANSFER_DST vkCmdClearColorImage writes the sentinel below
        createImage(RENDER_WIDTH, RENDER_HEIGHT, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageImage,
                    storageImageMemory, "Storage image");

        // GENERAL, and it never leaves except to be measured.
        transitionImageLayout(storageImage, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_GENERAL);

        clearStorageImage();

        storageImageView =
            createImageView(storageImage, VK_FORMAT_R8G8B8A8_UNORM);

        std::cout << "Storage image: " << RENDER_WIDTH << "x" << RENDER_HEIGHT
                  << ", resting layout "
                  << imageLayoutString(VK_IMAGE_LAYOUT_GENERAL)
                  << ", view created" << std::endl;

        verifyStorageImageClear();
    }

    // vkCmdClearColorImage is one of the few commands that writes an image
    // outside a render pass, and it needs the image in GENERAL or
    // TRANSFER_DST_OPTIMAL. GENERAL already being the resting layout here is
    // the whole reason this needs no barrier wrapped around it.
    void clearStorageImage() {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        // float32 regardless of the image's bit depth -- see trap 3 at the
        // top of the file. Dividing by 255 rather than writing 0.0f/1.0f
        // literals keeps one definition of the sentinel: the byte values the
        // checkpoint compares against are the source of truth, and a UNORM
        // format converts them back exactly.
        VkClearColorValue clearColor{};
        clearColor.float32[0] = STORAGE_SENTINEL[0] / 255.0f;
        clearColor.float32[1] = STORAGE_SENTINEL[1] / 255.0f;
        clearColor.float32[2] = STORAGE_SENTINEL[2] / 255.0f;
        clearColor.float32[3] = STORAGE_SENTINEL[3] / 255.0f;

        // The same subresource range the barriers use, for the same reason:
        // one mip level, one array layer, the colour aspect. A clear names a
        // range rather than an image because a mip chain or an array would
        // otherwise need one call per level.
        VkImageSubresourceRange range{};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel = 0;
        range.levelCount = 1;
        range.baseArrayLayer = 0;
        range.layerCount = 1;

        vkCmdClearColorImage(commandBuffer, storageImage,
                             VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);

        endSingleTimeCommands(commandBuffer);
    }

    // The checkpoint, and it has to be analytic: nothing on screen changes in
    // this step at all. An image that was allocated but never cleared, a
    // clear that went to the wrong layout, and an 800x600 extent that came
    // out transposed are all silent -- and all three would let step 5 appear
    // to "work" on a 480,000-texel surface of garbage.
    void verifyStorageImageClear() {
        std::vector<uint8_t> actual = readbackImage(
            storageImage, RENDER_WIDTH, RENDER_HEIGHT,
            VK_IMAGE_LAYOUT_GENERAL);

        size_t mismatches = 0;
        size_t firstMismatch = 0;
        for (size_t i = 0; i < actual.size(); i++) {
            if (actual[i] != STORAGE_SENTINEL[i % 4]) {
                if (mismatches == 0) {
                    firstMismatch = i;
                }
                mismatches++;
            }
        }

        std::cout << "Storage image clear verified: " << actual.size()
                  << " B read back against sentinel "
                  << static_cast<int>(STORAGE_SENTINEL[0]) << ","
                  << static_cast<int>(STORAGE_SENTINEL[1]) << ","
                  << static_cast<int>(STORAGE_SENTINEL[2]) << ","
                  << static_cast<int>(STORAGE_SENTINEL[3]) << ", "
                  << mismatches << " mismatched byte(s)" << std::endl;

        // Corner probes. The round trip through
        // GENERAL -> TRANSFER_SRC_OPTIMAL -> GENERAL is itself under test
        // here: it is the first time this image has been through a transition
        // pair, and it is the exact one step 5's checkpoint depends on.
        const uint32_t probes[4][2] = {{0, 0},
                                       {RENDER_WIDTH - 1, 0},
                                       {0, RENDER_HEIGHT - 1},
                                       {RENDER_WIDTH - 1, RENDER_HEIGHT - 1}};
        for (const auto& p : probes) {
            size_t i = (static_cast<size_t>(p[1]) * RENDER_WIDTH + p[0]) * 4;
            std::cout << "\ttexel (" << p[0] << "," << p[1] << "): "
                      << static_cast<int>(actual[i + 0]) << ","
                      << static_cast<int>(actual[i + 1]) << ","
                      << static_cast<int>(actual[i + 2]) << ","
                      << static_cast<int>(actual[i + 3]) << std::endl;
        }

        if (mismatches != 0) {
            throw std::runtime_error(
                "storage image does not hold the sentinel clear colour "
                "(first mismatch at byte " +
                std::to_string(firstMismatch) + ")");
        }
    }

    // Phase 6 step 1. The buffer the checkpoint reads, and the first SHADER
    // STORAGE BUFFER in this project.
    void createRayProbeBuffer() {
        VkDeviceSize size = static_cast<VkDeviceSize>(RENDER_WIDTH) *
                            RENDER_HEIGHT * PROBE_VEC4S_PER_PIXEL * 4 *
                            sizeof(float);

        // HOST_VISIBLE and HOST_COHERENT, where every buffer the renderer
        // itself used was DEVICE_LOCAL behind a staged transfer. The reasoning
        // inverts because the direction does: a vertex buffer is written once
        // by the host and read constantly by the GPU, so it belongs in the
        // fastest memory the GPU has and pays one copy to get there. This is
        // written by the GPU and read once by the host, by a checkpoint that
        // runs when the device is idle and does not care how fast it is.
        // Host-visible means no staging buffer, no transfer command and no
        // barrier -- the mapped pointer simply holds the answer.
        //
        // HOST_COHERENT is the half that is easy to forget. Without it the
        // host's view of this memory could be stale after the GPU wrote it,
        // and vkInvalidateMappedMemoryRanges would be needed before every
        // read. With it the write is visible as soon as the submission
        // completes, which is why the checkpoint reads the pointer directly.
        createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     rayProbeBuffer, rayProbeMemory, "Ray probe buffer");

        if (vkMapMemory(device, rayProbeMemory, 0, size, 0, &rayProbeMapped) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to map the ray probe buffer!");
        }

        // POISONED, for the same reason the storage image is cleared to a
        // sentinel before anything writes it. An all-zero buffer that is never
        // written reads as 0.0 everywhere, which is a legal draw -- so a
        // dispatch that silently did not run would look like a generator stuck
        // at zero rather than like a dispatch that did not run. -1.0 cannot be
        // produced by nextFloat and is reported separately from a mismatch.
        float* poison = static_cast<float*>(rayProbeMapped);
        size_t count = static_cast<size_t>(size) / sizeof(float);
        for (size_t i = 0; i < count; i++) {
            poison[i] = RAY_PROBE_SENTINEL;
        }

        std::cout << "Ray probe buffer: " << size << " B = "
                  << (static_cast<size_t>(RENDER_WIDTH) * RENDER_HEIGHT)
                  << " pixels x " << PROBE_VEC4S_PER_PIXEL
                  << " vec4, host-visible and mapped at " << rayProbeMapped
                  << ", filled with the sentinel " << RAY_PROBE_SENTINEL
                  << std::endl;
    }

    // Phase 6 step 2. camera::initialize() from camera.h, run once on the
    // host, plus the uniform buffer its six results travel in.
    //
    // THE SPLIT IS THE INTERESTING PART. initialize() turns six scene
    // parameters into six vectors; get_ray() uses only those vectors. So this
    // runs once per program and the shader runs 480,000 times a frame, and
    // nothing about the field of view or the basis is recomputed in between.
    // Deciding where that line falls is most of what porting a renderer to a
    // GPU is.
    void createCameraUniformBuffer() {
        // The CPU's inputs, from the first scene in src/main.cpp.
        const glm::dvec3 lookfrom(13.0, 2.0, -3.0);
        const glm::dvec3 lookat(0.0, 0.0, 0.0);
        const glm::dvec3 vup(0.0, 1.0, 0.0);

        // pi and degrees_to_radians from rtweekend.h, inline. glm::radians
        // would do the same; written out so the lines below read as a
        // transliteration of initialize() rather than as a reimplementation.
        const double pi = 3.1415926535897932385;
        double theta = CAMERA_VFOV * pi / 180.0;
        double h = std::tan(theta / 2.0);
        double viewportHeight = 2.0 * h * CAMERA_FOCUS_DIST;

        // THE RATIO OF THE INTEGER DIMENSIONS, not the aspect_ratio
        // parameter, and the CPU does the same thing for a reason worth
        // keeping: it derives image_height as int(image_width / aspect_ratio),
        // so after that truncation the stored ratio and the real one differ.
        // Using the integers reproduces whatever the truncation did. Here they
        // are 800 and 600 and the two agree, which is exactly when a
        // substitution like this goes unnoticed.
        double viewportWidth = viewportHeight *
                               (static_cast<double>(RENDER_WIDTH) /
                                static_cast<double>(RENDER_HEIGHT));

        // THE BASIS. w points BACKWARD, from the target toward the eye, which
        // is why the viewport is placed at -focus_dist * w below and why
        // every ray direction has a dot product of exactly -focus_dist with
        // it. u is right, v is up.
        //
        // v needs no normalisation: w and u are unit and perpendicular, so
        // their cross product is already unit. The CPU does not normalise it
        // either.
        cameraW = glm::normalize(lookfrom - lookat);
        cameraU = glm::normalize(glm::cross(vup, cameraW));
        cameraV = glm::cross(cameraW, cameraU);

        // viewportV is NEGATED. The viewport's vertical axis runs DOWN the
        // image because pixel rows are numbered downward, while v points up.
        // Dropping the minus renders the scene upside down and nothing
        // reports it -- which is the same hazard the fullscreen triangle's v
        // coordinate had in Phase 5, arriving a second time from the other
        // direction.
        glm::dvec3 viewportU = viewportWidth * cameraU;
        glm::dvec3 viewportV = viewportHeight * -cameraV;

        glm::dvec3 pixelDeltaU =
            viewportU / static_cast<double>(RENDER_WIDTH);
        glm::dvec3 pixelDeltaV =
            viewportV / static_cast<double>(RENDER_HEIGHT);

        glm::dvec3 viewportUpperLeft = lookfrom -
                                       (CAMERA_FOCUS_DIST * cameraW) -
                                       viewportU / 2.0 - viewportV / 2.0;

        // Half a pixel in from the corner, so pixel00Loc is the CENTRE of
        // pixel (0,0) rather than its corner. That half-pixel is what makes
        // the jitter's [-0.5, +0.5] range cover exactly the pixel.
        glm::dvec3 pixel00Loc =
            viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);

        // The lens radius. Note the HALF ANGLE: defocus_angle is the full
        // cone subtended by the aperture seen from the focal plane, so the
        // radius uses half of it. At 0.6 degrees this is 10 * tan(0.3 deg) =
        // about 0.052 world units -- a small lens, which is why the blur is
        // subtle at this camera and would be easy to get wrong unnoticed.
        double defocusRadius =
            CAMERA_FOCUS_DIST *
            std::tan(CAMERA_DEFOCUS_ANGLE / 2.0 * pi / 180.0);

        // NARROWED HERE AND NOWHERE ELSE. Everything above is double, as the
        // CPU's is; the single conversion to float happens as the values go
        // into the block the shader reads. Doing the arithmetic in float
        // instead would round every intermediate, and the two backends would
        // then differ by more than one rounding without anything saying which
        // of the two was closer to right.
        cameraValues.center = glm::vec3(lookfrom);
        cameraValues.pixel00Loc = glm::vec3(pixel00Loc);
        cameraValues.pixelDeltaU = glm::vec3(pixelDeltaU);
        cameraValues.pixelDeltaV = glm::vec3(pixelDeltaV);
        cameraValues.defocusDiskU = glm::vec3(cameraU * defocusRadius);
        cameraValues.defocusDiskV = glm::vec3(cameraV * defocusRadius);
        cameraValues.defocusAngle = static_cast<float>(CAMERA_DEFOCUS_ANGLE);

        // ONE BUFFER, not one per frame in flight, and the rule is step 3's:
        // group by update frequency. The camera does not move in this phase,
        // so this is written once at startup and read by every frame
        // thereafter -- there is nothing per-frame about it and nothing to
        // synchronise. Phase 7's camera controls are precisely what will turn
        // it into a per-frame buffer with a descriptor each, which is the
        // arrangement step 6 deleted.
        createBuffer(sizeof(CameraUniform),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     cameraUniformBuffer, cameraUniformMemory,
                     "Camera uniform buffer");

        if (vkMapMemory(device, cameraUniformMemory, 0, sizeof(CameraUniform),
                        0, &cameraUniformMapped) != VK_SUCCESS) {
            throw std::runtime_error(
                "failed to map the camera uniform buffer!");
        }

        memcpy(cameraUniformMapped, &cameraValues, sizeof(CameraUniform));

        std::cout << "Camera uniform buffer: " << sizeof(CameraUniform)
                  << " B, std140, host-visible and mapped at "
                  << cameraUniformMapped
                  << "\n\tvfov:          " << CAMERA_VFOV << " deg"
                  << ", focus dist " << CAMERA_FOCUS_DIST
                  << ", defocus angle " << CAMERA_DEFOCUS_ANGLE << " deg"
                  << "\n\tlens radius:   " << defocusRadius
                  << " world units"
                  << "\n\tviewport:      " << viewportWidth << " x "
                  << viewportHeight << " at " << CAMERA_FOCUS_DIST
                  << " along -w"
                  << "\n\tcenter:        " << cameraValues.center.x << ", "
                  << cameraValues.center.y << ", " << cameraValues.center.z
                  << "\n\tpixel00Loc:    " << cameraValues.pixel00Loc.x
                  << ", " << cameraValues.pixel00Loc.y << ", "
                  << cameraValues.pixel00Loc.z
                  << "\n\tpixelDeltaU:   " << cameraValues.pixelDeltaU.x
                  << ", " << cameraValues.pixelDeltaU.y << ", "
                  << cameraValues.pixelDeltaU.z
                  << "\n\tpixelDeltaV:   " << cameraValues.pixelDeltaV.x
                  << ", " << cameraValues.pixelDeltaV.y << ", "
                  << cameraValues.pixelDeltaV.z
                  << "\n\tbasis u:       " << cameraU.x << ", " << cameraU.y
                  << ", " << cameraU.z
                  << "\n\tbasis v:       " << cameraV.x << ", " << cameraV.y
                  << ", " << cameraV.z
                  << "\n\tbasis w:       " << cameraW.x << ", " << cameraW.y
                  << ", " << cameraW.z << "  (points eye-ward)"
                  << std::endl;
    }

    void createComputeDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding storageImageBinding{};
        storageImageBinding.binding = 0;

        // STORAGE_IMAGE, not COMBINED_IMAGE_SAMPLER and not SAMPLED_IMAGE.
        // The difference is write access: a sampled image can only be read,
        // whatever its layout, because the descriptor type is part of what
        // the driver checks. Declaring the wrong type here and calling
        // imageStore in the shader is a validation error, not a silent one --
        // this is the rare case where the type system reaches across.
        storageImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        storageImageBinding.descriptorCount = 1;
        storageImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        storageImageBinding.pImmutableSamplers = nullptr;

        // Phase 6 step 1. STORAGE_BUFFER, and the type is what grants write
        // access just as it did for the image above. UNIFORM_BUFFER -- the
        // type step 6 deleted -- is read-only to the shader however the
        // memory was allocated, and is size-limited besides: the guaranteed
        // maximum range for a uniform buffer is 16 KB, against 128 MB for a
        // storage buffer. 7.68 MB of probe would not fit in one.
        //
        // That size limit is the real reason the scene data arriving in
        // Phase 8 will be storage buffers rather than uniform ones, and this
        // binding is the first instance of the type.
        VkDescriptorSetLayoutBinding probeBufferBinding{};
        probeBufferBinding.binding = 1;
        probeBufferBinding.descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        probeBufferBinding.descriptorCount = 1;
        probeBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        probeBufferBinding.pImmutableSamplers = nullptr;

        // Phase 6 step 2. UNIFORM_BUFFER, the type step 6 deleted, back for
        // the camera. Read-only to the shader by its type -- which is the
        // whole difference from the STORAGE_BUFFER above, and the reason this
        // is the right type for a camera and the wrong one for the probe.
        VkDescriptorSetLayoutBinding cameraBinding{};
        cameraBinding.binding = 2;
        cameraBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cameraBinding.descriptorCount = 1;
        cameraBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        cameraBinding.pImmutableSamplers = nullptr;

        // The binding NUMBERS are what the shader's layout(binding = N)
        // matches, not the order in this array. They happen to agree here;
        // relying on that would be a bug waiting for the day they do not.
        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            storageImageBinding, probeBufferBinding, cameraBinding};

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                        &computeDescriptorSetLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error(
                "failed to create compute descriptor set layout!");
        }

        std::cout << "Compute descriptor set layout: " << bindings.size()
                  << " bindings"
                  << "\n\tbinding 0: STORAGE_IMAGE x1, COMPUTE stage"
                  << "\n\tbinding 1: STORAGE_BUFFER x1, COMPUTE stage"
                  << "\n\tbinding 2: UNIFORM_BUFFER x1, COMPUTE stage"
                  << "\n\tnamespace: separate from the graphics layout's"
                     " binding 0"
                  << std::endl;
    }

    void createComputePipeline() {
        auto computeShaderCode =
            readFile(std::string(SHADER_BINARY_DIR) + "/shader.comp.spv");
        VkShaderModule computeShaderModule =
            createShaderModule(computeShaderCode);

        // The same struct the vertex and fragment stages use, with a
        // different stage bit. A shader stage is a shader stage; what makes
        // this one compute is where it gets plugged in.
        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;

        // THE PUSH CONSTANT RANGE, and it is part of the LAYOUT rather than of
        // any set: push constants are not descriptors and have no pool, no
        // allocation and no backing memory. The layout declares an offset, a
        // size and which stages may read it, and vkCmdPushConstants later
        // writes the bytes straight into the command buffer.
        //
        // stageFlags has to name the stage that actually reads the block, and
        // vkCmdPushConstants has to be handed the same flags again. Neither
        // the C++ compiler nor glslc can see a mismatch between those two and
        // the shader's layout(push_constant) declaration.
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(float);

        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                   &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error(
                "failed to create compute pipeline layout!");
        }

        // THE WHOLE PIPELINE. A stage and a layout. Compare the graphics
        // equivalent, which also carries vertex input, input assembly,
        // viewport, rasterizer, multisample, depth/stencil, blend, dynamic
        // state and a render pass with a subpass index -- every one of them
        // machinery for turning triangles into pixels, and none of it on this
        // path. basePipelineHandle and basePipelineIndex exist here too, for
        // deriving one pipeline from another, and are left at their defaults.
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                     nullptr, &computePipeline) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        // The create-info sizes alone UNDERSTATE the asymmetry, and the
        // first run of this step is what showed it: 96 B against 144 B, a
        // ratio of 1.5 rather than the order of magnitude the comment above
        // implies. The reason is that VkGraphicsPipelineCreateInfo holds
        // POINTERS to its ten state structures, so its own size counts ten
        // pointers and not the state. The sum of the seven state structs
        // this pipeline actually fills is printed beside it, derived with
        // sizeof rather than counted, because a hand-totalled number here
        // would desynchronise the same way step 3's attribute list did.
        size_t graphicsStateBytes =
            sizeof(VkPipelineVertexInputStateCreateInfo) +
            sizeof(VkPipelineInputAssemblyStateCreateInfo) +
            sizeof(VkPipelineViewportStateCreateInfo) +
            sizeof(VkPipelineRasterizationStateCreateInfo) +
            sizeof(VkPipelineMultisampleStateCreateInfo) +
            sizeof(VkPipelineColorBlendStateCreateInfo) +
            sizeof(VkPipelineDynamicStateCreateInfo);
        std::cout << "Compute pipeline created:"
                  << "\n\tSPIR-V:      " << computeShaderCode.size()
                  << " B from shader.comp.spv"
                  << "\n\tstages:      1 (COMPUTE)"
                  << "\n\tcreate info: "
                  << sizeof(VkComputePipelineCreateInfo) << " B, against "
                  << sizeof(VkGraphicsPipelineCreateInfo)
                  << " B for the graphics form -- which is a struct of"
                     " POINTERS, and the 7 state structs it points at are a"
                     " further "
                  << graphicsStateBytes << " B"
                  << "\n\tfixed state: none -- there is no rasterizer to"
                     " configure"
                  << "\n\trender pass: none, and no subpass index either";

        // The limit beside the use, as with maxSamplerAnisotropy in step 2.
        // 128 bytes is the spec's guaranteed minimum and a great deal of
        // hardware offers exactly that, so a push-constant block is budgeted
        // in bytes from the start rather than discovered to be too small
        // later. Four of those bytes are now spent.
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        std::cout << "\n\tpush const:  " << pushConstantRange.size
                  << " B at offset " << pushConstantRange.offset
                  << ", COMPUTE stage, of "
                  << properties.limits.maxPushConstantsSize
                  << " B this device allows" << std::endl;

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    void createComputeDescriptorSet() {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &computeDescriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo,
                                     &computeDescriptorSet) != VK_SUCCESS) {
            throw std::runtime_error(
                "failed to allocate compute descriptor set!");
        }

        VkDescriptorImageInfo imageInfo{};

        // The promise again, and this time it is unconditionally true: the
        // storage image reached GENERAL in step 4 and never leaves it except
        // inside readbackImage, which puts it back. Nothing here transitions
        // anything -- see step 3's note, which is the same note.
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfo.imageView = storageImageView;

        // IGNORED for STORAGE_IMAGE, and left explicit rather than omitted.
        // The same struct served the combined image sampler in step 3 with
        // all three fields live; which fields matter is decided by the
        // descriptorType below, exactly as pBufferInfo/pImageInfo are.
        imageInfo.sampler = VK_NULL_HANDLE;

        // Phase 6 step 1. pBufferInfo rather than pImageInfo, and the
        // descriptorType below is what decides which of the two the driver
        // reads -- the same rule the ignored sampler field above follows.
        //
        // VK_WHOLE_SIZE rather than the byte count. The buffer is used in its
        // entirety and the range is not a thing the shader can see: the
        // unsized array in the shader's buffer block takes its length from
        // the bound range, so passing a short range here would silently
        // shorten draws[] rather than fail.
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = rayProbeBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = VK_WHOLE_SIZE;

        // Phase 6 step 2. The camera's range is sizeof(CameraUniform), not
        // VK_WHOLE_SIZE as the probe's is, and the difference is not
        // stylistic: a uniform buffer's range is bounded by
        // maxUniformBufferRange, which the spec guarantees only to 16 KB
        // against 128 MB for a storage buffer. Naming the exact size is how
        // that stays checkable. It is also why the scene data in Phase 8 has
        // to be storage buffers -- 7.68 MB of probe would not fit in a
        // uniform block at all.
        VkDescriptorBufferInfo cameraInfo{};
        cameraInfo.buffer = cameraUniformBuffer;
        cameraInfo.offset = 0;
        cameraInfo.range = sizeof(CameraUniform);

        std::array<VkWriteDescriptorSet, 3> writes{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = computeDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &imageInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = computeDescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &bufferInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = computeDescriptorSet;
        writes[2].dstBinding = 2;
        writes[2].dstArrayElement = 0;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &cameraInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);

        std::cout << "Compute descriptor set written:"
                  << "\n\tbinding 0 -> view " << storageImageView
                  << ", sampler none, layout "
                  << imageLayoutString(VK_IMAGE_LAYOUT_GENERAL)
                  << "\n\tbinding 1 -> buffer " << rayProbeBuffer
                  << ", offset 0, range VK_WHOLE_SIZE"
                  << "\n\tbinding 2 -> buffer " << cameraUniformBuffer
                  << ", offset 0, range " << sizeof(CameraUniform)
                  << "\n\tsets:       1, not one per frame -- neither the"
                     " image nor the buffer it points at ever changes"
                  << std::endl;
    }

    // Just the four commands, so that the startup self-test below and the
    // per-frame path in recordCommandBuffer record the IDENTICAL sequence
    // rather than two copies of it that can drift apart.
    void recordComputeDispatch(VkCommandBuffer commandBuffer, float time) {
        // CEILING DIVISION, not RENDER_WIDTH / WORKGROUP_SIZE. A dispatch can
        // only ask for whole workgroups, so any dimension that is not a
        // multiple of the group size needs one more group and a guard in the
        // shader for the invocations that fall outside. Both are in place even
        // though 800 and 600 are both multiples of 8 and neither does
        // anything today.
        uint32_t groupsX =
            (RENDER_WIDTH + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        uint32_t groupsY =
            (RENDER_HEIGHT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // VK_PIPELINE_BIND_POINT_COMPUTE. A command buffer holds graphics and
        // compute bind points INDEPENDENTLY: binding here disturbs nothing
        // bound at the graphics point, and the same is true of the descriptor
        // set below. That is what lets this be recorded into the same buffer
        // as the draw.
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computePipeline);

        // The bind point is given again, because the sets bound for compute
        // and the sets bound for graphics are separate state.
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                computePipelineLayout, 0, 1,
                                &computeDescriptorSet, 0, nullptr);

        // FOUR BYTES, straight into the command buffer. The stage flags must
        // match the range declared in the pipeline layout, and the layout is
        // passed rather than the pipeline -- the same distinction as
        // vkCmdBindDescriptorSets, for the same reason: the layout is the
        // schema the driver checks against.
        vkCmdPushConstants(commandBuffer, computePipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float),
                           &time);

        // Three numbers, and they are WORKGROUPS rather than invocations --
        // the single most common thing to get wrong about this call. Asking
        // for 800x600 here would launch 800*600*64 invocations, 64 times the
        // intended work, with 98% of them writing out of bounds and silently
        // discarded. It would still produce a correct-looking image.
        vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);
    }

    // A dependency and nothing else. oldLayout and newLayout are both GENERAL
    // -- the same value -- so this changes nothing about the image and exists
    // purely to order one stage's access against another's. Step 1's barriers
    // did a layout change and a dependency together; this is the dependency on
    // its own, which is the clearest way to see that the four mask fields were
    // always the real content.
    void recordStorageImageBarrier(VkCommandBuffer commandBuffer,
                                   VkPipelineStageFlags srcStage,
                                   VkAccessFlags srcAccess,
                                   VkPipelineStageFlags dstStage,
                                   VkAccessFlags dstAccess,
                                   const char* hazard) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = storageImage;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;

        // A barrier orders the commands recorded BEFORE it against those
        // recorded after it -- and, because submissions on one queue begin in
        // order, also against everything submitted earlier on that queue.
        // That second half is what lets the write-after-read barrier at the
        // top of a frame reach back to the PREVIOUS frame's fragment reads.
        vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr,
                             0, nullptr, 1, &barrier);

        // Logged once per barrier, on the first recorded frame, and derived
        // from the arguments it was actually called with.
        if (barriersLogged < 2) {
            barriersLogged++;
            std::cout << "Frame barrier " << barriersLogged << " (" << hazard
                      << "):"
                      << "\n\tsrc:    " << pipelineStageString(srcStage) << " / "
                      << accessFlagsString(srcAccess)
                      << "\n\tdst:    " << pipelineStageString(dstStage) << " / "
                      << accessFlagsString(dstAccess)
                      << "\n\tlayout: " << imageLayoutString(barrier.oldLayout)
                      << " -> " << imageLayoutString(barrier.newLayout)
                      << " (no transition -- a dependency only)" << std::endl;
        }
    }

    // THE STARTUP SELF-TEST, and the last user of the single-time envelope for
    // a dispatch. One dispatch at t = 0, where the shader's scroll reduces to
    // step 5's formula exactly, then the byte comparison that step 5
    // established. endSingleTimeCommands ends in vkQueueWaitIdle, so this path
    // still needs no barrier -- which is precisely why the per-frame path does.
    void dispatchCompute(float time) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        recordComputeDispatch(commandBuffer, time);
        endSingleTimeCommands(commandBuffer);

        uint32_t groupsX =
            (RENDER_WIDTH + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        uint32_t groupsY =
            (RENDER_HEIGHT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        std::cout << "Compute dispatched: " << groupsX << "x" << groupsY
                  << " workgroups of " << WORKGROUP_SIZE << "x"
                  << WORKGROUP_SIZE << " = "
                  << (groupsX * groupsY * WORKGROUP_SIZE * WORKGROUP_SIZE)
                  << " invocations for "
                  << (RENDER_WIDTH * RENDER_HEIGHT) << " texels, time "
                  << time << " s" << std::endl;

        verifyCameraRays(time, "startup self-test");
    }

    // THE HOST REFERENCE FOR THE SHADER'S GENERATOR, written out operator
    // for operator. These three functions and their counterparts in
    // shader.comp are one algorithm expressed twice, in two languages, built
    // by two compilers -- and the checkpoint below is the two copies checking
    // each other. That is trap 1 in its sharpest form, and there is no
    // arrangement that avoids it: nothing can be shared between a .comp and a
    // .cpp. An edit to one that is not mirrored in the other shows up as a
    // total mismatch rather than a subtle one, which is the only help the
    // situation offers.
    //
    // uint32_t is unsigned, so both multiplies wrap modulo 2^32 exactly as
    // GLSL's do. Unsigned overflow being DEFINED in C++ -- unlike signed
    // overflow, which is undefined behaviour -- is the property this leans on.
    static uint32_t hostNextUint(uint32_t& state) {
        state = state * 747796405u + 2891336453u;
        uint32_t word =
            ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    // The top 24 bits divided by 2^24, and the reason this comparison can be
    // exact at all. The numerator is below 2^24 so it converts to float
    // without loss; the denominator is a power of two so the multiply moves
    // the exponent and rounds nothing. No operation anywhere in the generator
    // is inexact, which is why the host can predict every bit. See trap 6 for
    // what the obvious version does instead.
    static float hostNextFloat(uint32_t& state) {
        return static_cast<float>(hostNextUint(state) >> 8) *
               (1.0f / 16777216.0f);
    }

    // pixel_sample_seed from include/raytracer/core/rng.h, which shader.comp
    // also transliterates. Ported with its collision flaw intact -- see the
    // file header for the measurement and the reasoning.
    static uint32_t hostPixelSampleSeed(int i, int j, int sampleIndex) {
        uint32_t seed = 0;
        seed ^= static_cast<uint32_t>(i) * 1973u;
        seed ^= static_cast<uint32_t>(j) * 9277u;
        seed ^= static_cast<uint32_t>(sampleIndex) * 26699u;
        seed += 1u;
        return seed;
    }

    // What getRay() should have produced, plus the draws it took on the way.
    // The draws are returned because they ARE part of what the function did:
    // the distribution tests below run on them, and the GPU inherits those
    // results because the rays agreeing is what establishes that both sides
    // drew the same numbers.
    struct HostRay {
        glm::vec3 origin;
        glm::vec3 direction;
        float time;
        float draws[5];
        int drawCount;
    };

    // getRay from shader.comp, transliterated back. See trap 1: one function
    // written twice. Two details are load-bearing and both are about the
    // random stream rather than the geometry.
    //
    // THE DRAWS ARE SEQUENCED INTO NAMED LOCALS, matching the shader, because
    // neither GLSL nor C++ defines the evaluation order of an expression's
    // operands. Written the compact way on both sides, the two compilers
    // could disagree about which draw is the x jitter and nothing would say
    // so -- the image would antialias correctly off a different stream.
    //
    // IT STARTS FROM cameraValues, the float block the shader reads, not from
    // the doubles those were narrowed from. Otherwise the comparison would be
    // measuring the narrowing as well as the arithmetic, and a tolerance
    // covering both could not tell them apart.
    HostRay hostGetRay(int i, int j, uint32_t& state) const {
        const float PI_F = 3.1415926535897932385f;

        HostRay out{};
        out.drawCount = 0;

        float jitterX = hostNextFloat(state);
        float jitterY = hostNextFloat(state);
        out.draws[out.drawCount++] = jitterX;
        out.draws[out.drawCount++] = jitterY;

        glm::vec3 pixelSample =
            cameraValues.pixel00Loc +
            (static_cast<float>(i) + (jitterX - 0.5f)) *
                cameraValues.pixelDeltaU +
            (static_cast<float>(j) + (jitterY - 0.5f)) *
                cameraValues.pixelDeltaV;

        out.origin = cameraValues.center;

        // The conditional pair. At a camera with no lens these two draws do
        // not happen and the shutter time below is the third number rather
        // than the fifth -- see trap 7.
        if (cameraValues.defocusAngle > 0.0f) {
            float diskR = hostNextFloat(state);
            float diskTheta = hostNextFloat(state);
            out.draws[out.drawCount++] = diskR;
            out.draws[out.drawCount++] = diskTheta;

            float r = std::sqrt(diskR);
            float theta = 2.0f * PI_F * diskTheta;
            out.origin = cameraValues.center +
                         (r * std::cos(theta)) * cameraValues.defocusDiskU +
                         (r * std::sin(theta)) * cameraValues.defocusDiskV;
        }

        float shutter = hostNextFloat(state);
        out.draws[out.drawCount++] = shutter;
        out.time = shutter;

        // Unnormalised, as the CPU leaves it.
        out.direction = pixelSample - out.origin;
        return out;
    }

    // THE CHECKPOINT FOR THIS STEP. Four independent questions, and the
    // reason there are four is that this step's arithmetic is no longer
    // exact: step 1's generator was all integers and a power-of-two scale, so
    // the host could predict every bit. A camera ray involves divisions, a
    // square root and a sine, and the Vulkan specification allows a couple of
    // units in the last place on those. Bit-exact comparison is therefore
    // gone, and a tolerance alone would be a weak replacement -- so the
    // tolerance is backed by invariants that hold exactly in geometry and
    // cannot be satisfied by a ray field that is merely close.
    //
    //   1. AGREEMENT. Every ray against hostGetRay, within a tolerance, which
    //      catches wrong arithmetic, a wrong uniform offset, a transposed
    //      index, and a draw order that disagrees with the host's.
    //
    //   2. PLANARITY, which is the strong one. Every ray direction must have
    //      a dot product with w of exactly -focus_dist. That is true because
    //      the viewport is a plane perpendicular to w at that distance, and
    //      because the lens offset lies in the u-v plane, so moving the
    //      origin across the lens does not change it. It is one number per
    //      pixel and it pins the basis orthogonality, the focus distance and
    //      the defocus disk's orientation all at once -- a camera that was
    //      skewed, mis-scaled or had its disk built from the wrong vectors
    //      fails it. No picture could show this.
    //
    //   3. SYMMETRY. Pixel-centre directions for mirrored pairs must sum to
    //      exactly -2 * focus_dist * w, from the host's own uniform values.
    //      This catches a flipped viewport axis, which planarity cannot: an
    //      upside-down camera is still perfectly planar.
    //
    //   4. DISTRIBUTION. Step 1's tests, on the five draws each ray consumes,
    //      so the generator stays gated now that the probe no longer carries
    //      raw draws.
    //
    // What it still cannot catch is a missing barrier. See trap 5.
    void verifyCameraRays(float time, const char* label) {
        // Geometry tolerances are absolute and in world units, where the
        // scene is about 13 across. A float holds ~7 significant digits, so a
        // single rounding at this magnitude is near 1e-6; a handful of
        // operations plus the fused multiply-adds a compiler may introduce
        // puts the expected disagreement around 1e-5. 1e-3 is therefore about
        // a hundredfold margin -- loose enough never to fail spuriously,
        // tight enough that a real error cannot hide, since the smallest
        // meaningful one here is a pixel's worth of direction at 6e-3.
        //
        // The observed maxima are printed, which is the part worth reading:
        // a threshold tells you a run passed, a maximum tells you how close
        // it came.
        constexpr double RAY_TOLERANCE = 1e-3;
        constexpr double PLANARITY_TOLERANCE = 1e-3;
        constexpr double SYMMETRY_TOLERANCE = 1e-3;
        constexpr double ORTHONORMAL_TOLERANCE = 1e-9;
        constexpr double MEAN_TOLERANCE = 2e-3;
        constexpr double VARIANCE_TOLERANCE = 1e-3;
        constexpr double CHI_SQUARE_LIMIT = 50.0;
        constexpr double SERIAL_LIMIT = 0.01;

        int sampleIndex = static_cast<int>(time * 60.0f);

        const size_t pixels =
            static_cast<size_t>(RENDER_WIDTH) * RENDER_HEIGHT;
        const float* probe = static_cast<const float*>(rayProbeMapped);

        // Promoted to double so the host's own checking arithmetic contributes
        // nothing to the numbers being judged.
        const glm::dvec3 w = cameraW;

        double maxRayError = 0.0;
        size_t worstRayPixel = 0;
        double maxPlanarityError = 0.0;
        size_t worstPlanarityPixel = 0;
        size_t sentinels = 0;

        size_t drawCount = 0;
        size_t outOfRange = 0;
        float minDraw = 2.0f;
        float maxDraw = -2.0f;
        double sum = 0.0;
        double sumSquares = 0.0;
        double serialSum = 0.0;
        size_t serialPairs = 0;
        std::array<size_t, RNG_BUCKETS> bucketCounts{};

        size_t wrongDrawCount = 0;

        for (uint32_t y = 0; y < RENDER_HEIGHT; y++) {
            for (uint32_t x = 0; x < RENDER_WIDTH; x++) {
                uint32_t state = hostPixelSampleSeed(
                    static_cast<int>(x), static_cast<int>(y), sampleIndex);
                HostRay expected = hostGetRay(static_cast<int>(x),
                                              static_cast<int>(y), state);

                if (expected.drawCount !=
                    static_cast<int>(RAY_DRAWS_WITH_LENS)) {
                    wrongDrawCount++;
                }

                size_t base = (static_cast<size_t>(y) * RENDER_WIDTH + x) *
                              PROBE_VEC4S_PER_PIXEL * 4;

                glm::dvec3 gotOrigin(probe[base + 0], probe[base + 1],
                                     probe[base + 2]);
                double gotTime = probe[base + 3];
                glm::dvec3 gotDirection(probe[base + 4], probe[base + 5],
                                        probe[base + 6]);

                // 1. AGREEMENT, as the largest single component error across
                // the origin, the direction and the shutter time.
                double error = 0.0;
                for (int c = 0; c < 3; c++) {
                    error = std::max(
                        error, std::abs(gotOrigin[c] -
                                        static_cast<double>(
                                            expected.origin[c])));
                    error = std::max(
                        error, std::abs(gotDirection[c] -
                                        static_cast<double>(
                                            expected.direction[c])));
                }
                error = std::max(
                    error,
                    std::abs(gotTime - static_cast<double>(expected.time)));

                if (error > maxRayError) {
                    maxRayError = error;
                    worstRayPixel =
                        static_cast<size_t>(y) * RENDER_WIDTH + x;
                }

                // 2. PLANARITY, on the GPU's own direction.
                double planarity =
                    std::abs(glm::dot(gotDirection, w) + CAMERA_FOCUS_DIST);
                if (planarity > maxPlanarityError) {
                    maxPlanarityError = planarity;
                    worstPlanarityPixel =
                        static_cast<size_t>(y) * RENDER_WIDTH + x;
                }

                if (probe[base + 0] == RAY_PROBE_SENTINEL &&
                    probe[base + 1] == RAY_PROBE_SENTINEL &&
                    probe[base + 2] == RAY_PROBE_SENTINEL) {
                    sentinels++;
                }

                // 4. DISTRIBUTION, on the draws this ray consumed.
                for (int d = 0; d < expected.drawCount; d++) {
                    float got = expected.draws[d];
                    drawCount++;

                    if (!(got >= 0.0f && got < 1.0f)) {
                        outOfRange++;
                    }
                    minDraw = std::min(minDraw, got);
                    maxDraw = std::max(maxDraw, got);
                    sum += got;
                    sumSquares += static_cast<double>(got) * got;

                    int bucket = static_cast<int>(got * RNG_BUCKETS);
                    bucket = std::max(0, std::min(RNG_BUCKETS - 1, bucket));
                    bucketCounts[static_cast<size_t>(bucket)]++;

                    if (d > 0) {
                        serialSum +=
                            static_cast<double>(expected.draws[d - 1]) * got;
                        serialPairs++;
                    }
                }
            }
        }

        double mean = sum / static_cast<double>(drawCount);
        double variance =
            sumSquares / static_cast<double>(drawCount) - mean * mean;

        double expectedPerBucket =
            static_cast<double>(drawCount) / RNG_BUCKETS;
        double chiSquare = 0.0;
        for (size_t b = 0; b < RNG_BUCKETS; b++) {
            double deviation =
                static_cast<double>(bucketCounts[b]) - expectedPerBucket;
            chiSquare += deviation * deviation / expectedPerBucket;
        }

        double serialR =
            (serialSum / static_cast<double>(serialPairs) - 0.25) * 12.0;

        // 3. SYMMETRY, host-side from the uniform. The pixel-CENTRE direction,
        // with no jitter and no lens offset, which is the deterministic part
        // of the camera and the only part a symmetry statement can be made
        // about.
        auto centreDirection = [this](uint32_t i, uint32_t j) {
            glm::dvec3 p(cameraValues.pixel00Loc);
            p += static_cast<double>(i) * glm::dvec3(cameraValues.pixelDeltaU);
            p += static_cast<double>(j) * glm::dvec3(cameraValues.pixelDeltaV);
            return p - glm::dvec3(cameraValues.center);
        };

        glm::dvec3 expectedSum = -2.0 * CAMERA_FOCUS_DIST * w;
        double maxSymmetryError = 0.0;
        const uint32_t symmetryProbes[5][2] = {
            {0, 0},
            {RENDER_WIDTH - 1, 0},
            {0, RENDER_HEIGHT - 1},
            {RENDER_WIDTH / 3, RENDER_HEIGHT / 7},
            {RENDER_WIDTH / 2, RENDER_HEIGHT / 2}};
        for (const auto& sp : symmetryProbes) {
            glm::dvec3 pairSum =
                centreDirection(sp[0], sp[1]) +
                centreDirection(RENDER_WIDTH - 1 - sp[0],
                                RENDER_HEIGHT - 1 - sp[1]);
            for (int c = 0; c < 3; c++) {
                maxSymmetryError = std::max(
                    maxSymmetryError, std::abs(pairSum[c] - expectedSum[c]));
            }
        }

        // The basis, in the double it was built in. Not a port check -- the
        // basis never leaves the host -- but if it is not orthonormal then
        // every test above is measuring against the wrong frame, so it is
        // worth one line of certainty before trusting the other three.
        double orthoError = 0.0;
        orthoError = std::max(orthoError, std::abs(glm::length(cameraU) - 1.0));
        orthoError = std::max(orthoError, std::abs(glm::length(cameraV) - 1.0));
        orthoError = std::max(orthoError, std::abs(glm::length(cameraW) - 1.0));
        orthoError = std::max(orthoError, std::abs(glm::dot(cameraU, cameraV)));
        orthoError = std::max(orthoError, std::abs(glm::dot(cameraU, cameraW)));
        orthoError = std::max(orthoError, std::abs(glm::dot(cameraV, cameraW)));

        // THE IMAGE, against the probe's direction rather than the host's, so
        // this tests the display path alone. Within 1, because the
        // float-to-UNORM tie rule is implementation-chosen.
        std::vector<uint8_t> image = readbackImage(
            storageImage, RENDER_WIDTH, RENDER_HEIGHT,
            VK_IMAGE_LAYOUT_GENERAL);

        size_t imageOff = 0;
        size_t alphaOff = 0;
        for (size_t pixel = 0; pixel < pixels; pixel++) {
            size_t base = pixel * PROBE_VEC4S_PER_PIXEL * 4;
            glm::vec3 direction(probe[base + 4], probe[base + 5],
                                probe[base + 6]);
            glm::vec3 shown = glm::normalize(direction) * 0.5f + 0.5f;

            for (int c = 0; c < 3; c++) {
                int want =
                    static_cast<int>(std::lround(shown[c] * 255.0f));
                int got = image[pixel * 4 + static_cast<size_t>(c)];
                if (got - want > 1 || want - got > 1) {
                    imageOff++;
                }
            }
            if (image[pixel * 4 + 3] != 255) {
                alphaOff++;
            }
        }

        std::cout << "Camera rays verified (" << label << "): " << pixels
                  << " rays read back at time " << time << " s, sample index "
                  << sampleIndex << std::endl;

        std::ios_base::fmtflags oldFlags = std::cout.flags();
        std::streamsize oldPrecision = std::cout.precision(9);
        std::cout << std::defaultfloat;

        std::cout << "\tagreement:   max component error " << maxRayError
                  << " at pixel " << worstRayPixel << "  (limit "
                  << RAY_TOLERANCE << ")"
                  << "\n\tplanarity:   max |dot(dir,w) + focus_dist| "
                  << maxPlanarityError << " at pixel " << worstPlanarityPixel
                  << "  (limit " << PLANARITY_TOLERANCE << ")"
                  << "\n\tsymmetry:    max mirrored-pair error "
                  << maxSymmetryError << " over "
                  << (sizeof(symmetryProbes) / sizeof(symmetryProbes[0]))
                  << " pairs  (limit " << SYMMETRY_TOLERANCE << ")"
                  << "\n\tbasis:       max orthonormality error "
                  << orthoError << "  (limit " << ORTHONORMAL_TOLERANCE << ")"
                  << "\n\tstream:      " << drawCount << " draws, "
                  << (drawCount / pixels) << " per ray, " << wrongDrawCount
                  << " rays took the wrong count, " << sentinels
                  << " still hold the sentinel"
                  << "\n\trange:       [" << minDraw << ", " << maxDraw
                  << "], " << outOfRange << " outside [0,1)"
                  << "\n\tmean:        " << mean << "  (0.5 +/- "
                  << MEAN_TOLERANCE << ")"
                  << "\n\tvariance:    " << variance << "  (0.0833333 +/- "
                  << VARIANCE_TOLERANCE << ")"
                  << "\n\tchi-square:  " << chiSquare << " over "
                  << RNG_BUCKETS << " buckets, " << (RNG_BUCKETS - 1)
                  << " df  (limit " << CHI_SQUARE_LIMIT << ")"
                  << "\n\tserial corr: " << serialR << "  (0 +/- "
                  << SERIAL_LIMIT << ")"
                  << "\n\timage:       " << imageOff << " of " << (pixels * 3)
                  << " RGB bytes differ from the probe by more than 1, "
                  << alphaOff << " alpha bytes not 255" << std::endl;

        const uint32_t probePixels[4][2] = {
            {0, 0},
            {RENDER_WIDTH - 1, 0},
            {0, RENDER_HEIGHT - 1},
            {RENDER_WIDTH / 2, RENDER_HEIGHT / 2}};
        for (const auto& pp : probePixels) {
            size_t base = (static_cast<size_t>(pp[1]) * RENDER_WIDTH + pp[0]) *
                          PROBE_VEC4S_PER_PIXEL * 4;
            std::cout << "\tpixel (" << pp[0] << "," << pp[1] << "): origin "
                      << probe[base + 0] << ", " << probe[base + 1] << ", "
                      << probe[base + 2] << "  dir " << probe[base + 4]
                      << ", " << probe[base + 5] << ", " << probe[base + 6]
                      << "  t " << probe[base + 3] << std::endl;
        }

        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);

        if (maxRayError > RAY_TOLERANCE) {
            throw std::runtime_error(
                "camera rays disagree with the host reference: max component "
                "error " + std::to_string(maxRayError) + " at pixel " +
                std::to_string(worstRayPixel));
        }
        if (maxPlanarityError > PLANARITY_TOLERANCE) {
            throw std::runtime_error(
                "ray directions are not planar at focus_dist: max error " +
                std::to_string(maxPlanarityError) + " at pixel " +
                std::to_string(worstPlanarityPixel));
        }
        if (maxSymmetryError > SYMMETRY_TOLERANCE) {
            throw std::runtime_error(
                "mirrored pixel pairs are not symmetric about w: max error " +
                std::to_string(maxSymmetryError));
        }
        if (orthoError > ORTHONORMAL_TOLERANCE) {
            throw std::runtime_error(
                "the camera basis is not orthonormal: max error " +
                std::to_string(orthoError));
        }
        if (wrongDrawCount != 0) {
            throw std::runtime_error(
                std::to_string(wrongDrawCount) +
                " ray(s) consumed the wrong number of draws");
        }
        if (outOfRange != 0) {
            throw std::runtime_error(
                "RNG produced " + std::to_string(outOfRange) +
                " value(s) outside [0,1) -- see trap 6 at the top of the file");
        }
        if (mean < 0.5 - MEAN_TOLERANCE || mean > 0.5 + MEAN_TOLERANCE) {
            throw std::runtime_error("RNG mean " + std::to_string(mean) +
                                     " is not 0.5 within tolerance");
        }
        double varianceError = variance - 1.0 / 12.0;
        if (varianceError < -VARIANCE_TOLERANCE ||
            varianceError > VARIANCE_TOLERANCE) {
            throw std::runtime_error("RNG variance " +
                                     std::to_string(variance) +
                                     " is not 1/12 within tolerance");
        }
        if (chiSquare > CHI_SQUARE_LIMIT) {
            throw std::runtime_error(
                "RNG histogram is not flat: chi-square " +
                std::to_string(chiSquare));
        }
        if (serialR < -SERIAL_LIMIT || serialR > SERIAL_LIMIT) {
            throw std::runtime_error(
                "consecutive RNG draws are correlated: r = " +
                std::to_string(serialR));
        }
        if (imageOff != 0 || alphaOff != 0) {
            throw std::runtime_error(
                "the storage image does not hold the probe's directions: " +
                std::to_string(imageOff) + " RGB byte(s) off by more than 1, " +
                std::to_string(alphaOff) + " alpha byte(s) not 255");
        }
    }

    void createDescriptorSetLayout() {
        // ONE binding, where step 3 had two. The uniform block went with the
        // quad, and what is left is the image the compute shader wrote.
        //
        // Renumbered from 1 to 0, which is tidiness and carries a real hazard:
        // this number must match `layout(binding = 0)` in shader.frag, in
        // another language and another file, and nothing checks that it does.
        // Step 5's WORKGROUP_SIZE trap wearing different clothes.
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 0;
        samplerLayoutBinding.descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        samplerLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &samplerLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                        &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error(
                "failed to create descriptor set layout!");
        }

        // ONE IMAGE IS NOW IN TWO SET LAYOUTS AT ONCE, under two different
        // descriptor types: STORAGE_IMAGE for the compute shader that writes
        // it, COMBINED_IMAGE_SAMPLER for the fragment shader that reads it.
        // Legal because step 4's usage flags declared both STORAGE and
        // SAMPLED, and because GENERAL is the one layout that permits a shader
        // write and a sampled read of the same image. This is the concrete
        // reason GENERAL had to be the resting layout rather than a stop on
        // the way somewhere better.
        std::cout << "Descriptor set layout created: "
                  << layoutInfo.bindingCount << " binding"
                  << "\n\tbinding 0:  COMBINED_IMAGE_SAMPLER x"
                  << samplerLayoutBinding.descriptorCount
                  << ", readable by FRAGMENT"
                  << "\n\tdescribes:  the storage image -- which the compute"
                     " layout also names, as STORAGE_IMAGE"
                  << "\n\tpoints at:  nothing -- a layout names no buffer,"
                     " image or sampler"
                  << std::endl;
    }
    void createDescriptorPool() {
        // Two types, one descriptor each. Both budgets SHRANK in step 6: the
        // per-frame uniform descriptors went with the per-frame buffers, and
        // the per-frame sets went with them, since pointing two identical
        // descriptors at one unchanging image is not a use of anything.
        std::array<VkDescriptorPoolSize, 4> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 1;

        // Phase 6 step 1. A THIRD TYPE, not a third set: the probe descriptor
        // goes into the compute set that already exists, so maxSets below is
        // unchanged at 2. The per-type budgets and the set count are
        // independent, and this is the clearest case of it so far -- one more
        // descriptor, no more sets.
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[2].descriptorCount = 1;

        // Phase 6 step 2. A FOURTH TYPE, still two sets: the camera
        // descriptor joins the same compute set. One, not MAX_FRAMES_IN_FLIGHT
        // of them, because the camera never changes in this phase -- which is
        // the opposite answer to the one step 3 reached for its uniform
        // buffer, from the same rule.
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[3].descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        // One graphics set and one compute set. Still two independent budgets:
        // this one, and the per-type counts above. Either being short is the
        // same VK_ERROR_OUT_OF_POOL_MEMORY from a different direction.
        poolInfo.maxSets = 2;

        // No flags. VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT would
        // permit vkFreeDescriptorSets on individual sets; without it the only
        // way to reclaim a set is to destroy or reset the whole pool, which is
        // all this needs -- the sets live exactly as long as the program.
        poolInfo.flags = 0;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr,
                                   &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }

        std::cout << "Descriptor pool created:"
                  << "\n\tcapacity:  " << poolInfo.maxSets << " sets, "
                  << poolSizes[0].descriptorCount
                  << " COMBINED_IMAGE_SAMPLER + "
                  << poolSizes[1].descriptorCount << " STORAGE_IMAGE + "
                  << poolSizes[2].descriptorCount << " STORAGE_BUFFER + "
                  << poolSizes[3].descriptorCount
                  << " UNIFORM_BUFFER descriptors"
                  << "\n\tbudgets:   per type, and independent of maxSets"
                  << "\n\tgrowth:    none -- a pool is fixed at creation"
                  << "\n\tfreeing:   sets are released with the pool, not"
                     " individually"
                  << std::endl;
    }
    // Allocating a set gives it the right SHAPE; it does not point anywhere.
    // vkUpdateDescriptorSets writes the handles in, and it is worth noticing
    // what it is NOT: not a command, not recorded, not submitted, no queue. It
    // mutates the set on the host, immediately.
    void createDescriptorSets() {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor set!");
        }

        // THE STORAGE IMAGE, not a texture. This one line is what puts the
        // compute shader's output on screen: the view handed to the fragment
        // shader is the view the dispatch wrote through.
        VkDescriptorImageInfo imageInfo{};

        // GENERAL, and it is the promise again. Step 3 declared
        // SHADER_READ_ONLY_OPTIMAL here because that is where the texture
        // rested; this image rests in GENERAL, so GENERAL is what has to be
        // declared. Nothing in this call checks the layout or changes it, and
        // declaring one the image is not in yields undefined texels.
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfo.imageView = storageImageView;

        // The step 2 sampler, unchanged and unaware. It referenced no image
        // then and references none now.
        imageInfo.sampler = textureSampler;

        VkWriteDescriptorSet samplerWrite{};
        samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        samplerWrite.dstSet = descriptorSet;
        samplerWrite.dstBinding = 0;
        samplerWrite.dstArrayElement = 0;
        samplerWrite.descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerWrite.descriptorCount = 1;

        // pImageInfo, as in step 3. The three payload pointers are mutually
        // exclusive and descriptorType picks which one is read.
        samplerWrite.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, 1, &samplerWrite, 0, nullptr);

        std::cout << "Descriptor set written:"
                  << "\n\tbinding 0 -> view "
                  << reinterpret_cast<const void*>(storageImageView)
                  << " + sampler "
                  << reinterpret_cast<const void*>(textureSampler)
                  << ", declared layout "
                  << imageLayoutString(VK_IMAGE_LAYOUT_GENERAL)
                  << "\n\tsame view:  as the compute set's binding 0, through"
                     " a different descriptor type"
                  << "\n\tsets:       1, where step 3 had one per frame in"
                     " flight"
                  << "\n\twritten:    on the host, no command buffer and no"
                     " submit"
                  << std::endl;
    }
    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;

        // PRIMARY: submittable to a queue, but cannot be called from another
        // command buffer. SECONDARY is the exact inverse, and is the
        // multithreading tool -- record chunks in parallel as secondary, then
        // have a single primary execute them.
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount =
            static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo,
                                     commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer,
                             uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error(
                "failed to begin recording command buffer!");
        }

        // Elapsed seconds, as a float, which is the value the push constant
        // carries and the value the post-loop checkpoint compares against.
        // Kept in a member before it is pushed so that the host and the shader
        // derive their scroll offset from bit-identical inputs.
        lastPushedTime = std::chrono::duration<float, std::ratio<1>>(
                             std::chrono::steady_clock::now() - startTime)
                             .count();

        // HAZARD 2, WRITE-AFTER-READ, and the one that is easy to miss. The
        // PREVIOUS frame's fragment shader may still be reading this image --
        // MAX_FRAMES_IN_FLIGHT is 2, so it is genuinely still in flight, and
        // nothing else orders these two: the fence only covers the submission
        // from two frames ago, and the semaphores order acquire and present
        // rather than shader stages.
        //
        // This needs only an EXECUTION dependency. Nothing has to become
        // visible to anyone; the new write simply must not begin before the
        // old reads finish. The access masks are stated anyway, because
        // naming the hazard is what makes the barrier reviewable.
        recordStorageImageBarrier(
            commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, "write-after-read, across frames");

        // The dispatch, recorded BEFORE vkCmdBeginRenderPass because compute
        // work and render passes do not nest. Same four commands the startup
        // self-test records, through the same helper.
        recordComputeDispatch(commandBuffer, lastPushedTime);

        // HAZARD 1, READ-AFTER-WRITE, within this frame. The fragment shader
        // below must not begin reading until the dispatch above has completed
        // AND its writes are visible -- which is both an execution dependency
        // and a MEMORY dependency, and the difference from the barrier above
        // is exactly that second half.
        recordStorageImageBarrier(
            commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT, "read-after-write, within a frame");

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;

        // Which of the three framebuffers to draw into. In step 9b this index
        // arrives from vkAcquireNextImageKHR rather than being passed in.
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        // The colour that loadOp = CLEAR in step 6a actually clears to.
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        // INLINE says the commands live in this primary buffer. The
        // alternative says they live in secondary buffers this one executes.
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                             VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          graphicsPipeline);

        // Obligatory rather than optional: step 5 declared these dynamic, so
        // the pipeline does not carry them and a draw without them is
        // undefined. Resize-without-rebuild was the benefit; this is the bill.
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // NO vkCmdBindVertexBuffers AND NO vkCmdBindIndexBuffer. There is
        // nothing to bind: the pipeline declares zero vertex bindings and zero
        // attributes, and shader.vert computes its three positions from
        // gl_VertexIndex.
        //
        // What that buys is not three vertices' worth of memory. It is that
        // the drawing path no longer has geometry in it at all, which is the
        // shape a path tracer wants -- the image is computed, and the graphics
        // pipeline exists only to put it on the screen.

        // Command buffer state, as the bindings it replaced were: the pipeline
        // was compiled against the LAYOUT, and this chooses the contents for
        // this draw.
        //
        // pipelineLayout is passed, not the pipeline, and not to bind anything
        // -- it is the schema the driver checks this set against for
        // compatibility. The 0 after it is firstSet, the index in the layout's
        // array of set layouts; the path tracer's second and third sets will
        // be 1 and 2 here. The trailing 0 and nullptr are dynamic offsets,
        // which belong to DYNAMIC descriptor types and let one buffer serve
        // many draws at different offsets -- a real alternative to one buffer
        // per frame, not taken here.
        //
        // Not indexed any more. There was one set per frame while each held a
        // different uniform buffer; one unchanging image needs one set, and
        // every frame binds the same one.
        //
        // VK_PIPELINE_BIND_POINT_GRAPHICS, where dispatchCompute passes
        // VK_PIPELINE_BIND_POINT_COMPUTE for its own set. The two bind points
        // are independent state in the same command buffer -- which is what
        // step 7 relies on when it records a dispatch and this draw together.
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &descriptorSet, 0,
                                nullptr);

        // Back to vkCmdDraw, and the 3 is a VERTEX COUNT WITH NO BUFFER
        // BEHIND IT. The draw call decides how many times the vertex shader
        // runs; gl_VertexIndex is the loop counter handed to each invocation,
        // and nothing requires that counter to address memory. After the
        // count: instanceCount, firstVertex, firstInstance.
        //
        // Three vertices covering the whole screen, because the triangle is
        // bigger than the screen -- (-1,-1), (3,-1), (-1,3). One primitive, no
        // index buffer, no diagonal seam across the middle of the image, and
        // the hardware clips away the 75% that falls outside for nothing.
        //
        // Its winding is clockwise in Vulkan's y-down framebuffer space,
        // matching what the quad's triangles had, so it survives the
        // VK_CULL_MODE_BACK_BIT the rasterizer still sets. A reversed
        // fullscreen triangle is a blank window with a clean validation log,
        // which is the suspect the rasterizer's own comment names.
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        // Created already signalled. Without this the first frame's
        // vkWaitForFences blocks forever: nothing has been submitted yet, so
        // nothing will ever signal it. The program hangs at startup with no
        // error and no validation message -- pure silence.
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                                  &imageAvailableSemaphores[i]) !=
                    VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr,
                              &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error(
                    "failed to create synchronization objects for a frame!");
            }
        }

        std::cout << "Sync objects created:\n"
                  << "\timageAvailable semaphores: "
                  << imageAvailableSemaphores.size()
                  << " (one per frame in flight)\n"
                  << "\trenderFinished semaphores: "
                  << renderFinishedSemaphores.size()
                  << " (one per swapchain image)\n"
                  << "\tinFlight fences:           "
                  << inFlightFences.size()
                  << " (one per frame in flight), pre-signalled\n"
                  << "\tcommand buffers:           "
                  << commandBuffers.size() << " (one per frame in flight)"
                  << std::endl;
    }

    // Per swapchain IMAGE, so these share the images' lifetime: built with
    // them here and destroyed with them in cleanupSwapChain(). A rebuild could
    // in principle change the image count, and a vector kept from before would
    // then be indexed out of bounds by imageIndex.
    void createRenderFinishedSemaphores() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        renderFinishedSemaphores.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                                  &renderFinishedSemaphores[i]) !=
                VK_SUCCESS) {
                throw std::runtime_error(
                    "failed to create render-finished semaphores!");
            }
        }
    }

    // Everything whose shape depends on the swapchain, gathered so a rebuild
    // can throw away exactly this and nothing else.
    void cleanupSwapChain() {
        // Framebuffers reference the image views, so they go first.
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        // Views reference images the swapchain owns, so before the swapchain.
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        // The swapchain's VkImages go with it; they need no calls of their own.
        vkDestroySwapchainKHR(device, swapChain, nullptr);

        // After the swapchain, whose destruction releases the presentation
        // engine's hold on these.
        for (auto semaphore : renderFinishedSemaphores) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
    }

    void recreateSwapChain() {
        // A minimised window reports a framebuffer of 0x0, and a swapchain
        // with a zero extent is invalid -- creation would simply fail. Wait
        // here until the window comes back. glfwWaitEvents blocks until an
        // event arrives, unlike glfwPollEvents, so a minimised app idles
        // rather than spinning a core.
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        // Nothing still in flight may be using the objects about to go. Brute
        // force, and acceptable only because recreation is rare.
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        // Deliberately NOT rebuilt: the render pass and the pipeline. Step 5
        // made viewport and scissor dynamic, so the pipeline bakes in no size.
        // Step 6b's render-pass compatibility is decided by format, and a
        // resize does not change the format. Moving between an SDR and an HDR
        // display could, which would break that -- not a case handled here.
        createSwapChain();
        createImageViews();
        createRenderFinishedSemaphores();
        createFramebuffers();

        std::cout << "Swapchain recreated at " << swapChainExtent.width << "x"
                  << swapChainExtent.height << std::endl;
    }

    // SHADER_BINARY_DIR is baked in by CMakeLists.txt, so this path does not
    // depend on which directory the program is launched from.
    static std::vector<char> readFile(const std::string& filename) {
        // ate = start at the end, so tellg() gives the size straight away
        // without a separate stat. binary stops newline translation from
        // corrupting the bytecode on platforms that would do it.
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file: " + filename);
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();

        // codeSize counts bytes but pCode is uint32_t*, because SPIR-V is a
        // stream of 32-bit words. The cast is safe only because vector's
        // allocator guarantees alignment for any scalar type -- the same cast
        // off a raw char array on the stack would be undefined behaviour.
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr,
                                 &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    void createGraphicsPipeline() {
        auto vertShaderCode =
            readFile(std::string(SHADER_BINARY_DIR) + "/shader.vert.spv");
        auto fragShaderCode =
            readFile(std::string(SHADER_BINARY_DIR) + "/shader.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;

        // The entry point to run. A SPIR-V module can hold several, which is
        // why this is named rather than implied; glslc only ever emits "main".
        // pSpecializationInfo is left null, but it is how constants get baked
        // in at pipeline-creation time -- the natural home for the path
        // tracer's MAX_DEPTH later, instead of recompiling GLSL.
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {
            vertShaderStageInfo,
            fragShaderStageInfo
        };

        std::cout << "Shader stages prepared: "
                  << sizeof(shaderStages) / sizeof(shaderStages[0]) << "\n"
                  << "\tvertex:   " << vertShaderCode.size()
                  << " bytes SPIR-V, entry point \""
                  << shaderStages[0].pName << "\"\n"
                  << "\tfragment: " << fragShaderCode.size()
                  << " bytes SPIR-V, entry point \""
                  << shaderStages[1].pName << "\"" << std::endl;

        // ---- Fixed-function state ----

        // The escape hatch from pipeline immutability: these two are left out
        // of the baked object and supplied at record time with
        // vkCmdSetViewport / vkCmdSetScissor. Doing it here is what makes the
        // window resizing in step 11 survivable -- otherwise every resize
        // would mean rebuilding the whole pipeline.
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount =
            static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // EMPTY, AND STILL REQUIRED. Zeroed rather than omitted: a pipeline
        // always has a vertex input stage, and declaring that it reads nothing
        // is a different thing from not declaring it at all.
        //
        // Zero bindings and zero attributes is what makes a vertex buffer
        // unnecessary. The vertex shader is still invoked three times -- a
        // draw always runs it once per vertex -- it simply has no attributes
        // to be handed, and derives its position from gl_VertexIndex.
        //
        // The rule from step 1 still holds and now cuts the other way:
        // declaring attributes the shader ignores is legal, but a shader INPUT
        // with no matching attribute fails pipeline creation outright. Which
        // is why shader.vert had to lose its three `in` declarations in the
        // same step this lost its descriptions.
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

        // What the vertices form. TRIANGLE_LIST: every three vertices are one
        // triangle, with no reuse between them.
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // The counts stay baked in even though the values are dynamic: the
        // pipeline must know how many viewports to expect, just not where.
        //
        // Viewport and scissor are easy to conflate and do different jobs. The
        // viewport is a transformation -- it maps clip space onto framebuffer
        // pixels, so a smaller one SCALES the image down. The scissor is a
        // filter -- fragments outside it are discarded, so a smaller one CROPS.
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

        // Fragments past the near/far planes are discarded rather than
        // clamped. Clamping is a shadow-mapping trick and needs a GPU feature.
        rasterizer.depthClampEnable = VK_FALSE;

        // VK_TRUE would stop geometry reaching the rasterizer at all, which
        // disables output entirely.
        rasterizer.rasterizerDiscardEnable = VK_FALSE;

        // LINE (wireframe) and POINT each need a GPU feature enabled.
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

        // Anything above 1.0 needs the wideLines feature.
        rasterizer.lineWidth = 1.0f;

        // Back-face culling. Worth remembering as a suspect: if step 9b shows
        // a blank window and validation says nothing, a winding-order mismatch
        // here is the classic cause -- the triangle gets culled in silence.
        // Our three vertices wind clockwise in Vulkan's Y-down framebuffer
        // space, so they survive this.
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

        rasterizer.depthBiasEnable = VK_FALSE;

        // Off. It antialiases polygon edges by sampling a pixel several times
        // and needs a GPU feature. The roadmap skips multisampling outright,
        // because a path tracer antialiases by jittering rays within the pixel
        // -- which falls out of the sampling it already does.
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Blending off, so a fragment's colour replaces what the framebuffer
        // held. The write mask still has to name the channels to write.
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        // A bitwise combination instead of blending. Enabling it would switch
        // off the per-attachment blending above entirely.
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        // The shaders' uniform interface: which descriptor sets and push
        // constants they can see. Created empty since Phase 2, because the two
        // shaders read nothing from outside. It now carries set 0's layout --
        // the first time this pipeline declares access to memory a shader
        // reaches by itself, and the slot the path tracer's BVH, primitive,
        // material, camera and image bindings will join.
        //
        // The LAYOUT, never a set: the pipeline is compiled against the schema,
        // and which actual buffer fills it is command-buffer state decided per
        // draw. Same separation as the vertex binding slot versus
        // vkCmdBindVertexBuffers, and the reason one pipeline can draw many
        // meshes.
        //
        // Declaring more than the shaders use is legal, which is what makes
        // this step invisible: slot 0 exists here, and shader.vert still does
        // not mention it. Step 2 changes that.
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                   &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        std::cout << "Pipeline layout created; fixed-function state described:"
                  << "\n\tdynamic state: viewport + scissor ("
                  << dynamicState.dynamicStateCount << " states)"
                  << "\n\tvertex input:  "
                  << vertexInputInfo.vertexBindingDescriptionCount
                  << " bindings, "
                  << vertexInputInfo.vertexAttributeDescriptionCount
                  << " attributes -- the triangle computes its own positions";

        std::cout << "\n\ttopology:      TRIANGLE_LIST, "
                  << viewportState.viewportCount << " viewport"
                  << "\n\trasterizer:    FILL, cull BACK, front face CLOCKWISE"
                  << "\n\tmultisample:   "
                  << multisampling.rasterizationSamples << " sample"
                  << "\n\tblending:      off, "
                  << colorBlending.attachmentCount << " attachment"
                  << "\n\tuniforms:      "
                  << pipelineLayoutInfo.setLayoutCount
                  << " descriptor set layout at set 0, "
                  << pipelineLayoutInfo.pushConstantRangeCount
                  << " push constant ranges"
                  << std::endl;

        // ---- The pipeline object itself ----

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;

        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;

        // Null because there is no depth or stencil buffer. The roadmap skips
        // depth buffering outright: a path tracer resolves depth by ray
        // distance, so a z-buffer has nothing to contribute.
        pipelineInfo.pDepthStencilState = nullptr;

        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;

        // A handle, not a pointer: the layout is a real object, unlike the
        // state structs above which are consumed and forgotten.
        pipelineInfo.layout = pipelineLayout;

        // The render pass is a COMPATIBILITY reference, not an exclusive tie.
        // This pipeline works with any render pass compatible with this one,
        // where compatibility means matching attachment formats and sample
        // counts rather than being the same object. That, together with
        // viewport and scissor being dynamic from step 5, is what lets step
        // 11a rebuild the swapchain on a resize without rebuilding this.
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        // Pipeline derivatives: a new pipeline sharing most of its state with
        // an existing one can be cheaper to create. Unused, and it would need
        // VK_PIPELINE_CREATE_DERIVATIVE_BIT in flags to have any effect.
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        // Plural, and it takes an array, because real engines create many
        // pipelines at once. The VK_NULL_HANDLE is a VkPipelineCache slot -- a
        // blob the driver can serialise to disk and reload, so shader
        // compilation is paid once ever instead of at every launch. That is
        // the actual fix for the compilation stutter games are known for.
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                      nullptr, &graphicsPipeline) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        std::cout << "Graphics pipeline created:"
                  << "\n\tstages:      " << pipelineInfo.stageCount
                  << "\n\tsubpass:     " << pipelineInfo.subpass
                  << " of the render pass above"
                  << "\n\tdepth test:  none (a path tracer uses ray distance)"
                  << "\n\tcache:       none (compiled fresh this launch)"
                  << std::endl;

        // Destroyed here, not in cleanup(). A shader module is only an input
        // to pipeline creation; once the pipeline exists it holds the compiled
        // machine code and these are dead weight. Reverse order of creation,
        // for the same reason as everywhere else.
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    // A queue family is a group of queues that all accept the same kinds of
    // work. Commands are recorded into buffers and submitted to a queue, so
    // we need a family whose advertised capabilities cover what we intend to
    // submit -- here, graphics and compute.
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(
            device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                indices.graphicsAndComputeFamily = i;
            }

            // No queueFlags bit exists for this. Whether a family can present
            // is a property of the family *and* a particular surface -- on a
            // two-GPU machine the display may be wired to the other one -- so
            // it is a question about the pair, not a capability to read off.
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
                                                 &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    // Returning VK_FALSE means "do not abort the call that triggered this".
    // VK_TRUE is reserved for testing the layers themselves.
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage
                  << std::endl;

        return VK_FALSE;
    }

    // Called twice, as most Vulkan enumeration functions are: once with a
    // null pointer to learn the count, once more to fill a sized buffer.
    void listAvailableExtensions() {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(
            nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(
            nullptr, &extensionCount, extensions.data());

        std::cout << extensionCount << " available instance extensions:\n";
        for (const auto& extension : extensions) {
            std::cout << '\t' << extension.extensionName << '\n';
        }
    }

    void drawFrame() {
        // Wait for the previous frame's GPU work. UINT64_MAX means no timeout.
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                        UINT64_MAX);

        // Returns an index immediately -- the image is NOT ready yet. The call
        // says which image you will get; the semaphore says when it is
        // actually yours. That gap is why imageAvailableSemaphore exists.
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(
            device, swapChain, UINT64_MAX,
            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE,
            &imageIndex);

        // OUT_OF_DATE: the swapchain no longer matches the surface and cannot
        // be drawn into, so rebuild it and skip this frame. SUBOPTIMAL is a
        // SUCCESS code and is deliberately let through here: an image has
        // already been acquired and its semaphore is committed to being
        // signalled, so carrying on is the consistent choice. It gets caught
        // after present instead, once the frame is finished.
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Fences do NOT reset themselves: a binary semaphore is unsignalled by
        // the wait that consumes it, but a fence stays signalled until told
        // otherwise. Reset only here, once a submit is certain. Step 9b did it
        // before the acquire, which deadlocks the moment the early return
        // above fires -- the fence is left unsignalled with no submission
        // coming to signal it, so the next wait on this slot never returns.
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // Allowed because the pool was created with
        // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT in step 8.
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {
            imageAvailableSemaphores[currentFrame]
        };

        // Per-STAGE waiting, not a blanket block: the vertex shader may run
        // before the image is available, and only the colour write has to
        // wait. The stage matches the subpass dependency from step 9a -- both
        // sides of the handshake have to name the same stage or the layout
        // transition can still race the acquire.
        VkPipelineStageFlags waitStages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        // Indexed by the acquired image, not by frame: see the member
        // declaration for why one shared semaphore is a spec violation.
        VkSemaphore signalSemaphores[] = {
            renderFinishedSemaphores[imageIndex]
        };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        // The fence is signalled when this submission completes, which is
        // exactly what the wait at the top of the next frame is waiting for.
        if (vkQueueSubmit(graphicsAndComputeQueue, 1, &submitInfo,
                          inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        // So a half-drawn image is never displayed.
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        // By now the frame is finished, so rebuilding is clean for both codes.
        // The flag is consulted alongside them because OUT_OF_DATE is NOT
        // guaranteed on a resize: the spec lets a driver keep presenting a
        // mismatched swapchain and return VK_SUCCESS, which would leave the
        // image stretched with nothing to signal it.
        if (result == VK_ERROR_OUT_OF_DATE_KHR ||
            result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        // Hand the next frame the other slot, so its command buffer, semaphore
        // and fence are untouched by the work just submitted.
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

        reportFrameTiming();
    }

    void reportFrameTiming() {
        using clock = std::chrono::steady_clock;

        frameCounter++;

        if (frameCounter == 1) {
            std::cout << "First frame presented." << std::endl;
            frameWindowStart = clock::now();
            return;
        }

        if (frameCounter % 200 == 0) {
            auto now = clock::now();
            double ms =
                std::chrono::duration<double, std::milli>(
                    now - frameWindowStart).count() / 200.0;

            std::cout << std::fixed << std::setprecision(2)
                      << "frame " << frameCounter << ": " << ms
                      << " ms avg, " << (1000.0 / ms) << " fps" << std::endl;

            frameWindowStart = now;
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        // The loop exits the instant the window closes, but the GPU may still
        // be executing the last frame. Destroying objects it is using would be
        // a use-after-free. Brute-force, and correct only because this is
        // shutdown -- never acceptable in a hot path.
        vkDeviceWaitIdle(device);
    }

    void cleanup() {
        // The rule that actually matters: destroy dependents before whatever
        // they reference. Reverse creation order, which the earlier steps
        // followed, is one convenient way to satisfy it; grouping the
        // swapchain objects below is another. Strict reverse order no longer
        // holds, but every reference still dies before its target.

        // Per-frame sync objects. Nothing references them.
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        // The pool before the framebuffers: freeing it frees the command
        // buffers, so nothing recorded still refers to a framebuffer or
        // pipeline destroyed below. Command buffers need no call of their own.
        vkDestroyCommandPool(device, commandPool, nullptr);

        // Before the image below, because both sets inside it hold that
        // image's handles -- the file's rule is that a reference dies before
        // its target. Destroying the pool frees every set in it, so the two
        // set handles need no calls of their own.
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        // The sampler first, though it is the one object here that references
        // nothing and could go at any point. The view genuinely must precede
        // the image it interprets, and the image its own memory.
        //
        // Both descriptor sets named this one image -- the graphics set as a
        // sampled image, the compute set as a storage image -- and both died
        // with the pool above.
        vkDestroySampler(device, textureSampler, nullptr);

        // The probe buffer, whose descriptor died with the pool just above.
        // Unmapping before freeing is NOT required -- vkFreeMemory unmaps
        // implicitly -- and is done anyway, because a live mapping into freed
        // memory is exactly the class of thing the validation layers are on
        // for, and a pointer that cannot be dereferenced is better than one
        // that can be dereferenced into nothing.
        vkUnmapMemory(device, rayProbeMemory);
        rayProbeMapped = nullptr;
        vkDestroyBuffer(device, rayProbeBuffer, nullptr);
        vkFreeMemory(device, rayProbeMemory, nullptr);

        // The camera uniform buffer, by the same reasoning.
        vkUnmapMemory(device, cameraUniformMemory);
        cameraUniformMapped = nullptr;
        vkDestroyBuffer(device, cameraUniformBuffer, nullptr);
        vkFreeMemory(device, cameraUniformMemory, nullptr);

        vkDestroyImageView(device, storageImageView, nullptr);
        vkDestroyImage(device, storageImage, nullptr);
        vkFreeMemory(device, storageImageMemory, nullptr);

        // Framebuffers, image views, the swapchain and its per-image
        // render-finished semaphores. The framebuffers reference the render
        // pass, which is why this comes before it.
        cleanupSwapChain();

        // The pipeline references its layout and the render pass, so it goes
        // before both.
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        // The compute path, by the same rule: pipeline, then the layout it
        // was built from, then the set layout that layout names. Its
        // descriptor set died with the pool destroyed earlier, along with the
        // graphics ones.
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout,
                                     nullptr);

        // After the pipeline layout built from it, by the same rule as
        // everything else here. A layout is only a schema, so once that
        // pipeline layout is gone nothing of ours still refers to it.
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        // The device made everything above; the messenger, surface and
        // instance outlive it.
        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        // Before the instance it was created from. Omitting this does not
        // crash: validation simply reports a leaked VkSurfaceKHR at
        // vkDestroyInstance, which is the whole point of having the layers on.
        vkDestroySurfaceKHR(instance, surface, nullptr);

        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main() {
    vulkan_app app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
