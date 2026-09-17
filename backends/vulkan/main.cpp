// Vulkan backend entry point.
//
// Phase 2, step 11a -- vulkan-tutorial.com "Swap chain recreation":
// https://vulkan-tutorial.com/Drawing_a_triangle/Swap_chain_recreation
//
// A swapchain is built for one exact surface state, and goes stale when the
// window resizes or moves to a display with a different scale factor. Vulkan
// reports that as OUT_OF_DATE or SUBOPTIMAL from acquire and present; this
// rebuilds only the swapchain-shaped objects in response. The window is still
// non-resizable -- step 11b turns that on.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
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
// it on any device that advertises it, and the tutorial does not emphasise
// that -- omitting it trips validation errors later rather than here.
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
// Deviation from the tutorial, per the roadmap: it looks for a graphics-only
// family. This backend will submit compute work too, and on Apple Silicon one
// family serves both, so both bits are required here rather than revisiting
// this in Phase 3.
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

class vulkan_app {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
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

    // Deliberate deviation: the tutorial uses ONE render-finished semaphore.
    // That violates VUID-vkQueueSubmit-pSignalSemaphores-00067, and current
    // validation layers catch it. A present's wait on a binary semaphore is
    // only known to have completed once that image is re-acquired, so reusing
    // one semaphore across three images can signal it while the presentation
    // engine still holds it. One per swapchain image fixes it, because image
    // i coming round again is exactly the proof its last present finished.
    //
    // Step 10 deliberately leaves this one alone. The tutorial converts it to
    // MAX_FRAMES_IN_FLIGHT indexed by currentFrame there, which reintroduces
    // exactly the violation above -- frames in flight and image count are
    // independent numbers, and this one belongs to the image.
    //
    // Belonging to the image also means sharing its lifetime: these are
    // rebuilt with the swapchain in recreateSwapChain(), not kept alongside
    // the per-frame objects.
    std::vector<VkSemaphore> renderFinishedSemaphores;

    // Only so the frame rate can be reported. FIFO is vsync, so the average
    // should settle on the display's refresh interval rather than running
    // free -- which is the cheapest confirmation that frames really present.
    uint32_t frameCounter = 0;
    std::chrono::steady_clock::time_point frameWindowStart;

    void initWindow() {
        glfwInit();

        // GLFW was written for OpenGL and creates an OpenGL context by
        // default. Vulkan has no context -- we manage every resource
        // ourselves -- so that default has to be switched off.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // Resizing invalidates the swap chain and needs recreation logic,
        // which arrives in Phase 2. Disabled until then.
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
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
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
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

        // The tutorial sets device-level layers here "for compatibility with
        // older implementations". That advice is now out of date: the current
        // spec requires enabledLayerCount to be 0
        // (VUID-VkDeviceCreateInfo-enabledLayerCount-12384). Device layers
        // were removed, not just deprecated; the instance-level layers cover
        // device calls.
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

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];

            // How to read it: a plain 2D texture, in the format the swapchain
            // was created with -- which is why step 2c kept that around.
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            // The swizzle. IDENTITY throughout means no channel remapping;
            // this is where you would broadcast red to every channel for a
            // monochrome view without touching the image underneath.
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            // Which part of the image the view covers: colour data rather than
            // depth or stencil, no mipmaps, and one array layer -- which has
            // to agree with the imageArrayLayers = 1 set in step 2c.
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr,
                                  &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }

        std::cout << "Image views created: " << swapChainImageViews.size()
                  << " for " << swapChainImages.size()
                  << " swapchain image(s)" << std::endl;
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

        // 3 vertices, 1 instance, starting from vertex 0 and instance 0. That
        // firstVertex is where gl_VertexIndex begins counting in shader.vert.
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

        // Empty on purpose: shader.vert hardcodes its three vertices, so
        // nothing arrives from a buffer. Phase 3's "Vertex input description"
        // step is exactly where this stops being empty.
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

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
        // and needs a GPU feature. The roadmap skips the Multisampling chapter
        // outright, because a path tracer antialiases by jittering rays within
        // the pixel -- which falls out of the sampling it already does.
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
        // constants they can see. Empty, because these two read nothing from
        // outside. Phase 4 is where the path tracer's BVH, primitive,
        // material, camera and image bindings get declared right here.
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
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
                  << " bindings (vertices live in the shader)"
                  << "\n\ttopology:      TRIANGLE_LIST, "
                  << viewportState.viewportCount << " viewport"
                  << "\n\trasterizer:    FILL, cull BACK, front face CLOCKWISE"
                  << "\n\tmultisample:   "
                  << multisampling.rasterizationSamples << " sample"
                  << "\n\tblending:      off, "
                  << colorBlending.attachmentCount << " attachment"
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
        // the Depth buffering chapter outright: a path tracer resolves depth
        // by ray distance, so a z-buffer has nothing to contribute.
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
        if (result == VK_ERROR_OUT_OF_DATE_KHR ||
            result == VK_SUBOPTIMAL_KHR) {
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

        // Framebuffers, image views, the swapchain and its per-image
        // render-finished semaphores. The framebuffers reference the render
        // pass, which is why this comes before it.
        cleanupSwapChain();

        // The pipeline references its layout and the render pass, so it goes
        // before both.
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
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
