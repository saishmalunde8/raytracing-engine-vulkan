// Vulkan backend entry point.
//
// Phase 2, step 5 -- vulkan-tutorial.com "Fixed functions":
// https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions
//
// Describes every part of the pipeline that is not programmable. A VkPipeline
// is immutable, so nearly all render state is baked in at creation and the
// driver optimises once instead of patching shaders at draw time. Viewport
// and scissor are the exception, left dynamic so a resize does not force a
// pipeline rebuild. Creates one object: the VkPipelineLayout.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

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

    // The shaders' uniform interface -- empty for now. Unlike the shader
    // modules, this outlives pipeline creation, so it is freed in cleanup().
    VkPipelineLayout pipelineLayout;

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
        createGraphicsPipeline();
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

        // Becomes meaningful in step 11a, where a resize builds a replacement
        // swapchain and passes the outgoing one in here.
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

        // Step 6a adds the render pass here; 6b then feeds all of the above,
        // plus shaderStages, into vkCreateGraphicsPipelines.

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

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        // Reverse creation order throughout. The layout was made after the
        // views, so it goes before them.
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        // The views go before the swapchain, since
        // they reference images it owns; the swapchain before the device that
        // made it; the messenger next; the instance outlives everything made
        // through it. The swapchain's VkImages need no calls of their own --
        // they go with the swapchain.
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);

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
