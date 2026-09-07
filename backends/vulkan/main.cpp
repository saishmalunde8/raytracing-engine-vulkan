// Vulkan backend entry point.
//
// Phase 2, step 1b -- vulkan-tutorial.com "Window surface":
// https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Window_surface
//
// Finds a queue family that can present to the surface and opens the device
// with one queue per unique family. Presentation is not a queueFlags bit --
// there is no VK_QUEUE_PRESENT_BIT -- because the answer depends on which
// surface, so it has to be asked about the family/surface pair.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
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
        return indices.isComplete();
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
        // Reverse creation order. The device was made from the instance and
        // owns the queue, so it goes first; the messenger next; the instance
        // outlives everything made through it.
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
