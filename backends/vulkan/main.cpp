// Vulkan backend entry point.
//
// Phase 0 scaffolding only. This does not render anything and creates no
// Vulkan objects -- it exists to prove the toolchain is wired up: the Vulkan
// headers are found, the loader links, and GLFW initialises. Real Vulkan
// setup (instance, device, queues) starts in Phase 1.

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <iostream>

int main() {
    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        return 1;
    }

    // The one Vulkan call in this file. Without it nothing would actually
    // require symbols from libvulkan, and the link would prove nothing.
    uint32_t api_version = 0;
    if (vkEnumerateInstanceVersion(&api_version) != VK_SUCCESS) {
        std::cerr << "vkEnumerateInstanceVersion failed\n";
        glfwTerminate();
        return 1;
    }

    std::cout << "Vulkan loader API version: "
              << VK_API_VERSION_MAJOR(api_version) << '.'
              << VK_API_VERSION_MINOR(api_version) << '.'
              << VK_API_VERSION_PATCH(api_version) << '\n';

    glfwTerminate();
    return 0;
}
