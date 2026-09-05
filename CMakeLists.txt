cmake_minimum_required(VERSION 3.16)

project(raytracer_vulkan_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This file builds the Vulkan backend only. The CPU renderer under src/ is
# still built with the single clang++ line documented in the README, so the
# two backends share no build settings and cannot break each other.

# Reads the VULKAN_SDK environment variable set by the SDK's setup-env.sh.
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(vulkan_backend
    backends/vulkan/main.cpp
)

# Imported targets carry their own include paths, libraries and rpath, which
# is why no -I/-L/-rpath flags appear anywhere in this file.
target_link_libraries(vulkan_backend PRIVATE
    Vulkan::Vulkan
    glfw
)
