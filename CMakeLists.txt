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

# ---- Shader compilation -----------------------------------------------------
#
# Vulkan consumes SPIR-V bytecode, never GLSL source, so every shader is
# compiled ahead of time by glslc. The find_package(Vulkan) above already
# located it and exposed it as Vulkan_GLSLC_EXECUTABLE, so nothing extra has
# to be discovered here.
#
# Two deviations from the tutorial, which runs glslc from a hand-written shell
# script and loads the output from a path relative to the working directory:
#
#   1. The build compiles the shaders. They regenerate whenever the GLSL
#      changes, never need committing, and there is no manual step to forget.
#   2. The output lands in build/shaders/ rather than the source tree, and its
#      location is handed to the compiler as SHADER_BINARY_DIR. The executable
#      then finds its shaders no matter which directory it is launched from.

set(SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADER_SOURCES
    shaders/shader.vert
    shaders/shader.frag
)

# Named <source>.spv rather than the tutorial's vert.spv/frag.spv, so the
# source each one came from stays visible once there are more than two.
set(SPIRV_BINARIES)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV ${SHADER_BINARY_DIR}/${SHADER_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPIRV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
        VERBATIM
    )

    list(APPEND SPIRV_BINARIES ${SPIRV})
endforeach()

add_custom_target(shaders DEPENDS ${SPIRV_BINARIES})
add_dependencies(vulkan_backend shaders)

target_compile_definitions(vulkan_backend PRIVATE
    SHADER_BINARY_DIR="${SHADER_BINARY_DIR}"
)
