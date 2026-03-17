cmake_minimum_required(VERSION 3.25)

include(FetchContent)
project(PyGpuBench CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80;90")

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBSECCOMP REQUIRED IMPORTED_TARGET libseccomp)

FetchContent_Declare(
        nanobind
        QUIET
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.9.2
)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")

find_package(Python COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)
FetchContent_MakeAvailable(nanobind)
nanobind_add_module(_pygpubench
        STABLE_ABI
        csrc/binding.cpp
        csrc/manager.cpp
        csrc/clear_l2.cu
        csrc/check.cu
        csrc/landlock.cpp
        csrc/obfuscate.cpp
)
target_link_libraries(_pygpubench PUBLIC Python::Module CUDA::cudart PkgConfig::LIBSECCOMP)
# set a bunch of hardening options to make it harder to tamper with the executable
target_compile_options(_pygpubench PUBLIC -fPIC -pie -fstack-protector-strong -fcf-protection=full -ftrivial-auto-var-init=zero -Wl,-z,relro,-z,now )
target_compile_definitions(_pygpubench PUBLIC -D_GLIBCXX_ASSERTIONS -D_FORTIFY_SOURCE=3)

# make sure we pick up cuda libraries from within the current python env, if available
set(RPATH_LIST
        "$ORIGIN"
        "$ORIGIN/../nvidia/cuda_runtime/lib"
)
list(JOIN RPATH_LIST ":" WHEEL_RPATH)

set_target_properties(_pygpubench PROPERTIES
        INSTALL_RPATH "${WHEEL_RPATH}"
        BUILD_WITH_INSTALL_RPATH OFF
        INSTALL_RPATH_USE_LINK_PATH FALSE
)

install(TARGETS _pygpubench
        LIBRARY DESTINATION pygpubench
)
