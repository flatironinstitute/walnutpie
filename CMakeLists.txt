cmake_minimum_required(VERSION 3.15)
project(walnutpie
  VERSION 0.1.0
  DESCRIPTION "Header-only Walnut Pie sampler library"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)


##########################
## Top-Level Options   ##
##########################
option(WALNUTPIE_BUILD_EXAMPLES "Build the example targets for the library" ON)
option(WALNUTPIE_BUILD_TESTS "Build the test targets for the library" OFF)
option(WALNUTPIE_USE_MIMALLOC "Statically link mimalloc" OFF)
option(WALNUTPIE_COVERAGE "Enable source-based test coverage instrumentation" OFF)

##########################
## Global Dependencies  ##
##########################
include(FetchContent)

FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        5.0.1
)
FetchContent_MakeAvailable(eigen)


set(WALNUTPIE_MIMALLOC_TAG "v2.2.3" CACHE STRING "Version of mimalloc to use")
if(WALNUTPIE_USE_MIMALLOC)
    set(MI_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(MI_PADDING      OFF CACHE BOOL "" FORCE)
    set(MI_DEBUG_FULL   OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
            mimalloc
            GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
            GIT_TAG ${WALNUTPIE_MIMALLOC_TAG}
    )
    FetchContent_MakeAvailable(mimalloc)
endif()


add_library(BridgeStan::BridgeStan INTERFACE IMPORTED)
target_include_directories(BridgeStan::BridgeStan INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/bridgestan"
)

#############################
## Windows patches         ##
#############################
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_link_options(-static-libgcc -static-libstdc++ -static -lpthread)
endif()

###############################
## Making Walnutpie Library  ##
###############################
add_library(walnutpie INTERFACE)
target_include_directories(walnutpie INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # for builds
  $<INSTALL_INTERFACE:include>                            # for installs
)

target_link_libraries(walnutpie INTERFACE Eigen3::Eigen)

if (WALNUTPIE_USE_MIMALLOC)
    target_link_libraries(walnutpie INTERFACE mimalloc-static)
endif()


##########################
##       Examples       ##
##########################
if (WALNUTPIE_BUILD_EXAMPLES)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/examples")
endif()

##########################
##       Python         ##
##########################
# scikit_build_core sets SKBUILD when it is invoking cmake
if (SKBUILD)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/python")
endif ()

##########################
##       Tests         ##
##########################
if (WALNUTPIE_BUILD_TESTS)
    enable_testing()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()

##########################
##       Format         ##
##########################
# Define list of files to format (you can make this recursive if desired)
file(GLOB_RECURSE ALL_SOURCE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

find_program(CLANG_FORMAT_BIN clang-format)

if (CLANG_FORMAT_BIN)
    add_custom_target(
    format
    COMMAND ${CLANG_FORMAT_BIN}
      -i
      -style=file
      ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on source files"
  )
else()
    message(WARNING "clang-format not found. 'format' target will not be available.")
endif()
