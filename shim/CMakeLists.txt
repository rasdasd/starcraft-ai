cmake_minimum_required(VERSION 3.20)
project(bwshim CXX)

# ---------------------------------------------------------------------------
# Options
#
#   SHIM_BACKEND = bwapi   : vendored BWAPI 4.4.0 client library (Windows, StarCraft 1.16.1).
#                            Builds `shim.exe` (client mode) and `shim_module.dll` (AI module).
#   SHIM_BACKEND = openbw  : OpenBW's BWAPI fork. Client mode is not supported there, so only
#                            `libshim_module.so` (AI module loaded by BWAPILauncher) is built.
#                            Requires OPENBW_BWAPI_SRC (path to openbw/bwapi checkout) and
#                            OPENBW_BWAPI_BUILD (its CMake build dir, for libBWAPILIB).
#   FLATC_EXECUTABLE       : path to a flatc matching the FlatBuffers headers. If empty, flatc
#                            is built from the fetched FlatBuffers sources.
# ---------------------------------------------------------------------------
set(SHIM_BACKEND "bwapi" CACHE STRING "bwapi | openbw")
set_property(CACHE SHIM_BACKEND PROPERTY STRINGS bwapi openbw)
set(OPENBW_BWAPI_SRC "" CACHE PATH "openbw/bwapi source checkout (SHIM_BACKEND=openbw)")
set(OPENBW_BWAPI_BUILD "" CACHE PATH "openbw/bwapi build directory (SHIM_BACKEND=openbw)")
set(FLATC_EXECUTABLE "" CACHE FILEPATH "flatc binary; built from source if empty")
set(FLATBUFFERS_TAG "v25.12.19" CACHE STRING "FlatBuffers release tag (must match tools/flatc)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(SHIM_BACKEND STREQUAL "bwapi" AND NOT WIN32)
  message(FATAL_ERROR "SHIM_BACKEND=bwapi only builds on Windows (needs the StarCraft process). Use -DSHIM_BACKEND=openbw on Linux.")
endif()
if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND SHIM_BACKEND STREQUAL "bwapi")
  message(FATAL_ERROR "BWAPI 4.4.0 is 32-bit only. Configure with a Win32 toolchain (vcvarsall x86 / -A Win32).")
endif()

# ---------------------------------------------------------------------------
# FlatBuffers (headers + optionally flatc)
# ---------------------------------------------------------------------------
include(FetchContent)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_STATIC_FLATC OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_SHAREDLIB OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_ENABLE_PCH OFF CACHE BOOL "" FORCE)
if(FLATC_EXECUTABLE)
  set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
else()
  set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(flatbuffers
  URL https://github.com/google/flatbuffers/archive/refs/tags/${FLATBUFFERS_TAG}.zip
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(flatbuffers)

if(FLATC_EXECUTABLE)
  set(_flatc "${FLATC_EXECUTABLE}")
  set(_flatc_dep "")
else()
  set(_flatc "$<TARGET_FILE:flatc>")
  set(_flatc_dep flatc)
endif()

set(SHIM_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
set(SHIM_SCHEMA ${CMAKE_CURRENT_SOURCE_DIR}/../proto/bw.fbs)
add_custom_command(
  OUTPUT ${SHIM_GEN_DIR}/bw_generated.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHIM_GEN_DIR}
  COMMAND ${_flatc} --cpp --scoped-enums --gen-object-api --cpp-std c++17 -o ${SHIM_GEN_DIR} ${SHIM_SCHEMA}
  DEPENDS ${SHIM_SCHEMA} ${_flatc_dep}
  COMMENT "flatc: generating bw_generated.h"
  VERBATIM)
add_custom_target(shim_proto DEPENDS ${SHIM_GEN_DIR}/bw_generated.h)

# ---------------------------------------------------------------------------
# BWAPI backend
# ---------------------------------------------------------------------------
if(SHIM_BACKEND STREQUAL "bwapi")
  set(BWAPI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/bwapi)
  # BWAPI's own sources target XP; keep those defs private to the vendored libs.
  set(BWAPI_DEFS NOMINMAX WIN32 _WIN32_WINNT=0x0501 NTDDI_VERSION=0x05010300 _CRT_SECURE_NO_WARNINGS)

  file(GLOB BWAPILIB_SRC ${BWAPI_ROOT}/BWAPILIB/Source/*.cpp ${BWAPI_ROOT}/BWAPILIB/UnitCommand.cpp)
  add_library(BWAPILIB STATIC ${BWAPILIB_SRC})
  target_include_directories(BWAPILIB PUBLIC ${BWAPI_ROOT}/include)
  target_compile_definitions(BWAPILIB PRIVATE ${BWAPI_DEFS} PUBLIC NOMINMAX)

  file(GLOB BWAPICLIENT_SRC ${BWAPI_ROOT}/BWAPIClient/Source/*.cpp ${BWAPI_ROOT}/Shared/*.cpp)
  add_library(BWAPIClient STATIC ${BWAPICLIENT_SRC})
  target_include_directories(BWAPIClient
    PUBLIC ${BWAPI_ROOT}/include ${BWAPI_ROOT}/include/BWAPI/Client
    PRIVATE ${BWAPI_ROOT}/Shared)
  target_compile_definitions(BWAPIClient PRIVATE ${BWAPI_DEFS})
  target_link_libraries(BWAPIClient PUBLIC BWAPILIB)

  set(SHIM_BWAPI_LIBS BWAPILIB)
  set(SHIM_BWAPI_INCLUDE ${BWAPI_ROOT}/include)
  set(SHIM_BACKEND_NAME "bwapi-4.4.0")
else()
  if(NOT OPENBW_BWAPI_SRC OR NOT OPENBW_BWAPI_BUILD)
    message(FATAL_ERROR "SHIM_BACKEND=openbw needs -DOPENBW_BWAPI_SRC=<openbw/bwapi checkout> -DOPENBW_BWAPI_BUILD=<its build dir>")
  endif()
  find_library(OPENBW_BWAPILIB NAMES BWAPILIB libBWAPILIB PATHS ${OPENBW_BWAPI_BUILD}/lib NO_DEFAULT_PATH)
  if(NOT OPENBW_BWAPILIB)
    message(FATAL_ERROR "libBWAPILIB not found under ${OPENBW_BWAPI_BUILD}/lib - build openbw/bwapi first")
  endif()
  set(SHIM_BWAPI_LIBS ${OPENBW_BWAPILIB})
  set(SHIM_BWAPI_INCLUDE ${OPENBW_BWAPI_SRC}/bwapi/include)
  set(SHIM_BACKEND_NAME "openbw")
endif()

# ---------------------------------------------------------------------------
# BWEM (map analysis). Vendored community fork; mapPrinter is compiled out.
# ---------------------------------------------------------------------------
set(BWEM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/bwem-community/BWEM)
file(GLOB BWEM_SRC ${BWEM_DIR}/src/*.cpp)
add_library(BWEM STATIC ${BWEM_SRC})
target_include_directories(BWEM PUBLIC ${BWEM_DIR}/include)
target_include_directories(BWEM PRIVATE ${SHIM_BWAPI_INCLUDE})
target_link_libraries(BWEM PUBLIC ${SHIM_BWAPI_LIBS})
target_compile_definitions(BWEM PRIVATE NOMINMAX)
if(MSVC)
  target_compile_options(BWEM PRIVATE /W3 /EHsc /MP /wd4244 /wd4267 /wd4996 /permissive-)
  target_compile_definitions(BWEM PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(SHIM_BACKEND STREQUAL "openbw" AND NOT MSVC)
  target_compile_options(BWEM PRIVATE -include cstdint)
endif()

# ---------------------------------------------------------------------------
# Shim core (backend-agnostic: serializer, command applier, transport)
# ---------------------------------------------------------------------------
add_library(shim_core STATIC
  src/Transport.cpp
  src/Serializer.cpp
  src/CommandApplier.cpp
  src/Bridge.cpp
  src/MapAnalysis.cpp)
add_dependencies(shim_core shim_proto)
target_include_directories(shim_core
  PUBLIC src ${SHIM_GEN_DIR} ${SHIM_BWAPI_INCLUDE} ${flatbuffers_SOURCE_DIR}/include)
target_compile_definitions(shim_core PUBLIC
  SHIM_BACKEND_NAME="${SHIM_BACKEND_NAME}"
  SHIM_VERSION="0.1.0")
if(SHIM_BACKEND STREQUAL "openbw")
  target_compile_definitions(shim_core PUBLIC SHIM_OPENBW=1)
  if(NOT MSVC)
    # The fork's Game.h uses uint32_t without <cstdint>; GCC >= 13 no longer includes it transitively.
    target_compile_options(shim_core PUBLIC -include cstdint)
  endif()
endif()
target_link_libraries(shim_core PUBLIC ${SHIM_BWAPI_LIBS} BWEM)
if(WIN32)
  target_link_libraries(shim_core PUBLIC ws2_32)
  target_compile_definitions(shim_core PUBLIC NOMINMAX WIN32 _WIN32_WINNT=0x0601 _CRT_SECURE_NO_WARNINGS)
else()
  target_link_libraries(shim_core PUBLIC pthread)
endif()
if(MSVC)
  # /wd4244: FlatBuffers headers mix 64-bit SizeT with 32-bit size_t on x86 builds.
  target_compile_options(shim_core PUBLIC /W3 /EHsc /MP /bigobj /wd4244)
else()
  target_compile_options(shim_core PUBLIC -Wall -Wextra -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
if(SHIM_BACKEND STREQUAL "bwapi")
  add_executable(shim src/main_client.cpp)
  target_link_libraries(shim PRIVATE shim_core BWAPIClient)
endif()

add_library(shim_module SHARED src/main_module.cpp)
set_target_properties(shim_module PROPERTIES PREFIX "")   # shim_module.dll / shim_module.so
target_link_libraries(shim_module PRIVATE shim_core)
