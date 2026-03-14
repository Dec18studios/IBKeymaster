cmake_minimum_required(VERSION 3.20)

project(IBKeyer LANGUAGES CXX)

set(PLUGIN_BASENAME "IBKeyer")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  # keeps universal builds predictable for Intel + Apple Silicon users on macOS 13.
  set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target")
endif()

if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "macOS architectures to build")
endif()

if(WIN32)
  set(IBKEYER_ENABLE_CUDA_DEFAULT ON)
elseif(UNIX AND NOT APPLE)
  set(IBKEYER_ENABLE_CUDA_DEFAULT ON)
else()
  set(IBKEYER_ENABLE_CUDA_DEFAULT OFF)
endif()

option(IBKEYER_ENABLE_CUDA "Build the CUDA backend into the plugin" ${IBKEYER_ENABLE_CUDA_DEFAULT})

set(OFX_SUPPORT_SOURCES
  Support/Library/ofxsCore.cpp
  Support/Library/ofxsImageEffect.cpp
  Support/Library/ofxsInteract.cpp
  Support/Library/ofxsLog.cpp
  Support/Library/ofxsMultiThread.cpp
  Support/Library/ofxsParams.cpp
  Support/Library/ofxsProperty.cpp
  Support/Library/ofxsPropertyValidation.cpp
)

set(PLUGIN_SOURCES
  IBKeymaster/IBKeyer.cpp
  IBKeymaster/IBKeyerBackend.cpp
  ${OFX_SUPPORT_SOURCES}
)

if(IBKEYER_ENABLE_CUDA AND (WIN32 OR (UNIX AND NOT APPLE)))
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
  endif()
  if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  endif()
  find_package(CUDAToolkit REQUIRED)
  list(APPEND PLUGIN_SOURCES IBKeymaster/IBKeyerCuda.cu)
endif()

if(APPLE)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  list(APPEND PLUGIN_SOURCES IBKeymaster/MetalKernel.mm)
endif()

add_library(${PLUGIN_BASENAME} SHARED ${PLUGIN_SOURCES})

target_include_directories(${PLUGIN_BASENAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/OpenFX-1.4/include
  ${CMAKE_CURRENT_SOURCE_DIR}/Support/include
  ${CMAKE_CURRENT_SOURCE_DIR}/IBKeymaster
)

target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
  _CRT_SECURE_NO_WARNINGS
)

if(IBKEYER_ENABLE_CUDA AND (WIN32 OR (UNIX AND NOT APPLE)))
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
    OFX_SUPPORTS_CUDARENDER
  )
  target_include_directories(${PLUGIN_BASENAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(MSVC)
  target_compile_options(${PLUGIN_BASENAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /EHsc /bigobj /MP>
  )
endif()

if(WIN32 AND IBKEYER_ENABLE_CUDA)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart_static)
elseif(UNIX AND NOT APPLE AND IBKEYER_ENABLE_CUDA)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart)
  target_link_options(${PLUGIN_BASENAME} PRIVATE -static-libstdc++ -static-libgcc)
endif()

if(APPLE)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework CoreFoundation"
    "-framework AppKit"
  )
endif()

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  OUTPUT_NAME "${PLUGIN_BASENAME}"
)

if(WIN32)
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/IBKeymaster/${PLUGIN_BASENAME}.ofx.bundle/Contents/Win64")
elseif(APPLE)
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/IBKeymaster/${PLUGIN_BASENAME}.ofx.bundle/Contents/MacOS")
else()
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/IBKeymaster/${PLUGIN_BASENAME}.ofx.bundle/Contents/Linux-x86-64")
endif()

file(MAKE_DIRECTORY "${OFX_OUTPUT_DIR}")

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${OFX_OUTPUT_DIR}"
)
