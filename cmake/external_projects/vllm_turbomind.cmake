#
# TurboMind SM70 GEMM kernels from lmdeploy
#
# Provides SM70 (V100) AWQ and W8A16 quantization kernels from TurboMind.
#
# Source is resolved in this order:
#   1. VLLM_TURBOMIND_SRC_DIR environment variable (local development)
#   2. FetchContent from focalcrest/lmdeploy (pinned commit on sm70/main)
#

# Environment variable takes precedence
if(DEFINED ENV{VLLM_TURBOMIND_SRC_DIR})
  set(VLLM_TURBOMIND_SRC_DIR $ENV{VLLM_TURBOMIND_SRC_DIR})
endif()

if(VLLM_TURBOMIND_SRC_DIR)
  set(vllm-turbomind_SOURCE_DIR "${VLLM_TURBOMIND_SRC_DIR}")
  message(STATUS "Using TurboMind from VLLM_TURBOMIND_SRC_DIR: ${vllm-turbomind_SOURCE_DIR}")
else()
  include(FetchContent)
  FetchContent_Declare(
    vllm-turbomind
    GIT_REPOSITORY https://github.com/focalcrest/lmdeploy.git
    GIT_TAG        1173a0c4fb839c538877593bf1bef92293eb52ea
    GIT_PROGRESS   TRUE
    SOURCE_DIR     ${CMAKE_BINARY_DIR}/vllm-turbomind
  )
  FetchContent_Populate(vllm-turbomind)
  message(STATUS "TurboMind fetched to ${vllm-turbomind_SOURCE_DIR}")
endif()
