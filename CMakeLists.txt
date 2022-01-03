
cmake_minimum_required(VERSION 3.14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

set(PROJ sycl-workaround)

project(${PROJ})

find_package(IntelDPCPP REQUIRED)

add_executable(${PROJ} main.cc)

target_compile_options(${PROJ}
  PRIVATE
  -std=c++20)

# target_link_libraries(${PROJ}
#   PRIVATE
#   OpenCL)

add_custom_target(run
  DEPENDS ${PROJ}
  COMMAND ${PROJ})
