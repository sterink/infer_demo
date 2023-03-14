cmake_minimum_required(VERSION 3.10) 

# set the project name 
project(demo VERSION 1.0) 

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
add_compile_options(-g)

# specify the C++ standard 
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED True) 

find_package(spdlog REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CUDA_PATH "/usr/local/corex" CACHE PATH "cuda installation root path")
link_directories (${PROJECT_SOURCE_DIR}/lib)
link_directories (${CUDA_PATH}/lib)
include_directories(${CUDA_PATH}/include)

# add the executable 
add_executable(demo demo.cpp util_infer.cpp corex_infer.cpp dummy_engine.cpp tdnn_engine.cpp)

target_link_libraries(demo PRIVATE ${JSONCPP_LIBRARIES} spdlog::spdlog Threads::Threads ixrt cudart)
