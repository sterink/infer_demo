cmake_minimum_required(VERSION 3.10) 

# set the project name 
project(demo VERSION 1.0) 

add_definitions(-D_GLIBCXX_USE_CXX11_ABI)
add_compile_options(-g)

# specify the C++ standard 
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED True) 

find_package(spdlog REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

include_directories(${JSONCPP_INCLUDEDIR})
link_libraries(${JSONCPP_LIBRARIES})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CUDA_PATH "/usr/local/corex" CACHE PATH "cuda installation root path")
link_directories (${PROJECT_SOURCE_DIR}/lib)
link_directories (${CUDA_PATH}/lib)
include_directories(${PROJECT_SOURCE_DIR}/include ${CUDA_PATH}/include)

#LINK_DIRECTORIES(/usr/local/Cellar/jsoncpp/1.9.5/lib)
# add the executable 
add_executable(demo demo.cpp util_infer.cpp corex_infer.cpp dummy_engine.cpp) 

message(${JSONCPP_LIBRARIES})
message(${JSONCPP_INCLUDEDIR})
target_link_libraries(demo PRIVATE ${JSONCPP_LIBRARIES} spdlog::spdlog Threads::Threads libixrt.so cudart)
