cmake_minimum_required(VERSION 3.4)

add_executable (mnist_network main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../MNIST.cpp)

target_include_directories (mnist_network PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/)
target_compile_features(mnist_network PRIVATE cxx_range_for)