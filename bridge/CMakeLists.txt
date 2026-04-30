cmake_minimum_required(VERSION 3.16)
project(ai_trader_bridge LANGUAGES CXX)

add_executable(consumer examples/consumer.cpp)
target_include_directories(consumer PRIVATE include)
target_compile_features(consumer PRIVATE cxx_std_17)
target_link_libraries(consumer PRIVATE rt pthread)

