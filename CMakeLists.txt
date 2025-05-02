cmake_minimum_required(VERSION 3.25)
project(gaming LANGUAGES CXX)

find_package(absl REQUIRED)
find_package(glm REQUIRED)
find_package(GTest REQUIRED)
find_package(absl REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wpedantic -g)

add_executable(engine src/main.cpp)
target_link_libraries(engine
  PRIVATE
    absl::base
    absl::log
    absl::log_initialize
    absl::status
    absl::inlined_vector
    absl::statusor)

enable_testing()

# add_executable(test_property_tree tests/test_property_tree.cpp)
#
# target_link_libraries(test_property_tree
#   PRIVATE
#     sunset
#     GTest::GTest
#     GTest::Main
# )
#
# add_test(NAME TestPropertyTree COMMAND test_property_tree)
