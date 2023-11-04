project(mathematical_robotics CXX)

cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20) 

include(GNUInstallDirs)

find_package(Eigen3 REQUIRED)
find_package(Matplot++ REQUIRED)

include_directories(../optimizer)

set(libs 
    Eigen3::Eigen
    Matplot++::matplot
)

add_executable(optimization "optimization.cpp")
target_link_libraries(optimization 
    ${libs}
)