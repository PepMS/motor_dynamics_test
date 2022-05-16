cmake_minimum_required(VERSION 3.11.0)
project(motor_dynamics_tests VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(crocoddyl REQUIRED)
find_package(pinocchio REQUIRED)
find_package(eagle_mpc REQUIRED)

add_executable(takeoff takeoff.cpp)

target_include_directories(takeoff PUBLIC 
                           ${CROCODDYL_INCLUDE_DIRECTORIES} 
                           ${PINOCCHIO_INCLUDE_DIRECTORIES}
                           ${EXAMPLE_ROBOT_DATA_INCLUDE_DIRECTORIES})

target_link_libraries(takeoff crocoddyl::crocoddyl pinocchio::pinocchio eagle_mpc::eagle_mpc)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
