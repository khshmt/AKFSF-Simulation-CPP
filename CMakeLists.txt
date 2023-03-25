cmake_minimum_required(VERSION 3.2)
project(AKFSFSimulation)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 CONFIG REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_ttf CONFIG REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SRC_CPP_DEFAULT
    src/main.cpp
    src/display.cpp
    src/beacons.cpp
    src/sensors.cpp
    src/simulation.cpp
    src/utils.cpp
)

add_library(LKF src/LKF/kalmanfilter_lkf.cpp include/kalmanfilter.h)
target_link_libraries(LKF
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>
        Eigen3::Eigen
)

add_library(EKF src/EKF/kalmanfilter_ekf.cpp include/kalmanfilter.h)
target_link_libraries(EKF
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>
        Eigen3::Eigen
)

add_library(UKF src/UKF/kalmanfilter_ukf.cpp include/kalmanfilter.h)
target_link_libraries(UKF
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>
        Eigen3::Eigen
)

add_executable(${PROJECT_NAME}_lkf
        ${SRC_CPP_DEFAULT}
)
target_link_libraries(${PROJECT_NAME}_lkf
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>
        Eigen3::Eigen
        LKF
)

add_executable(${PROJECT_NAME}_ekf
        ${SRC_CPP_DEFAULT}
)
target_link_libraries(${PROJECT_NAME}_ekf
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>
        Eigen3::Eigen
        EKF
)

add_executable(${PROJECT_NAME}_ukf
        ${SRC_CPP_DEFAULT}
)
target_link_libraries(${PROJECT_NAME}_ukf
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>
        Eigen3::Eigen
        UKF
)

# add_custom_command(TARGET ${PROJECT_NAME}_lkf PRE_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#     ${CMAKE_SOURCE_DIR}/data/ $<TARGET_FILE_DIR:${PROJECT_NAME}>)

# add_custom_command(TARGET ${PROJECT_NAME}_ekf PRE_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#     ${CMAKE_SOURCE_DIR}/data/ $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    
# add_custom_command(TARGET ${PROJECT_NAME}_ukf PRE_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#     ${CMAKE_SOURCE_DIR}/data/ $<TARGET_FILE_DIR:${PROJECT_NAME}>)