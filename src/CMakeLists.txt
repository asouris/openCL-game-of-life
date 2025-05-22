

file(COPY CalcStep.cl DESTINATION ${PROJECT_SOURCE_DIR}/bin/)
file(COPY CalcStepGroups.cl DESTINATION ${PROJECT_SOURCE_DIR}/bin/)


add_executable(openclConway parallel-conway.cpp)
target_include_directories(openclConway PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(openclConway OpenCL)