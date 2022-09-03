project (SPMM)

cuda_add_executable(spmm main.cu mm.cu)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(spmm PRIVATE -march=native -fopenmp)
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")


set(CUSPARSELT_PATH /home/slzhang/projects/spgemm/libcusparse_lt-linux-x86_64-0.3.0.3-archive)
message(STATUS ${CUSPARSELT_PATH})

target_link_libraries(spmm ${TARGET_LINK_LIBS} ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} ${CUSPARSELT_PATH}/lib/libcusparseLt.so ${CUPTI_LIB} gomp)
target_include_directories(spmm
    PUBLIC ${CUSPARSELT_PATH}/include
)