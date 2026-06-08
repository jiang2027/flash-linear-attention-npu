if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(causal_conv1d_bwd_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/causal_conv1d_bwd_tiling.cpp
        )
    AddOpTestCase(causal_conv1d_bwd "Ascend910B1" "" "${causal_conv1d_bwd_tiling_files}")
    if(ENABLE_ASAN)
        execute_process(
            COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
            OUTPUT_VARIABLE LIBASAN_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        get_filename_component(LIBASAN_DIR "${LIBASAN_PATH}" DIRECTORY)
        target_link_libraries(causal_conv1d_bwd_Ascend910B1_tiling_tmp PRIVATE -Wl,-rpath,${LIBASAN_DIR} -lasan)
    endif()
endif()