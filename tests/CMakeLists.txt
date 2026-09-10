if(MSVC)
    set(gtest_force_shared_crt on)
endif()

set(BUILD_GMOCK OFF CACHE BOOL "Disable building Google Mock" FORCE)
FetchContent_Declare(googletest
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

##########################
##   Source Coverage    ##
##########################

function(add_test TEST_NAME)
    add_executable(${TEST_NAME}_test ${TEST_NAME}_test.cpp)
    target_include_directories(${TEST_NAME}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TEST_NAME}_test
    PRIVATE
      gtest_main
      Eigen3::Eigen
      walnutpie
  )
    if(WALNUTPIE_COVERAGE)
        target_compile_options(${TEST_NAME}_test PRIVATE
      -fprofile-instr-generate
      -fcoverage-mapping
    )
        target_link_options(${TEST_NAME}_test PRIVATE
      -fprofile-instr-generate
    )
    endif()
    gtest_discover_tests(${TEST_NAME}_test)
endfunction()

add_test(summary)
add_test(config)
add_test(util)
add_test(endpoint)
add_test(walnuts)
