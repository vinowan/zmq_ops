cmake_minimum_required(VERSION 3.10)

project (zmq_dataset)

# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -DNDEBUG)

set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/zmq_ops)

if(NOT DEFINED ENV{CONDA_ENV_PATH})
    message(FATAL_ERROR "ENV CONDA_ENV_PATH not set")
else()
    message(STATUS "env CONDA_ENV_PATH: $ENV{CONDA_ENV_PATH}")
endif()

set(python_bin_path $ENV{CONDA_ENV_PATH}/bin/python)

execute_process(COMMAND ${python_bin_path} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE tf_inc
                )

if(NOT ${result} EQUAL 0)
    message(FATAL_ERROR "failed to get tensorflow include ${result}")
else()
    string(STRIP ${tf_inc} tf_inc)
    message(STATUS "tensorflow include path: ${tf_inc}")
endif()

execute_process(COMMAND ${python_bin_path} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE tf_lib
                )
if(NOT ${result} EQUAL 0)
    message(FATAL_ERROR "failed to get tensorflow library ${result}")
else()
    string(STRIP ${tf_lib} tf_lib)
    message(STATUS "tensorflow library path: ${tf_lib}")
endif()

execute_process(COMMAND ${python_bin_path} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE tf_flag
                )
if(NOT ${result} EQUAL 0)
    message(FATAL_ERROR "failed to get tensorflow compile flags ${result}")
else()
    string(STRIP ${tf_flag} tf_flag)
    message(STATUS "tensorflow compile flags: ${tf_flag}")
endif()

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${tf_flag} ${CMAKE_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed ${tf_lib} ${CMAKE_SHARED_LINKER_FLAGS}")

set(conda_inc $ENV{CONDA_ENV_PATH}/include)
message(STATUS "conda include path: ${conda_inc}")

set(conda_lib $ENV{CONDA_ENV_PATH}/lib)
message(STATUS "conda library path: ${conda_lib}")

set(protoc_bin_path $ENV{CONDA_ENV_PATH}/bin/protoc)

add_subdirectory(proto)
add_subdirectory(src)
