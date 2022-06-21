set(
    zmq_srcs 
    zmq_reader_op.cc
    zmq_server_op.cc
    zmq_ops.cc
    tensor_array.pb.cc
)

link_directories(${conda_lib})

add_library(zmq_ops SHARED ${zmq_srcs})
target_include_directories(zmq_ops PUBLIC ${conda_inc} ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(zmq_ops zmq)