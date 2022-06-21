file(GLOB proto_files ${CMAKE_CURRENT_SOURCE_DIR}/*.proto)
foreach(proto_file ${proto_files})
    message(STATUS "compile proto file ${proto_file}")
    execute_process(
        COMMAND ${protoc_bin_path} --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
                --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/tensorflow
                --cpp_out=${CMAKE_SOURCE_DIR}/src 
                --python_out=${CMAKE_SOURCE_DIR}/zmq_ops
                ${proto_file}
    )
endforeach(proto_file)
