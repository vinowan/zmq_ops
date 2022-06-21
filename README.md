# 1. 简介
zmq_ops是Avatar训练框架依赖的一个组件，通过把它集成到tensorflow中，可以使得tensorflow支持在线实时学习和训练。它的主要功能包括：

1. 符合tensorflow io接口标准，能够和tensorflow集成在一起
2. 提供单向数据传输的PUSH-PULL模式，也支持双向数据传输的REQ-ROUTER模式

# 2. 安装

## 2.1 安装依赖
```bash
conda install zeromq
conda install tensorflow
```
## 2.2 从源码安装
```bash
# 编译前要设置conda环境路径
export CONDA_ENV_PATH=/path/to/conda/env
cd zmq_ops
python setup.py install
```
## 2.3 二进制安装
```bash
pip install zmq-ops
```

# 3. 使用

## 3.1 ZmqReader
zmq reader主要提供ZMQ中的PUSH-PULL模式中的PULL端，它提供了3个OP：

1. zmq_reader_init(end_point, hwm)：初始化zmq reader
2. zmq_reader_next(resource, types, shapes)：读取下一组数据
3. zmq_reader_readable(resource)：判断zmq reader是否可读

## 3.2 ZmqServer
zmq server主要提供ZMQ中的REQ-ROUTER模式中的ROUTER端，它提供了3个OP

1. zmq_server_init(end_point, hwm)：初始化zmq server
2. zmq_server_recv_all(resource, types, shapes, min_cnt, max_cnt)：尽量从zmq server多读取数据，最少min_cnt条数据，最多max_cnt条数据，并把数据组成一个batch返回，返回client_id和tensors
3. zmq_server_send_all(resource, client_id, tensors)：把tensors按照client_id发送给不同的客户端

具体使用案例可以参考zmq_reader_test.py和zmq_server_test.py文件