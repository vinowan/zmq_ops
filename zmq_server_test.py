#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file benchmark.py
# @brief
# @author vinowan
# @date 2021/4/9

import zmq
import sys
import tqdm
import argparse
import time
import numpy as np
import tensorflow as tf

from zmq_ops import zmq_ops 
from zmq_ops.tensor_array_pb2 import TensorArrayProto

PIPE = "tcp://127.0.0.1:5555"

TQDM_BAR_FMT = '{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
TQDM_BAR_LEN = 1000

def send():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.set_hwm(args.hwm)    
    socket.connect(PIPE)

    data = [
        np.random.rand(64, 224, 224, 3).astype('float32'),
        np.random.rand(64).astype('float32')
    ]   # 37MB per data
    
    tensor_array_proto = TensorArrayProto()
    tensor_proto_0 = tensor_array_proto.tensors.add()
    tensor_proto_0.CopyFrom(tf.make_tensor_proto(data[0]))
    tensor_proto_1 = tensor_array_proto.tensors.add()
    tensor_proto_1.CopyFrom(tf.make_tensor_proto(data[1]))
        
    while True:            
        with tqdm.trange(TQDM_BAR_LEN, ascii=True, bar_format=TQDM_BAR_FMT) as pbar:
            for k in range(TQDM_BAR_LEN):
                socket.send(tensor_array_proto.SerializeToString(), copy=False)
                socket.recv()
                pbar.update(1)        


def recv():
    graph = tf.compat.v1.Graph()    
    with graph.as_default():
        resource = zmq_ops.zmq_server_init(PIPE, args.hwm)
        client_id, tensors = zmq_ops.zmq_server_recv_all(resource, 
                                        types=[tf.float32, tf.float32], 
                                        shapes=[tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None])],
                                        min_cnt=1, max_cnt=4)
        send_op = zmq_ops.zmq_server_send_all(resource, client_id, tensors)
    
    with tf.compat.v1.Session(graph=graph) as sess:
        while True:
            with tqdm.trange(TQDM_BAR_LEN, ascii=True, bar_format=TQDM_BAR_FMT) as pbar:
                for k in range(TQDM_BAR_LEN):
                    sess.run(send_op)
                    pbar.update(1)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['send', 'recv'])
    parser.add_argument('--hwm', type=int, default=10)
    args = parser.parse_args()

    if args.task == 'send':
        send()
    elif args.task == 'recv':
        recv()
