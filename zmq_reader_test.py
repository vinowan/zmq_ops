#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file benchmark.py
# @brief
# @author vinowan
# @date 2020/9/3

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
    """ We use float32 data to pressure the system. In practice you'd use uint8 images."""
    data = [
        np.random.rand(64, 224, 224, 3).astype('float32'),
        np.random.rand(64).astype('float32')
    ]   # 37MB per data
    
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.set_hwm(args.hwm)    
    socket.connect(PIPE)
    
    tensor_array_proto = TensorArrayProto()
    tensor_proto_0 = tensor_array_proto.tensors.add()
    tensor_proto_0.CopyFrom(tf.make_tensor_proto(data[0]))
    tensor_proto_1 = tensor_array_proto.tensors.add()
    tensor_proto_1.CopyFrom(tf.make_tensor_proto(data[1]))
  
    try:        
        while True:            
            with tqdm.trange(TQDM_BAR_LEN, ascii=True, bar_format=TQDM_BAR_FMT) as pbar:
                for k in range(TQDM_BAR_LEN):
                    socket.send(tensor_array_proto.SerializeToString(), copy=False)
                    pbar.update(1)        
    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        if not ctx.closed:
            ctx.destroy(0)
        sys.exit()


def recv():
    graph = tf.compat.v1.Graph()    
    with graph.as_default():
        resource = zmq_ops.zmq_reader_init(PIPE, args.hwm)
        fetches = []
        for k in range(8):  # 8 GPUs pulling together in one sess.run call
            fetches.extend(zmq_ops.zmq_reader_next(resource,
                                    [tf.float32, tf.float32],
                                    [tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None])]
                                    ))
        fetch_op = tf.group(*fetches)


    with tf.compat.v1.Session(graph=graph) as sess:
        while True:
            with tqdm.trange(TQDM_BAR_LEN, ascii=True, bar_format=TQDM_BAR_FMT) as pbar:
                for k in range(TQDM_BAR_LEN // 8):
                    sess.run(fetch_op)
                    pbar.update(8)                       


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['send', 'recv'])
    parser.add_argument('--hwm', type=int, default=10)
    args = parser.parse_args()

    if args.task == 'send':
        send()
    elif args.task == 'recv':
        recv()
