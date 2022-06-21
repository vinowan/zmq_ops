#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from zmq_ops import zmq_ops

class ZmqStreamDataset(tf.data.Dataset):
    def __init__(self, end_point, hwm, types, shapes, num_parallel_calls=8):
        assert len(types) == len(shapes), f"mismatch length: types({types}) shapes({shapes})"

        with tf.name_scope("ZmqStreamDataset"):
            resource = zmq_ops.zmq_reader_init(end_point, hwm)
            self._resource = resource

            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda _: zmq_ops.zmq_reader_next(self._resource, types, shapes),
                num_parallel_calls = num_parallel_calls
            )
            dataset = dataset.unbatch()

            self._dataset = dataset            
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    def _as_variant_tensor(self):
        return self._dataset._variant_tensor

    @property
    def element_spec(self):
        return self._dataset.element_spec
        
    @property
    def readable(self):
        return zmq_ops.zmq_readable(self._resource)
