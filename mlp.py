# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf

def decode_transfer_fn(transfer_fn):
    if transfer_fn == "relu": return tf.nn.relu
    elif transfer_fn == "tanh": return tf.nn.tanh
    elif transfer_fn == "sig": return tf.nn.sigmoid
    elif transfer_fn == "id":  return (lambda x: x)
    else:
        raise Exception("Unsupported transfer function %s" % transfer_fn)

def init_ws_bs(opts, name, d_in, d_outs):
    ws = []
    bs = []
    d = d_in

    with tf.variable_scope(name) as scope:
        for i, d_out in enumerate(d_outs):
            with tf.variable_scope('%d' % i) as scope:
                ws.append(tf.get_variable(name="w", shape=[d, d_out], initializer=tf.initializers.random_uniform(minval=-opts.init_range,
                                                                                                                 maxval=opts.init_range)))
                bs.append(tf.get_variable(name="b", shape=[d_out], initializer=tf.zeros_initializer()))
            d = d_out

    return (ws, bs)

class MLP(object):
    def __init__(self, opts, d_in, d_outs, name):

        (self.ws, self.bs) = init_ws_bs(opts, name, d_in, d_outs)

        self.opts = opts
        self.name = name
        self.transfer_fn = decode_transfer_fn(opts.mlp_transfer_fn)
        self.output_size = d_outs[-1]

    def stop_gradients(self):
        for i in range(len(self.ws)):
            self.ws[i] = tf.stop_gradient(self.ws[i])
        for i in range(len(self.bs)):
            self.bs[i] = tf.stop_gradient(self.bs[i])

    def forward(self, z):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('fwd') as scope:
                x = z
                for i in range(len(self.ws)):
                    with tf.variable_scope('%d' % i) as scope:
                        old_x = x
                        x = tf.matmul(x, self.ws[i]) + self.bs[i]
                        if i + 1 < len(self.ws):
                            x = self.transfer_fn(x)
                        if self.opts.res_net and tf.shape(x) == tf.shape(old_x):
                            x += old_x
        return x
