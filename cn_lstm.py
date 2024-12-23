import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from preprocess import data_to_supervised_3
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
import time

class LayerNormLSTMCell(rnn_cell_impl.RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.
  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               forget_bias=1.0,
               activation=None,
               layer_norm=False,
               norm_gain=1.0,
               norm_shift=0.0,
               reuse=None):
    """Initialize the parameters for an LSTM cell.
    """
    super(LayerNormLSTMCell, self).__init__(_reuse=reuse)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._forget_bias = forget_bias
    self._activation = activation or math_ops.tanh
    self._layer_norm = layer_norm
    self._norm_gain = norm_gain
    self._norm_shift = norm_shift

    if num_proj:
      self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_proj))
      self._output_size = num_proj
    else:
      self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_units))
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def _linear(self,
              args,
              output_size,
              bias,
              bias_initializer=None,
              kernel_initializer=None,
              layer_norm=False):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a Variable.
    """
    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
      weights = vs.get_variable(
          "kernel", [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if len(args) == 1:
        res = math_ops.matmul(args[0], weights)
      else:
        res = math_ops.matmul(array_ops.concat(args, 1), weights)
      if not bias:
        return res
      with vs.variable_scope(outer_scope) as inner_scope:
        inner_scope.set_partitioner(None)
        if bias_initializer is None:
          bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
        biases = vs.get_variable(
            "bias", [output_size], dtype=dtype, initializer=bias_initializer)

    if not layer_norm:
      res = nn_ops.bias_add(res, biases)

    return res

  def call(self, inputs, state):
    """Run one step of LSTM.
    """
    sigmoid = math_ops.sigmoid

    (c_prev, m_prev) = state

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      lstm_matrix = self._linear(
          [inputs, m_prev],
          4 * self._num_units,
          bias=True,
          bias_initializer=None,
          layer_norm=self._layer_norm)
      i, j, f, o = array_ops.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)

      if self._layer_norm:
        i = _norm(self._norm_gain, self._norm_shift, i, "input")
        j = _norm(self._norm_gain, self._norm_shift, j, "transform")
        f = _norm(self._norm_gain, self._norm_shift, f, "forget")
        o = _norm(self._norm_gain, self._norm_shift, o, "output")

      # Diagonal connections
      if self._use_peepholes:
        with vs.variable_scope(unit_scope):
          w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (
            sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
            sigmoid(i + w_i_diag * c_prev) * self._activation(j))
      else:
        c = (
            sigmoid(f + self._forget_bias) * c_prev +
            sigmoid(i) * self._activation(j))

      if self._layer_norm:
        c = _norm(self._norm_gain, self._norm_shift, c, "state")

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * self._activation(c)
      else:
        m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        with vs.variable_scope("projection"):
          m = self._linear(m, self._num_proj, bias=False)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

    new_state = (rnn_cell_impl.LSTMStateTuple(c, m))
    return m, new_state

start = time.perf_counter()
path = 'dataset/newaichi15min.csv'
df = pd.read_csv(path, header=0, index_col=None)

time_df = df.pop('時間').tolist()
df = df.drop(['風向','温度'],axis=1)
value_df = df.values.astype('float32')
shifted_n = 9
shifted_value = data_to_supervised_3(value_df,shifted_n)
#print(shifted_value.head())

scaler = MinMaxScaler(feature_range=(0,1))
reframed = scaler.fit_transform(shifted_value)

train_start = '2/18/2006 15:00:00'
train_end = '3/20/2006 15:00:00'
test_start = '3/20/2006 15:00:00'
test_end = '3/21/2006 15:00:00'

train_start_n= time_df.index(train_start ) - shifted_n
train_end_n  = time_df.index(train_end )- shifted_n
test_start_n  = time_df.index(test_start )- shifted_n
test_end_n  = time_df.index(test_end )- shifted_n

train_x = reframed[train_start_n : train_end_n, :-1]
test_x = reframed[test_start_n : test_end_n, :-1]
# split into input and outputs
train_y = reframed[train_start_n : train_end_n,-1]
test_y = reframed[test_start_n : test_end_n,-1]
train_y = train_y.reshape(len(train_y),1)
test_y = test_y.reshape(len(test_y),1)

input_x = tf.placeholder(tf.float32,[None,2*(shifted_n+1)])
output_y = tf.placeholder(tf.float32,[None,1])

input_x_images = tf.reshape(input_x,[-1,2,shifted_n+1,1])
conv1 = tf.layers.conv2d(inputs=input_x_images, filters=32, kernel_size = [2,2],\
                         strides = 1, padding='same',activation=tf.nn.relu)

flat = tf.reshape(conv1,[-1,1,2*(shifted_n+1)*32])
cell = tf.contrib.rnn.BasicLSTMCell(60)
outputs,states = tf.nn.dynamic_rnn(cell,flat,dtype=tf.float32)
dense = tf.layers.dense(inputs=outputs,units=60,activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense,rate=0.5)
logits=tf.layers.dense(inputs=dropout,units=1)
logits = tf.reshape(logits,shape = [-1,logits.shape[2]])

lr = 0.2
loss = tf.reduce_mean(tf.square(logits - train_y))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={input_x: train_x, output_y: train_y})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={input_x: train_x, output_y: train_y}))
    y_pre = sess.run(logits,feed_dict={input_x: test_x})
    print("CNLSTM-time:{}".format(time.perf_counter()-start))

    x_y_pre = np.concatenate((test_x,y_pre),axis=1)
    x_y_pre_inv = scaler.inverse_transform(x_y_pre)
    y_pre_inv = x_y_pre_inv[:,-1]

    x_y_test = np.concatenate((test_x, test_y), axis=1)
    x_y_test_inv = scaler.inverse_transform(x_y_test)
    y_test_inv = x_y_test_inv[:, -1]

    rmse = np.math.sqrt(mean_squared_error(y_test_inv,y_pre_inv))
    nrmse = rmse / 24.75 * 100
    print('rmse:{},nrmse{}'.format(rmse,nrmse))

    plt.plot(y_test_inv,label = 'actual')
    plt.plot(y_pre_inv,label = 'CN-LSTM')
    plt.legend(loc='upper right')
    plt.xlabel('Time Point(15min)')
    plt.ylabel('Wind Power(MW)')
    y_stick = np.arange(0, 25, 6)
    x_stick = np.arange(0, 97, 16)
    plt.xticks(x_stick[:])
    plt.yticks(y_stick[1:])
    plt.show()