import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1" # "-1" for cpu


import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, KLDivergence, mean_squared_error


from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.losses import util as tf_losses_util
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from tensorflow.python.ops import nn_ops



def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

  def _smooth_labels():
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return K.mean(
      K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)


def mean_squared_error(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)

def mean_absolute_error(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(math_ops.abs(y_pred - y_true), axis=-1)

# tf.nn.weighted_cross_entropy_with_logits()

def weighted_cross_entropy_with_logits_v2(labels, logits, pos_weight,
                                          name=None):
  """Computes a weighted cross entropy.
  """
  with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    try:
      labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))

    log_weight = 1 + (pos_weight - 1) * labels
    return math_ops.add(
        (1 - labels) * logits,
        log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                      nn_ops.relu(-logits)),
        name=name)
