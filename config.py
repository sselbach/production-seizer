import tensorflow as tf
import sys

"""
Read out arguments.
"""

"""
GPU options.
"""
if 'gpu' in sys.argv:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    #tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

"""
Choice of Net Architecture.
"""
# default on simple conv
key = 'simple_conv'

if 'simple_conv' in sys.argv:
    key = 'simple_conv'
if 'simple_no_conv' in sys.argv:
    key = 'simple_no_conv'
if 'res_net' in sys.argv:
    key = 'res_net'
