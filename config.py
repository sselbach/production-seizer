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


if 'simple_conv' in sys.argv:
    key = 'simple_conv'
elif 'simple_no_conv' in sys.argv:
    key = 'simple_no_conv'
elif 'res_net' in sys.argv:
    key = 'res_net'
elif 'wide_conv' in sys.argv:
    key = 'wide_conv'
elif 'wide_no_conv' in sys.argv:
    key = 'wide_no_conv'

else:
    # default on simple conv
    key = 'simple_conv'
