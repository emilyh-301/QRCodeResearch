import tensorflow as tf

# tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None )

print(tf.config.list_physical_devices('GPU'))
