import tensorflow as tf



def plot_model(model, filename):
    tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True)
