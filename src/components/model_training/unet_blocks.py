
import tensorflow as tf



def conv_block(inputs, num_filters):
    """
        Args:
            input :
            num_filters: int
        Return:
            act :
    """
    # implementing the first conv block.
    conv = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    batch_norm = tf.keras.layers.BatchNormalization()(conv)
    act = tf.keras.layers.Activation('relu')(batch_norm)

    # implementing the second conv block.
    conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(act)
    batch_norm = tf.keras.layers.BatchNormalization()(conv)
    act = tf.keras.layers.Activation("relu")(batch_norm)

    return act


def encoder_block(inputs, num_filters):
    """
        Args:
            inputs :
            num_filters: int
        Return:
            skip :
            max_pool :
    """
    
    skip = conv_block(inputs, num_filters) # convolutional block
    max_pool = tf.keras.layers.MaxPooling2D((2,2))(input) # pooling  block

    return skip, max_pool


def decoder_block(inputs, skip, num_filters):
    """
        Args:
            inputs: 
            skip: 
            num_filters: int
        Return :
            out :
    """
    # upsampling and concatenating the input features.
    upsample = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2,2), padding='same')(inputs)  # upsampling block
    connect_skip = tf.keras.layers.Concatenate([upsample, skip])
    out = conv_block(connect_skip, num_filters)

    return out
