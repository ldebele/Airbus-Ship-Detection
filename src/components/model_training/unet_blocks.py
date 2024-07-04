
import tensorflow as tf



def conv_block(x, num_filters):
    """
        Args:
            x :
            num_filters : int
            dropout :
            l2 :

        Return:
            x :
    """

    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x


def encoder_block(x, num_filters):
    """
        Args:
            x :
            num_filters :
            dropout :
            l2 :

        Return:
            x :
            p :
    """

    skip = conv_block(x, num_filters) # convolutional block
    p = tf.keras.layers.MaxPooling2D((2,2))(x) # pooling  block

    return skip, p


def decoder_block(x, skip, num_filters):
    """
        Args:
            x :
            p :
            num_filters :
            dropout :
            l2 :

        Return :
            x :
    """

    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2,2), padding='same')(x)  # upsampling block
    x = tf.keras.layers.Concatenate([x, skip])
    x = conv_block(x, num_filters)

    return x
