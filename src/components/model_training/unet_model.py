
import tensorflow as tf



def conv_block(inputs, num_filters):
    """
    A function defines a convolutional block.
        Args:
            input: The input tensor to the convolutional block.
            num_filters (int): The number of filters for the convolutional layers.
            
        Return:
            act: The output tensor ater applying the convolutional block.
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
    A function defines encoder block for the U-Net architecture.

        Args:
            inputs: The input tensor.
            num_filters: The number of filters for the convolutional layers.

        Returns:
            skip: The output tensor from the convolutional block.
            max_pool: The output tensor after max pooling.
    """
    
    skip = conv_block(inputs, num_filters) # convolutional block
    max_pool = tf.keras.layers.MaxPooling2D((2,2))(inputs) # pooling  block

    return skip, max_pool


def decoder_block(inputs, skip, num_filters):
    """
    A function defines a decoder block for the U-Net architecture.

        Args:
            inputs: The input tensor.
            skip:  The skip connection tensor from the corresponding encoder block.
            num_filters: The number of filters for the convolutional layers.

        Return:
            out: The output tensor after applying the decoder block.
    """
    # upsampling and concatenating the input features.
    upsample = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2,2), padding='same')(inputs)  # upsampling block
    connect_skip = tf.keras.layers.concatenate([upsample, skip])
    out = conv_block(connect_skip, num_filters)

    return out



def build_unet(n_classes: int, height: int, width: int, channel: int):
    """
    A function that defines and build the U-Net model for image segmentation.
    
        Args :
            n_classes (int): The number of output classes.
            height (int): The height of the input images.
            width (int): The width of the input images.
            channel (int): The number of channels in the input images.

        Return:
            model: The compiled U-Net model.
    """

    inputs = tf.keras.layers.Input((height, width, channel))

    # Defining the encoder (down sampler)
    skip1, max_pool1 = encoder_block(inputs, num_filters=16)
    skip2, max_pool2 = encoder_block(max_pool1, num_filters=32)
    skip3, max_pool3 = encoder_block(max_pool2, num_filters=64)
    skip4, max_pool4 = encoder_block(max_pool3, num_filters=128)
    skip5, max_pool5 = encoder_block(max_pool4, num_filters=256)
    skip6, max_pool6 = encoder_block(max_pool5, num_filters=512)

    # Defining the bottleneck
    bridge = conv_block(max_pool6, num_filters=1024)

    # Defining the decoder (up sampler)
    u6 = decoder_block(bridge, skip6, num_filters=512)
    u5 = decoder_block(u6, skip5, num_filters=256)
    u4 = decoder_block(u5, skip4, num_filters=128)
    u3 = decoder_block(u4, skip3, num_filters=64)
    u2 = decoder_block(u3, skip2, num_filters=32)
    u1 = decoder_block(u2, skip1, num_filters=16)

    # output function
    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(u1)
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs], name="Unet")

    return model
