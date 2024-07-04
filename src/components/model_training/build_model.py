import sys

import tensorflow as tf

sys.path.append('./')
from unet_blocks import conv_block, encoder_block, decoder_block



def build_unet(n_classes: int, height: int, width: int, channel: int):
    """
    Function that define the UNET Model.
    
        Args :
            n_classes: int
            height: int
            width: int
            channel: int
        Return:
            model:
    """

    inputs = tf.keras.layers.Input((height, width, channel))

    # Encoder (down sampler)
    skip1, max_pool1 = encoder_block(inputs, num_filters=16)
    skip2, max_pool2 = encoder_block(max_pool1, num_filters=32)
    skip3, max_pool3 = encoder_block(max_pool2, num_filters=64)
    skip4, max_pool4 = encoder_block(max_pool3, num_filters=128)
    skip5, max_pool5 = encoder_block(max_pool4, num_filters=256)
    skip6, max_pool6 = encoder_block(max_pool5, num_filters=512)

    # Bottleneck
    bridge = conv_block(max_pool6, num_filters=1024)

    # Decoder (up sampler)
    u6 = decoder_block(bridge, skip6, num_filters=512)
    u5 = decoder_block(u6, skip5, num_filters=256)
    u4 = decoder_block(u5, skip4, num_filters=128)
    u3 = decoder_block(u4, skip3, num_filters=64)
    u2 = decoder_block(u3, skip2, num_filters=32)
    u1 = decoder_block(u2, skip1, num_filters=16)


    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='softmax')(u1)
    model = tf.keras.models.Model(inputs[inputs], outputs=[outputs], name="UNET")

    return model
