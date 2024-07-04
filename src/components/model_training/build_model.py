import sys

import tensorflow as tf

sys.path.append('./')
from unet_blocks import conv_block, encoder_block, decoder_block



def build_unet(n_classes: int, height: int, width: int, channel: int):
    """
        Args :
            n_classes (int)
            height (int)
            width (int)
            channel (int)

        Return:
            model :
    """

    inputs = tf.keras.layers.Input((height, width, channel))

    # Encoder (down sampler)
    skip1, p1 = encoder_block(inputs, num_filters=16)
    skip2, p2 = encoder_block(p1, num_filters=32)
    skip3, p3 = encoder_block(p2, num_filters=64)
    skip4, p4 = encoder_block(p3, num_filters=128)
    skip5, p5 = encoder_block(p4, num_filters=256)
    skip6, p6 = encoder_block(p5, num_filters=512)

    # Bottleneck
    bridge = conv_block(p6, num_filters=1024)

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
