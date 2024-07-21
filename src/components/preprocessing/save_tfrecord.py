
import tensorflow as tf


class SaveTFRecord:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):  # if value is tensor
            value = value.numpy()  # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def serialize_example(self, image, mask):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        feature = {
            'image': self._bytes_feature(tf.io.encode_jpeg(tf.cast(image, tf.uint8))),
            'mask': self._bytes_feature(tf.io.encode_jpeg(tf.cast(mask, tf.uint8)))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    def save_tfrecord(self, dataset, filename):
        """Save dataset in TFRecord format."""
        with tf.io.TFRecordWriter(filename) as writer:
            for images, masks in dataset:
                for image, mask in zip(images, masks):
                    tf_example = self.serialize_example(image, mask)
                    writer.write(tf_example)