    
import tensorflow as tf


class ReadTFRecord:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.features_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
        }


    def _parse_function(self, example_proto):
        # Parse the input tf.train.Example proto using the feature description
        parsed_features = tf.io.parse_single_example(example_proto, self.features_description)
        
        # Decode the JPEG-encoded image and PNG-encoded mask
        image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
        mask = tf.io.decode_jpeg(parsed_features['mask'], channels=1)
        
        # Normalize the image
        image = tf.cast(image, tf.float32) / 255.0
        
        return image, mask
    

    def read_tfrecord(self, tfrecord_filename, batch_size):
        """Reads TFRecord file and returns a tf.data.Dataset."""
        raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
        parsed_dataset = raw_dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = parsed_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset