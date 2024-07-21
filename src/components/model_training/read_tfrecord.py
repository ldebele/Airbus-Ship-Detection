    
import tensorflow as tf


class ReadTFRecord:
    features_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }

    @staticmethod
    def _parse_function(example_proto):
        # Parse the input tf.train.Example proto using the feature description
        parsed_features = tf.io.parse_single_example(example_proto, ReadTFRecord.features_description)
        
        # Decode the JPEG-encoded image and PNG-encoded mask
        image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
        mask = tf.io.decode_jpeg(parsed_features['mask'], channels=1)
        
        # Normalize the image and mask
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32)
        
        return image, mask
    
    @staticmethod
    def load_tfrecord(filename, batch):
        """Reads TFRecord file and returns a tf.data.Dataset."""
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(ReadTFRecord._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch).prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset