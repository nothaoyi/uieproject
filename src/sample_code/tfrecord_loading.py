import tensorflow as tf
import numpy as np
import tempfile
import os
from scipy.misc import imread, imsave
from Skeleton import Skeleton

"""
An example of TFRecord data loading and preprocessing.
    - Creates input readers,
    - Loads a batch of samples,
    - Saves the tenth frame of the first sample onto disk.
"""

config = {}
config['model_dir'] = "./test_run/"
config['img_height'] = 80
config['img_width'] = 80
config['img_num_channels'] = 3
config['num_epochs'] = 10
config['batch_size'] = 16
# Capacity of the queue which contains the samples read by data readers.
# Make sure that it has enough capacity.
config['ip_queue_capacity'] = config['batch_size']*10
config['ip_num_read_threads'] = 4
# Directory of the data.
config['data_dir'] = "/home/eaksan/uie_data/train/"
# File naming
config['file_format'] = "dataTrain_%d.tfrecords"
# File IDs to be used for training.
config['file_ids'] = list(range(1,10))


def preprocessing_op(image_op, config):
    """
    Creates preprocessing operations that are going to be applied on a single frame.

    TODO: Customize for your needs.
    You can do any preprocessing (normalization/scaling of inputs, augmentation, etc.) by using tensorflow operations.
    Built-in image operations: https://www.tensorflow.org/api_docs/python/tf/image
    """
    with tf.name_scope("preprocessing"):
        # Reshape serialized image.
        image_op = tf.reshape(image_op, (config['img_height'],
                               config['img_width'],
                               config['img_num_channels'])
                          )
        # Integer to float.
        image_op = tf.to_float(image_op)

        # Normalize (zero-mean unit-variance) the image locally, i.e., by using statistics of the
        # image not the whole data or sequence.
        image_op = tf.image.per_image_standardization(image_op)

        return image_op

def read_and_decode_sequence(filename_queue, config):
    # Create a TFRecordReader.
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions)
    _, serialized_example = reader.read(filename_queue)

    # Read one sequence sample.
    # The training and validation files contains the following fields:
    # - label: label of the sequence which take values between 1 and 20.
    # - length: length of the sequence, i.e., number of frames.
    # - depth: sequence of depth images. [length x height x width x numChannels]
    # - rgb: sequence of rgb images. [length x height x width x numChannels]
    # - segmentation: sequence of segmentation maskes. [length x height x width x numChannels]
    # - skeleton: sequence of flattened skeleton joint positions. [length x numJoints]
    #
    # The test files doesn't contain "label" field.
    # [height, width, numChannels] = [80, 80, 3]
    with tf.name_scope("TFRecordDecoding"):
        context_encoded, sequence_encoded = tf.parse_single_sequence_example(
                serialized_example,
                # "label" and "lenght" are encoded as context features.
                context_features={
                    "label": tf.FixedLenFeature([], dtype=tf.int64),
                    "length": tf.FixedLenFeature([], dtype=tf.int64)
                },
                # "depth", "rgb", "segmentation", "skeleton" are encoded as sequence features.
                sequence_features={
                    "depth": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "segmentation": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "skeleton": tf.FixedLenSequenceFeature([], dtype=tf.string),
                })

        # Fetch required data fields.
        # TODO: Customize for your design. Assume that only the RGB images are used for now.
        # Decode the serialized RGB images.
        seq_rgb = tf.decode_raw(sequence_encoded['rgb'], tf.uint8)
        seq_label = context_encoded['label']
        seq_len = context_encoded['length']
        # Output dimnesionality: [seq_len, height, width, numChannels]
        # tf.map_fn applies the preprocessing function on every image in the sequence, i.e., frame.
        seq_rgb = tf.map_fn(lambda x: preprocessing_op(x, config),
                                elems=seq_rgb,
                                dtype=tf.float32,
                                back_prop=False)
        seq_skeleton = tf.decode_raw(sequence_encoded['skeleton'], tf.float32)

        return [seq_rgb, seq_skeleton, seq_label, seq_len]


def input_pipeline(filenames, config):
    with tf.name_scope("input_pipeline"):
        # Create a queue of TFRecord input files.
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=config['num_epochs'], shuffle=True)
        # Read the data from TFRecord files, decode and create a list of data samples by using threads.
        sample_list = [read_and_decode_sequence(filename_queue, config) for _ in range(config['ip_num_read_threads'])]
        # Create batches.
        # Since the data consists of variable-length sequences, allow padding by setting dynamic_pad parameter.
        # "batch_join" creates batches of samples and pads the sequences w.r.t the max-length sequence in the batch.
        # Hence, the padded sequence length can be different for different batches.
        batch_rgb, batch_skeleton, batch_labels, batch_lens = tf.train.batch_join(sample_list,
                                                    batch_size=config['batch_size'],
                                                    capacity=config['ip_queue_capacity'],
                                                    enqueue_many=False,
                                                    dynamic_pad=True,
                                                    name="batch_join_and_pad")

        return batch_rgb, batch_skeleton, batch_labels, batch_lens


def main(unused_argv):
    # Create a list of TFRecord input files.
    filenames = [os.path.join(config['data_dir'], config['file_format'] % i) for i in config['file_ids']]
    # Create data loading operators. This will be represented as a node in the computational graph.
    batch_rgb_op, batch_skeleton_op, batch_labels_op, batch_seq_len_op = input_pipeline(filenames, config)
    # TODO: your model can take batch_rgb, batch_labels and batch_seq_len ops as an input.

    # Create tensorflow session and initialize the variables (if any).
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    # Create threads to prefetch the data.
    # https://www.tensorflow.org/programmers_guide/reading_data#creating_threads_to_prefetch_using_queuerunner_objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    """
    # Training Loop
    # The input pipeline creates input batches for config['num_epochs'] epochs,
    # You can iterate over the training data by using coord.should_stop() signal.
    try:
        while not coord.should_stop():
            # TODO: Model training

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    # Close session
    sess.close()
    """

    # Instead of running training, fetch a sample and save one frame.
    # Fetch a batch of samples.
    batch_rgb, batch_skeleton, batch_labels, batch_seq_len = sess.run([batch_rgb_op, batch_skeleton_op, batch_labels_op, batch_seq_len_op])

    # Print
    print("# Samples: " + str(len(batch_rgb)))
    print("Sequence lengths: " + str(batch_seq_len))
    print("Sequence labels: " + str(batch_labels))

    # Note that the second dimension will give maximum-length in the batch, i.e., the padded sequence length.
    print("Sequence type: " + str(type(batch_rgb)))
    print("Sequence shape: " + str(batch_rgb.shape))

    # Fetch first clips 10th frame.
    img = batch_rgb[0][9]
    print("Image shape: " + str(img.shape))

    # Create a skeleton object.
    skeleton = Skeleton(batch_skeleton[0][10])
    # Resize the pixel coordinates.
    skeleton.resizePixelCoordinates()
    # Draw skeleton image.
    skeleton_img = skeleton.toImage(img.shape[0], img.shape[1])

    imsave(config['model_dir']+'rgb_test_img.png', img)
    imsave(config['model_dir']+'skeleton_test_img.png', skeleton_img)

if __name__ == '__main__':
    tf.app.run()
