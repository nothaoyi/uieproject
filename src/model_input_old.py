import tensorflow as tf
import numpy as np

 
def preprocessing_op(frame, config):
    """
    Creates preprocessing operations that are going to be applied on a single frame.
    
    TODO: Customize for your needs.
    You can do any preprocessing (masking, normalization/scaling of inputs, augmentation, etc.) by using tensorflow operations.
    Built-in image operations: https://www.tensorflow.org/api_docs/python/tf/image 
    """
    with tf.name_scope("preprocessing"):
        rgb = frame[0]
        seg = frame[1]
        dep = frame[2]
        skl = frame[3]
        
        # Reshape serialized image.
        rgb = tf.reshape(rgb, (config['img_height'], 
                               config['img_width'], 
                               config['img_num_channels'])
                        )
        
        seg = tf.reshape(seg, (config['img_height'], 
                               config['img_width'], 
                               config['img_num_channels'])
                        )
        
        dep = tf.reshape(dep, (config['img_height'], 
                               config['img_width'], 
                               1)
                        )
        
        # Convert from RGB to grayscale.
        rgb = tf.image.rgb_to_grayscale(rgb)
        seg = tf.image.rgb_to_grayscale(seg)
        
        # Integer to float.
        rgb = tf.to_float(rgb)
        dep = tf.to_float(dep)
        
        seg = tf.greater(seg, 150)
        
        rgb = tf.where(seg, rgb, tf.zeros_like(rgb))
        dep = tf.where(seg, dep, tf.zeros_like(rgb))
        
        # Crop
        #image_op = tf.image.resize_image_with_crop_or_pad(image_op, 60, 60)
        
        # Resize operation requires 4D tensors (i.e., batch of images).
        # Reshape the image so that it looks like a batch of one sample: [1,60,60,1]
        #image_op = tf.expand_dims(image_op, 0)
        # Resize
        #image_op = tf.image.resize_bilinear(image_op, np.asarray([32,32]))
        # Reshape the image: [32,32,1]
        #image_op = tf.squeeze(image_op, 0)
        
        # Normalize (zero-mean unit-variance) the image locally, i.e., by using statistics of the 
        # image not the whole data or sequence.
        
        rgb = tf.image.per_image_standardization(rgb)
        dep = tf.image.per_image_standardization(dep)
        
        rgb = tf.squeeze(rgb)
        dep = tf.squeeze(dep)
        
        bod = np.zeros((80,80))
        
        # create 80x80 grid
        x = np.linspace(0,79,80)
        y = np.linspace(0,79,80)
        X, Y = np.meshgrid(x,y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        
        # initialize bod
        bod = tf.zeros((80,80), dtype=tf.float64)
        
        # hip center
        hcx = tf.subtract(skl[7],140)
        hcy = tf.subtract(skl[8], 10)
        
        hcx = tf.cast(tf.div(hcx, 4), dtype=tf.float64)
        hcy = tf.cast(tf.div(hcy, 4), dtype=tf.float64)

        ds = tf.contrib.distributions
        
        mvn = ds.MultivariateNormalDiag(tf.stack((hcx, hcy)), np.array([20., 20.], dtype=np.float64))
        
        bod = tf.subtract(bod, mvn.pdf(pos))
        
        # shoulder center
        scx = tf.subtract(skl[25],140)
        scy = tf.subtract(skl[26], 10)
        
        scx = tf.cast(tf.div(scx, 4), dtype=tf.float64)
        scy = tf.cast(tf.div(scy, 4), dtype=tf.float64)

        ds = tf.contrib.distributions
        
        mvn = ds.MultivariateNormalDiag(tf.stack((scx, scy)), np.array([20., 20.], dtype=np.float64))
        
        bod = tf.subtract(bod, mvn.pdf(pos))
        
        # left hand
        lhx = tf.subtract(skl[70],140)
        lhy = tf.subtract(skl[71], 10)
        
        lhx = tf.cast(tf.div(lhx, 4), dtype=tf.float64)
        lhy = tf.cast(tf.div(lhy, 4), dtype=tf.float64)

        ds = tf.contrib.distributions
        
        mvn = ds.MultivariateNormalDiag(tf.stack((lhx, lhy)), np.array([10., 10.], dtype=np.float64))
        
        bod = tf.add(bod, mvn.pdf(pos))
        
        # right hand
        rhx = tf.subtract(skl[106],140)
        rhy = tf.subtract(skl[107], 10)
        
        rhx = tf.cast(tf.div(rhx, 4), dtype=tf.float64)
        rhy = tf.cast(tf.div(rhy, 4), dtype=tf.float64)

        ds = tf.contrib.distributions
        
        mvn = ds.MultivariateNormalDiag(tf.stack((rhx, rhy)), np.array([10., 10.], dtype=np.float64))
        
        bod = tf.add(bod, mvn.pdf(pos))
        
        # Flatten image
        #image_op = tf.reshape(image_op, [-1])
    
        return tf.stack([rgb, dep, tf.to_float(bod)], axis=-1)

    
def read_and_decode_sequence(filename_queue, config):
    # Create a TFRecordReader.
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions)
    _, serialized_example = reader.read(filename_queue)

    # read one sequence sample
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

        # fetch required data fields
        # sequence label
        seq_lbl = context_encoded['label']
        seq_lbl = seq_lbl-1
        
        # sequence length
        seq_len = tf.to_int32(context_encoded['length'])
        
        # skeleton
        seq_skl = tf.decode_raw(sequence_encoded['skeleton'], tf.float32)
        
        # channels
        seq_rgb = tf.decode_raw(sequence_encoded['rgb'], tf.uint8)
        seq_seg = tf.decode_raw(sequence_encoded['segmentation'], tf.uint8)
        seq_dep = tf.decode_raw(sequence_encoded['depth'], tf.uint8)
        
        # apply preprocessing to data
        seq_rgb = tf.map_fn(lambda x: preprocessing_op(x, config),
                                elems=(seq_rgb,seq_seg,seq_dep,seq_skl),
                                dtype=tf.float32,
                                back_prop=False)
        
        return [seq_rgb, seq_lbl, seq_len]

def read_and_decode_sequence_test_data(filename_queue, config):
    """
    Replace label field with id field because test data doesn't contain labels.
    """
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
                    "id": tf.FixedLenFeature([], dtype=tf.int64),
                    "length": tf.FixedLenFeature([], dtype=tf.int64)
                },
                # "depth", "rgb", "segmentation", "skeleton" are encoded as sequence features.
                sequence_features={
                    "depth": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "segmentation": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "skeleton": tf.FixedLenSequenceFeature([], dtype=tf.string),
                })


        # fetch required data fields
        # sequence id
        seq_id = context_encoded['id']
        
        # sequence length
        seq_len = tf.to_int32(context_encoded['length'])
        
        # skeleton
        seq_skl = tf.decode_raw(sequence_encoded['skeleton'], tf.float32)
        
        # channels
        seq_rgb = tf.decode_raw(sequence_encoded['rgb'], tf.uint8)
        seq_seg = tf.decode_raw(sequence_encoded['segmentation'], tf.uint8)
        seq_dep = tf.decode_raw(sequence_encoded['depth'], tf.uint8)
        
        # apply preprocessing to data
        seq_rgb = tf.map_fn(lambda x: preprocessing_op(x, config),
                                elems=(seq_rgb,seq_seg,seq_dep,seq_skl),
                                dtype=tf.float32,
                                back_prop=False)
        
        return [seq_rgb, seq_id, seq_len]


def input_pipeline(filenames, config, name='input_pipeline', shuffle=True, mode='training'):
    with tf.name_scope(name):
        # Read the data from TFRecord files, decode and create a list of data samples by using threads.
        if mode is "training":
            # Create a queue of TFRecord input files.
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=config['num_epochs'], shuffle=shuffle)
            sample_list = [read_and_decode_sequence(filename_queue, config) for _ in range(config['ip_num_read_threads'])]
            batch_rgb, batch_labels, batch_lens = tf.train.batch_join(sample_list,
                                                    batch_size=config['batch_size'],
                                                    capacity=config['ip_queue_capacity'],
                                                    enqueue_many=False,
                                                    dynamic_pad=True,
                                                    allow_smaller_final_batch = False,
                                                    name="batch_join_and_pad")
            return batch_rgb, batch_labels, batch_lens

        else:
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
            sample_list = [read_and_decode_sequence_test_data(filename_queue, config) for _ in range(config['ip_num_read_threads'])]
            batch_rgb, batch_ids, batch_lens = tf.train.batch_join(sample_list,
                                                    batch_size=config['batch_size'],
                                                    capacity=config['ip_queue_capacity'],
                                                    enqueue_many=False,
                                                    dynamic_pad=True,
                                                    allow_smaller_final_batch = True,
                                                    name="batch_join_and_pad")
            return batch_rgb, batch_ids, batch_lens
