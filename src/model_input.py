import tensorflow as tf

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
        
        # Flatten image
        #image_op = tf.reshape(image_op, [-1])
    
        return tf.stack([rgb, dep], axis=-1)

    
def read_and_decode_sequence(filename_queue, config, mode):
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
        
        if mode is not 'inference':
            context_features = {
                    "label": tf.FixedLenFeature([], dtype=tf.int64),
                    "length": tf.FixedLenFeature([], dtype=tf.int64)
                }
        else:
            context_features = {
                    "id": tf.FixedLenFeature([], dtype=tf.int64),
                    "length": tf.FixedLenFeature([], dtype=tf.int64)
                }
        
        sequence_features={
                "depth": tf.FixedLenSequenceFeature([], dtype=tf.string),
                "rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                "segmentation": tf.FixedLenSequenceFeature([], dtype=tf.string),
                "skeleton": tf.FixedLenSequenceFeature([], dtype=tf.string),
            }
            
        context_encoded, sequence_encoded = tf.parse_single_sequence_example(
                serialized_example,
                context_features,
                sequence_features
            )
        
        # fetch required data fields
        # sequence label (train) or id (test)
        if mode is not 'inference':
            seq_label = context_encoded['label']
            seq_label = seq_label-1
        else:
            seq_label = context_encoded['id']
        
        # sequence length
        seq_len = tf.to_int32(context_encoded['length'])
        
        # channels
        seq_rgb = tf.decode_raw(sequence_encoded['rgb'], tf.uint8)
        seq_seg = tf.decode_raw(sequence_encoded['segmentation'], tf.uint8)
        seq_dep = tf.decode_raw(sequence_encoded['depth'], tf.uint8)
        
        # apply preprocessing to data
        seq_rgb = tf.map_fn(lambda x: preprocessing_op(x, config),
                                elems=(seq_rgb,seq_seg,seq_dep),
                                dtype=tf.float32,
                                back_prop=False)
        
        """
        # Use skeleton only.
        seq_skeleton = tf.decode_raw(sequence_encoded['skeleton'], tf.float32)
        # Normalize skeleton so that every pose is a unit length vector.
        seq_skeleton = tf.nn.l2_normalize(seq_skeleton, dim=1)
        seq_skeleton.set_shape([None, config['skeleton_size']])
        
        seq_len = tf.to_int32(context_encoded['length'])
        seq_label = context_encoded['label']
        # Tensorflow requires the labels start from 0. Before you create submission csv, 
        # increment the predictions by 1.
        seq_label = seq_label - 1
        """
        
        return [seq_rgb, seq_label, seq_len]


def input_pipeline(filenames, config, name='input_pipeline', mode='training'):
    with tf.name_scope(name):
        # Read the data from TFRecord files, decode and create a list of data samples by using threads.
            
        num_epochs = (config['num_epochs'] if mode is 'training' else 1)
        shuffle = (mode is not 'training')
            
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
        
        sample_list = [
                read_and_decode_sequence(filename_queue, config, mode) for _ in range(config['ip_num_read_threads'])
            ]
            
        batch_rgb, batch_labels, batch_lens = tf.train.batch_join(sample_list,
                batch_size=config['lr']['batch_size'],
                capacity=config['ip_queue_capacity'],
                enqueue_many=False,
                dynamic_pad=True,
                allow_smaller_final_batch=(mode is not 'training'),
                name="batch_join_and_pad"
            )
            
        return batch_rgb, batch_labels, batch_lens
