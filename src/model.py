import tensorflow as tf

class CNNModel():
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables sharing the parameters so that both graphs share the parameters.
    """
    def __init__(self, config, input_op, mode):
        """
        Basic setup.
        Args:
          config: Object containing configuration parameters.
        """
        assert mode in ["training", "validation", "inference"]
        self.config = config
        self.inputs = input_op
        self.mode = mode
        self.is_training = self.mode == "training"
        self.reuse = self.mode == "validation"


    def build_model(self, input_layer):
        with tf.variable_scope("cnn_model", reuse=self.reuse, initializer=tf.contrib.layers.xavier_initializer()):
            # Convolutional Layer #1
            # Computes 32 features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            # Input Tensor Shape: [batch_size, 80, 80, num_channels]
            # Output Tensor Shape: [batch_size, 40, 40, num_filter1]
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=self.config['cnn_filters'][0],
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')

            # Input Tensor Shape: [batch_size, 40, 40, num_filter1]
            # Output Tensor Shape: [batch_size, 20, 20, num_filter2]
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=self.config['cnn_filters'][1],
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same')

            # Input Tensor Shape: [batch_size, 20, 20, num_filter2]
            # Output Tensor Shape: [batch_size, 10, 10, num_filter3]
            conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=self.config['cnn_filters'][2],
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, padding='same')

            # Input Tensor Shape: [batch_size, 10, 10, num_filter3]
            # Output Tensor Shape: [batch_size, 5, 5, num_filter4]
            conv4 = tf.layers.conv2d(
                inputs=pool3,
                filters=self.config['cnn_filters'][3],
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='same')


            # Flatten tensor into a batch of vectors
            # Input Tensor Shape: [batch_size, 5, 5, num_filter4]
            # Output Tensor Shape: [batch_size, 5 * 5 * num_filter4]
            conv_flat = tf.reshape(pool4, [-1, 5 * 5 * self.config['cnn_filters'][3]])

            # Add dropout operation;
            dropout = tf.layers.dropout(inputs=conv_flat, rate=self.config['dropout_rate'], training=self.is_training)

            # Dense Layer
            # Densely connected layer with <num_hidden_units> neurons
            # Input Tensor Shape: [batch_size, 5 * 5 * num_filter4]
            # Output Tensor Shape: [batch_size, num_hidden_units]
            dense = tf.layers.dense(inputs=dropout, units=self.config['num_hidden_units'],
                activation=tf.nn.relu)

            self.cnn_model = dense
            return dense

    def build_graph(self):
        """
        CNNs accept inputs of shape (batch_size, height, width, num_channels). However, we have inputs of shape
        (batch_size, sequence_length, height, width, num_channels) where sequence_length is inferred at run time.
        We need to iterate in order to get CNN representations. Similar to python's map function, "tf.map_fn"
        applies a given function on each entry in the input list.
        """
        # For the first time create a dummy graph and then share the parameters everytime.
        if self.is_training:
            self.reuse = False
            self.build_model(self.inputs[0])
            self.reuse = True

        # CNN takes a clip as if it is a batch of samples.
        # Have a look at tf.map_fn (https://www.tensorflow.org/api_docs/python/tf/map_fn)
        # You can set parallel_iterations or swap_memory in order to make it faster.
        # Note that back_prop argument is True in order to enable training of CNN.
        self.cnn_representations = tf.map_fn(lambda x: self.build_model(x),
                                                elems=self.inputs,
                                                dtype=tf.float32,
                                                back_prop=True,
                                                swap_memory=True,
                                                parallel_iterations=2)

        return self.cnn_representations


class RNNModel():
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables sharing the parameters so that both graphs share the parameters.
    """
    def __init__(self, config, input_op, target_op, seq_len_op, mode):
        """
        Basic setup.
        Args:
          config: Object containing configuration parameters.
        """
        assert mode in ["training", "validation", "inference"]
        self.config = config
        self.inputs = input_op
        self.targets = target_op
        self.seq_lengths = seq_len_op
        self.mode = mode
        self.reuse = self.mode == "validation"

    def build_rnn_model(self):
        with tf.variable_scope('rnn_cell', reuse=self.reuse, initializer=self.config['initializer']):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.config['num_hidden_units'])
        with tf.variable_scope('rnn_stack', reuse=self.reuse, initializer=self.config['initializer']):
            if self.config['num_layers'] > 1:
                rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell for _ in range(self.config['num_layers'])])
            self.model_rnn, self.rnn_state = tf.nn.dynamic_rnn(
                                            cell=rnn_cell,
                                            inputs=self.inputs,
                                            dtype = tf.float32,
                                            sequence_length=self.seq_lengths,
                                            time_major=False,
                                            swap_memory=True)
            # Fetch output of the last step.
            if self.config['loss_type'] == 'last_step':
                self.rnn_prediction = tf.gather_nd(self.model_rnn, tf.stack([tf.range(self.config['batch_size']), self.seq_lengths-1], axis=1))
            elif self.config['loss_type'] == 'average':
                self.rnn_prediction = self.model_rnn
            else:
                print("Invalid loss type")
                raise


    def build_model(self):
        self.build_rnn_model()
        # Calculate logits
        with tf.variable_scope('logits', reuse=self.reuse, initializer=self.config['initializer']):
            self.logits = tf.layers.dense(inputs=self.rnn_prediction, units=self.config['num_class_labels'],
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.contrib.layers.xavier_initializer())

            # In the case of average loss, take average of time steps in order to calculate
            # final prediction probabilities.
            if self.config['loss_type'] == 'average':
                self.logits = tf.reduce_mean(self.logits, axis=1)

    def loss(self):
        if self.mode is not "inference":
            # Loss calculations: cross-entropy
            with tf.name_scope("cross_entropy_loss"):
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))

                # Accuracy calculations.
        with tf.name_scope("accuracy"):
            # Return list of predictions (useful for making a submission)
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

            if self.mode is not "inference":
                # Return a bool tensor with shape [batch_size] that is true for the
                # correct predictions.
                self.correct_predictions = tf.equal(tf.argmax(self.logits, 1), self.targets)
                # Number of correct predictions in order to calculate average accuracy afterwards.
                self.num_correct_predictions = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32))
                # Calculate the accuracy per minibatch.
                self.batch_accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

    def build_graph(self):
        self.build_model()
        self.loss()
        self.num_parameters()

    def num_parameters(self):
        self.num_parameters = 0
        #iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters=1
            shape = variable.get_shape()  #getting shape of a variable
            for i in shape:
                local_parameters*=i.value  #mutiplying dimension values
            self.num_parameters+=local_parameters
