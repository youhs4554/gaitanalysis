import tensorflow as tf

class C3DNet:
    def __init__(self, pretrained_model_path, batch_size=1, scope=None, trainable=True):
        
        self.batch_size = batch_size
        
        if scope == None:
            self.scope = 'C3D'
            
        print('initialize with pretrained weight file...')

        with tf.variable_scope(self.scope):
            # load pre-trained weights(C3D)
            self._weights = {}
            self._biases = {}
            for var_name, var_shape in tf.contrib.framework.list_variables(pretrained_model_path):
                # load variable
                var = tf.contrib.framework.load_variable(pretrained_model_path, var_name)
                var_dict = self._biases if len(var_shape) == 1 else self._weights

                var_dict[var_name.split('/')[-1]] = tf.get_variable(var_name,
                                                                    var_shape,
                                                                    initializer=tf.constant_initializer(var),
                                                                    dtype='float32',
                                                                    trainable=trainable)
        
        print('Done!')

    def __call__(self, inputs):
        def conv3d(name, l_input, w, b):
            return tf.nn.bias_add(
                tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
                b, name=name)

        def max_pool(name, l_input, k):
            return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

        # Convolution Layer
        conv1 = conv3d('conv1', inputs, self._weights['wc1'], self._biases['bc1'])
        conv1 = tf.nn.relu(conv1, 'relu1')
        pool1 = max_pool('pool1', conv1, k=1)
        
        return pool1

        # Convolution Layer
        conv2 = conv3d('conv2', pool1, self._weights['wc2'], self._biases['bc2'])
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = max_pool('pool2', conv2, k=2)

        # Convolution Layer
        conv3 = conv3d('conv3a', pool2, self._weights['wc3a'], self._biases['bc3a'])
        conv3 = tf.nn.relu(conv3, 'relu3a')
        conv3 = conv3d('conv3b', conv3, self._weights['wc3b'], self._biases['bc3b'])
        conv3 = tf.nn.relu(conv3, 'relu3b')
        pool3 = max_pool('pool3', conv3, k=2)

        # Convolution Layer
        conv4 = conv3d('conv4a', pool3, self._weights['wc4a'], self._biases['bc4a'])
        conv4 = tf.nn.relu(conv4, 'relu4a')
        conv4 = conv3d('conv4b', conv4, self._weights['wc4b'], self._biases['bc4b'])
        conv4 = tf.nn.relu(conv4, 'relu4b')
        pool4 = max_pool('pool4', conv4, k=2)

        # Convolution Layer
        conv5 = conv3d('conv5a', pool4, self._weights['wc5a'], self._biases['bc5a'])
        conv5 = tf.nn.relu(conv5, 'relu5a')
        conv5 = conv3d('conv5b', conv5, self._weights['wc5b'], self._biases['bc5b'])
        conv5 = tf.nn.relu(conv5, 'relu5b')
        pool5 = max_pool('pool5', conv5, k=2)

#         return pool5

        # Fully connected layer
        # pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3]) # only for ucf
        dense1 = tf.reshape(pool5, [self.batch_size, self._weights['wd1'].get_shape().as_list()[
            0]])  # Reshape conv3 output to fit dense layer input
        dense1 = tf.matmul(dense1, self._weights['wd1']) + self._biases['bd1']

        dense1 = tf.nn.relu(dense1, name='fc1')  # Relu activation
        #dense1 = tf.nn.dropout(dense1, self.keep_rate)

        dense2 = tf.nn.relu(tf.matmul(dense1, self._weights['wd2']) + self._biases['bd2'], name='fc2')  # Relu activation
        #dense2 = tf.nn.dropout(dense2, self.keep_rate)
        
        return dense2

#         # Output: class prediction
#         out = tf.nn.softmax(tf.matmul(dense2, self._weights['wout']) + self._biases['bout'])

#         return out