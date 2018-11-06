import tensorflow as tf

def save_model(sess, path, inputs, outputs):
    tf.saved_model.simple_save(sess, path, inputs, outputs)

def get_c3d_graph_nodes(
        clip_size, height, width,
        activation=tf.nn.relu,
        batch_norm_momentum=0.9,
        starter_learning_rate=0.01,
        initializer=tf.contrib.layers.xavier_initializer()):

    X = tf.placeholder(tf.float32, shape=(None, clip_size, height, width, 3), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    is_training = tf.placeholder(tf.bool, name='training')

    layer1 = _one_conv3d_layer(
        X,
        is_training=is_training,
        filters=64,
        conv3d_kernel_size=(3, 3, 3),
        pool3d_size=(1, 2, 2),
        pool3d_strides=(1, 2, 2),
        activation=activation,
        batch_norm_momentum=batch_norm_momentum,
        initializer=initializer,
        layer_number=1,
        batchnorm=True
    )

    layer2 = _one_conv3d_layer(
        layer1,
        is_training=is_training,
        filters=128,
        conv3d_kernel_size=(3, 3, 3),
        pool3d_size=(2, 2, 2),
        pool3d_strides=(2, 2, 2),
        activation=activation,
        batch_norm_momentum=batch_norm_momentum,
        initializer=initializer,
        layer_number=2,
        batchnorm=True
    )

    layer3 = _two_conv3d_layer(
        layer2,
        is_training=is_training,
        filters=[256, 256],
        conv3d_kernel_sizes=[(3, 3, 3), (3, 3, 3)],
        pool3d_size=(2, 2, 2),
        pool3d_strides=(2, 2, 2),
        activation=activation,
        batch_norm_momentum=batch_norm_momentum,
        initializer=initializer,
        layer_number=3,
        batchnorm=True
    )

    layer4 = _two_conv3d_layer(
        layer3,
        is_training=is_training,
        filters=[512, 512],
        conv3d_kernel_sizes=[(3, 3, 3), (3, 3, 3)],
        pool3d_size=(2, 2, 2),
        pool3d_strides=(2, 2, 2),
        activation=activation,
        batch_norm_momentum=batch_norm_momentum,
        initializer=initializer,
        layer_number=4,
        batchnorm=True
    )

    layer5 = _two_conv3d_layer(
        layer4,
        is_training=is_training,
        filters=[256, 512],
        conv3d_kernel_sizes=[(3, 3, 3), (3, 3, 3)],
        pool3d_size=(2, 2, 2),
        pool3d_strides=(2, 2, 2),
        activation=activation,
        batch_norm_momentum=batch_norm_momentum,
        initializer=initializer,
        layer_number=5,
        batchnorm=True
    )

    gmp = tf.keras.layers.GlobalMaxPooling3D() (layer5)

    fc6 = _dense_layer(
        inputs=gmp,
        is_training=is_training,
        units=256,
        activation=activation,
        batch_norm_momentum=batch_norm_momentum,
        initializer=initializer,
        layer_number=6,
        batchnorm=True
    )

    fc7 = _dense_layer(
        inputs=fc6,
        is_training=is_training,
        units=128,
        activation=activation,
        batch_norm_momentum=batch_norm_momentum,
        initializer=initializer,
        layer_number=7,
        batchnorm=True
    )

    logits = tf.layers.dense(
        inputs=fc7,
        kernel_initializer=initializer,
        bias_initializer=initializer,
        units=1,
        activation=None,
        name='logits'
    )

    probabilities = tf.nn.sigmoid(logits, name='probabilities')
    predictions = tf.round(probabilities, name='predictions')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y), tf.float32), name='accuracy')
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=3.5), name='loss')

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 50, 0.9, staircase=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    graph_nodes = {
        'X': X,
        'y': y,
        'is_training': is_training,
        'optimizer': optimizer,
        'loss': loss,
        'accuracy': accuracy,
        'probabilities': probabilities
    }

    return graph_nodes

def _one_conv3d_layer(
        inputs, is_training, filters,
        conv3d_kernel_size, pool3d_size, pool3d_strides,
        activation, batch_norm_momentum, initializer, layer_number, batchnorm=True):

    conv3d = tf.layers.conv3d(
        inputs=inputs,
        filters=filters,
        kernel_initializer=initializer,
        bias_initializer=initializer,
        kernel_size=conv3d_kernel_size,
        padding='same',
        activation=activation,
        name='conv{0}'.format(layer_number)
    )

    if batchnorm:
        with tf.variable_scope('batchnorm', reuse=tf.AUTO_REUSE):
            conv3d = tf.layers.batch_normalization(
                inputs=conv3d,
                momentum=batch_norm_momentum,
                training=is_training,
                name='conv{0}_batchnorm'.format(layer_number)
            )

    pool3d = tf.layers.max_pooling3d(
        inputs=conv3d,
        pool_size=pool3d_size,
        strides=pool3d_strides,
        padding='valid',
        name='pool{0}'.format(layer_number)
    )

    return pool3d

def _two_conv3d_layer(
        inputs, is_training, filters,
        conv3d_kernel_sizes, pool3d_size, pool3d_strides,
        activation, batch_norm_momentum, initializer, layer_number, batchnorm=True):

    conv3da = tf.layers.conv3d(
        inputs=inputs,
        filters=filters[0],
        kernel_initializer=initializer,
        bias_initializer=initializer,
        kernel_size=conv3d_kernel_sizes[0],
        padding='same',
        activation=activation,
        name='conv{0}a'.format(layer_number)
    )

    if batchnorm:
        with tf.variable_scope('batchnorm', reuse=tf.AUTO_REUSE):
            conv3da = tf.layers.batch_normalization(
                inputs=conv3da,
                momentum=batch_norm_momentum,
                training=is_training,
                name='conv{0}a_batchnorm'.format(layer_number)
            )

    conv3db = tf.layers.conv3d(
        inputs=conv3da,
        filters=filters[1],
        kernel_initializer=initializer,
        bias_initializer=initializer,
        kernel_size=conv3d_kernel_sizes[1],
        padding='same',
        activation=activation,
        name='conv{0}b'.format(layer_number)
    )

    if batchnorm:
        with tf.variable_scope('batchnorm', reuse=tf.AUTO_REUSE):
            conv3db = tf.layers.batch_normalization(
                inputs=conv3db,
                momentum=batch_norm_momentum,
                training=is_training,
                name='conv{0}b_batchnorm'.format(layer_number)
            )

    pool3d = tf.layers.max_pooling3d(
        inputs=conv3db,
        pool_size=pool3d_size,
        strides=pool3d_strides,
        padding='valid',
        name='pool{0}'.format(layer_number)
    )

    return pool3d

def _dense_layer(
        inputs, is_training, units,
        activation, batch_norm_momentum, initializer, layer_number, batchnorm=True):

    fc = tf.layers.dense(
        inputs=inputs,
        kernel_initializer=initializer,
        bias_initializer=initializer,
        units=units,
        activation=activation,
        name='fc{0}'.format(layer_number)
    )

    if batchnorm:
        with tf.variable_scope('batchnorm', reuse=tf.AUTO_REUSE):
            fc = tf.layers.batch_normalization(
                inputs=fc,
                momentum=batch_norm_momentum,
                training=is_training,
                name='fc{0}_batchnorm'.format(layer_number)
            )

    return fc