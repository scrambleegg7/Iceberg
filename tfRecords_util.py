#
import tensorflow as tf

from tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim



lr = 0.001
decay_rate=0.1
decay_per=40 #epoch
num_iter = int(1604/32)

def infer(inputs, is_training=True):
    inputs = tf.cast(inputs, tf.float32)
    inputs = ((inputs / 255.0)-0.5)*2
    #Use Pretrained Base Model
    with tf.variable_scope("vgg_16"):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
    #Append fully connected layer
    net = slim.flatten(net)
    net = slim.fully_connected(net, 512,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            scope='finetune/fc1')
    net = slim.fully_connected(net, 2,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            scope='finetune/fc2')
    return net

def read_and_decode(filename_queue,BATCH_SIZE=32,shuffle=True):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'image/encoded/orig': tf.FixedLenFeature([], tf.string),
        #'image/format': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/depth': tf.FixedLenFeature([], tf.int64)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    image_o = tf.decode_raw(features['image/encoded/orig'], tf.uint8)

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    depth = tf.cast(features['image/depth'], tf.int32)

    image_shape = tf.stack([height, width, depth])
    print(image_shape)
    image_o = tf.reshape(image_o, image_shape)

    label = tf.cast(features['image/class/label'], tf.int32)
    #label = tf.reshape(label, tf.stack([1,AMAZON_CLASSES]) )
    #label.set_shape(1)
    #label = tf.cast(label, tf.float32)

    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.

    IMAGE_HEIGHT = 75
    IMAGE_WIDTH = 75
    image_o.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    image_o = tf.image.resize_image_with_crop_or_pad(image=image_o,
                                           target_height=224,
                                           target_width=224)

    if shuffle:
        images_o,labels = tf.train.shuffle_batch( [image_o, label],
                                                 batch_size=BATCH_SIZE,
                                                 capacity=256,
                                                 num_threads=2,
                                                 min_after_dequeue=32)
    else:
        images_o,labels = tf.train.batch( [image_o, label],
                                            batch_size=BATCH_SIZE ,
                                            allow_smaller_final_batch=False)




    return images_o,labels,height,width

def losses(logits, labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def optimize(losses):
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(lr, global_step,
                                             num_iter*decay_per, decay_rate, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(losses, global_step=global_step)#,
                #var_list=slim.get_model_variables("finetune"))
    return train_op

def main():

    #Hyper Parameter to play with
    batch_size=32
    num_epochs=10




    tfrecords_filename='tfRecords/train.tfrecords'

    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=num_epochs)
    image, label,h,w = read_and_decode(filename_queue,shuffle=False)

    prediction = infer(image)
    loss = losses(prediction, label)
    train_op = optimize(loss)


    with tf.Session() as sess:

        init_op = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
        restore = slim.assign_from_checkpoint_fn(
                   'vgg16/vgg_16.ckpt',
                   slim.get_model_variables("vgg_16"))
        sess.run(init_op)
        restore(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for e in range(num_epochs):
            avg_loss, acc = 0, 0
            for i in range(num_iter):
                _, l = sess.run([train_op, loss])
                avg_loss += l/num_iter
            print("Epoch%03d avg_loss: %f" % (e+1, avg_loss) )

        coord.request_stop()
        coord.join(threads)
        print('Training Done')
        saver = tf.train.Saver(slim.get_model_variables())
        saver.save(sess, 'model.ckpt')
        sess.close()



if __name__ == "__main__":
    main()
