from time import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from model import create_model, create_loss, LABEL_SHAPE, IMG_SIZE, CELLS, inference
slim = tf.contrib.slim
from skimage.data import imread

#########
# Flags #
#########

tf.app.flags.DEFINE_boolean("debug", False, "True if debug mode")
tf.app.flags.DEFINE_boolean("submit", False, "Create submission")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size")
tf.app.flags.DEFINE_float("learning_rate", .0001, "Learning rate")
tf.app.flags.DEFINE_string("ckptdir", None, "Checkpoint")
tf.app.flags.DEFINE_string("dataset", 'data/training_cropped.csv', "Images and labels")
tf.app.flags.DEFINE_string("logdir", None, "Directory to save logs")
FLAGS = tf.app.flags.FLAGS


if FLAGS.debug:
    tf.set_random_seed(1)
    np.random.seed(1)

#############
# Read data #
#############


def read_image(input_queue, shuffle):
    image_file = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_file, 3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Data augmentation
    augment = tf.random_uniform([2]) > .5
    if shuffle:
        image = tf.cond(
            augment[0], lambda: tf.image.flip_left_right(image), lambda: image)
        image = tf.cond(
            augment[1], lambda: tf.image.flip_up_down(image), lambda: image)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    image.set_shape([IMG_SIZE, IMG_SIZE, 3])

    # Annotation: [class_id, center_x, center_y, height, width] * n
    annotations = tf.reshape(input_queue[1], [-1, 5])

    def feat_map(annotations, augment, shuffle):
        label = np.zeros(LABEL_SHAPE).astype(np.float32)
        for row in annotations:
            class_id, x, y, h, w = row
            if np.isnan(class_id):
                continue
            if shuffle:
                if augment[0]:
                    x = IMG_SIZE - x - 1
                if augment[1]:
                    y = IMG_SIZE - y - 1
            x_cell = int(CELLS * x / IMG_SIZE)
            y_cell = int(CELLS * y / IMG_SIZE)
            label[x_cell, y_cell, :5] = [1, x, y, h, w]
            label[x_cell, y_cell, 5 + int(class_id)] = 1
        return label

    label = tf.py_func(feat_map, [annotations, augment, shuffle], tf.float32)
    label.set_shape(LABEL_SHAPE)
    return [image, label]


def create_batch(df, shuffle):
    # Extract clean examples
    image_list = df['path'].tolist()
    label_list = df.iloc[:, 1:].as_matrix()
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.float32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=None if shuffle else 1,
                                                shuffle=shuffle)
    res = read_image(input_queue, shuffle)
    num_preprocess_threads = 1 if FLAGS.debug else 4
    min_queue_examples = 100 if FLAGS.debug else 1000
    if shuffle:
        inputs = tf.train.shuffle_batch(
            res,
            batch_size=FLAGS.batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * FLAGS.batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        inputs = tf.train.batch(
            res,
            FLAGS.batch_size,
            allow_smaller_final_batch=True)
    return inputs

#########
# Train #
#########


def detector(sess):
    # load one image
    df = pd.DataFrame.from_csv(FLAGS.dataset)
    images, labels = create_batch(df.iloc[:-100], True)
    v_batch = create_batch(df.iloc[-100:], True)

    #######################
    # Model and objective #
    #######################

    # Loss and optimizer
    net = create_model(images, .1)
    loss = create_loss(net, labels)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(
            FLAGS.learning_rate).minimize(
                loss,
                global_step=global_step)

    # Summary
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    ########
    # Init #

    sess.run(tf.global_variables_initializer())
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver(max_to_keep=10)
    if FLAGS.ckptdir and os.path.exists(FLAGS.ckptdir) and not FLAGS.debug:
        checkpoint = tf.train.latest_checkpoint(FLAGS.ckptdir)
        if checkpoint:
            print('Restoring', checkpoint)
            saver.restore(sess, checkpoint)

    #########
    # Train #

    try:
        # Train
        while not coord.should_stop():
            start_time = time()
            _, tr_loss, sum_str, g_step = sess.run(
                [train_step, loss, summary, global_step])
            batch_time = 1000 * (time() - start_time) / FLAGS.batch_size

            # Calculate loss on validation set
            if g_step % 1000 == 0 or FLAGS.debug:
                images_v, labels_v = sess.run(v_batch)
                val_loss = sess.run(loss, feed_dict={images: images_v, labels: labels_v})
            else:
                val_loss = -1

            # Stats
            summary_writer.add_summary(sum_str, g_step)
            if g_step % (1 if FLAGS.debug else 100) == 0:
                print('[%5s] Loss %.3f, Val. loss: %.3f, Time: %dms' % (
                    g_step, tr_loss, val_loss, batch_time))

            # Save model
            if g_step % 1000 == 0 and FLAGS.ckptdir and not FLAGS.debug:
                print('Saving model')
                saver.save(
                    sess,
                    os.path.join(FLAGS.ckptdir, 'model.ckpt'),
                    global_step=global_step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

#############
# Inference #
#############


def load_inference(sess, ckptdir, threshold):
    images = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
    net = create_model(images, .1)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    if ckptdir and os.path.exists(ckptdir) and not FLAGS.debug:
        checkpoint = tf.train.latest_checkpoint(ckptdir)
        if checkpoint:
            print('Restoring', checkpoint)
            saver.restore(sess, checkpoint)
    return inference(net, threshold), images

###########
# Helpers #
###########


def load_images(paths):
    images = []
    for path in paths:
        image = imread(path)
        image = (image / 255 - .5) / 2
        images += [image]
    return np.stack(images)


def sample_images():
    df = pd.DataFrame.from_csv(FLAGS.dataset)
    df = df.sample(10)
    image_list = df['path'].tolist()
    label_list = df.iloc[:, 1:].as_matrix()
    return image_list, label_list

########
# Main #
########


def main(args):
    with tf.Session() as sess:
        detector(sess)

if __name__ == '__main__':
    tf.app.run()
