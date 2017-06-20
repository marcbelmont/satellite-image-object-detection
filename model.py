import numpy as np
import tensorflow as tf
from params import CELLS, LABEL_SHAPE, CLASSES, CELL_BOXES, IMG_SIZE

slim = tf.contrib.slim

#########
# Model #
#########


def create_model(images, alpha):
    with tf.variable_scope('yolo'):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                normalizer_fn=slim.batch_norm,
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(images, 32, 3)
            net = slim.max_pool2d(net, 2,)
            net = slim.conv2d(net, 64, 3, )
            net = slim.max_pool2d(net, 2,)
            net = slim.conv2d(net, 128, 3, )
            net = slim.conv2d(net, 64, 1, )
            net = slim.conv2d(net, 128, 3, )
            net = slim.max_pool2d(net, 2,)
            net = slim.conv2d(net, 256, 3, )
            net = slim.conv2d(net, 128, 1, )
            net = slim.conv2d(net, 256, 4, )
            net = slim.max_pool2d(net, 2,)
            net = slim.conv2d(net, 512, 3, )
            net = slim.conv2d(net, 256, 1,)
            net = slim.conv2d(net, 512, 3, )
            net = slim.conv2d(net, 256, 1,)
            net = slim.conv2d(net, 512, 3, )
            net = slim.max_pool2d(net, 2,)
            net = slim.conv2d(net, 1024, 3, )
            net = slim.conv2d(net, 512, 1,)
            net = slim.conv2d(net, 1024, 3, )
            net = slim.conv2d(net, 512, 1,)
            net = slim.conv2d(net, 1024, 3, )
            net = slim.conv2d(net, LABEL_SHAPE[-1], 1,
                              normalizer_fn=None,
                              activation_fn=None)
    return net


def model_endpoints(net, offset):
    confidence, p_box_param, p_classes = tf.split(net, [1, 4, CLASSES], 3)
    p_box_param = tf.reshape(p_box_param, [-1, CELLS, CELLS, CELL_BOXES, 4])
    p_box = tf.stack([
        (p_box_param[:, :, :, :, 0] + offset) / CELLS,  # x
        (p_box_param[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / CELLS,  # y
        tf.square(p_box_param[:, :, :, :, 2]),  # w
        tf.square(p_box_param[:, :, :, :, 3])])  # h
    p_box = tf.transpose(p_box, [1, 2, 3, 4, 0])
    return confidence, p_box_param, p_classes, p_box


#############
# Inference #
#############

def inference(output, threshold):
    offset = offset_map(output)
    confidence, p_box_param, p_classes, p_box = model_endpoints(output, offset)
    top_class = tf.reduce_max(p_classes, 3, keep_dims=True)
    top_class *= confidence
    mask = top_class > threshold

    # extract non zeros boxes
    box_list = tf.gather_nd(
        tf.reshape(p_box * IMG_SIZE, [-1, CELLS, CELLS, 4]),
        tf.where(tf.reshape(mask, [1, CELLS, CELLS])))
    box_list = tf.reshape(box_list, [-1, 4])

    # extract classes
    class_list = tf.gather_nd(
        tf.argmax(p_classes, 3),
        tf.where(tf.reshape(mask, [1, CELLS, CELLS])))
    class_list = tf.reshape(class_list, [-1])

    # confidence
    confidence_list = tf.gather_nd(top_class, tf.where(mask))
    return (box_list,
            class_list,
            confidence_list,
            tf.reshape(mask, [1, CELLS, CELLS]))


########
# Loss #
########


def create_loss(net, labels):
    ##########
    # Labels #

    mask, box, classes = tf.split(labels, [1, 4, CLASSES], 3)
    box /= IMG_SIZE
    box = tf.reshape(box, [-1, CELLS, CELLS, 1, 4])
    box = tf.tile(box, [1, 1, 1, CELL_BOXES, 1])
    offset = offset_map(net)
    box_param = tf.stack([
        box[:, :, :, :, 0] * CELLS - offset,
        box[:, :, :, :, 1] * CELLS - tf.transpose(offset, (0, 2, 1, 3)),
        tf.sqrt(box[:, :, :, :, 2]),  # w
        tf.sqrt(box[:, :, :, :, 3])])  # h
    box_param = tf.transpose(box_param, [1, 2, 3, 4, 0])

    ###############
    # Predictions #

    confidence, p_box_param, p_classes, p_box = model_endpoints(net, offset)

    #######
    # IOU #

    iou = calc_iou(p_box, box)
    object_mask = tf.reduce_max(iou, 3, keep_dims=True)
    object_mask = mask * tf.cast((iou >= object_mask), tf.float32)
    noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

    ##########
    # Losses #

    # Class loss
    class_delta = mask * (classes - p_classes)
    class_loss = 2 * tf.reduce_mean(
        tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]))

    # Object losses
    object_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(object_mask * (confidence - iou)), axis=[1, 2, 3]))
    noobject_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(noobject_mask * confidence), axis=[1, 2, 3]))

    # Boxes loss
    boxes_delta = tf.expand_dims(object_mask, 4) * (p_box_param - box_param)
    box_loss = 5 * tf.reduce_mean(
        tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3]))

    # Total loss
    tf.losses.add_loss(class_loss)
    tf.summary.scalar('class_loss', class_loss)
    tf.losses.add_loss(object_loss)
    tf.summary.scalar('object_loss', object_loss)
    tf.losses.add_loss(noobject_loss)
    tf.summary.scalar('noobject_loss', noobject_loss)
    tf.losses.add_loss(box_loss)
    tf.summary.scalar('box_loss', box_loss)
    loss = tf.losses.get_total_loss()  # Includes regularization losses
    tf.summary.scalar('total_loss', loss)
    return loss

###########
# Helpers #
###########


def calc_iou(boxes1, boxes2):
    with tf.variable_scope('iou'):
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
            (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
            (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def offset_map(net):
    # [[[0, 0, 0, 0, 0], [1,..]..],.. [..[6,..]]]
    offset = np.array([np.arange(CELLS)] * CELLS * CELL_BOXES)
    offset = np.reshape(offset, (CELL_BOXES, CELLS, CELLS))
    offset = np.transpose(offset, (1, 2, 0))
    offset = tf.constant(offset, dtype=tf.float32)
    offset = tf.reshape(offset, [1, CELLS, CELLS, CELL_BOXES])
    offset = tf.tile(offset, [tf.shape(net)[0], 1, 1, 1])
    return offset


def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op
