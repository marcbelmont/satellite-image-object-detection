from PIL import Image
from main import load_images, sample_images
from main import load_inference
from visualization_utils import draw_bounding_box_on_image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
net, input = load_inference(sess, 'checkpoints/small_lr', .03)


COLORS = ['#ff00ff', '#dddddd', '#dddddd', '#888888', '#888888', '#ff0000', '#ff0000', '#dddddd']
NAMES = 'moto short long short long short long van'.split()

image_list, _ = sample_images()

for path in image_list[:]:
    results = sess.run([net], feed_dict={input: load_images([path])})
    p_box, p_classes, p_confidence, _ = results[0]
    image = Image.open(path)
    boxes = np.zeros(p_box.shape)
    boxes[:, 0] = p_box[:, 1] - p_box[:, 3] / 2
    boxes[:, 1] = p_box[:, 0] - p_box[:, 2] / 2
    boxes[:, 2] = p_box[:, 1] + p_box[:, 3] / 2
    boxes[:, 3] = p_box[:, 0] + p_box[:, 2] / 2
    boxes /= image.size[0]
    for i in range(boxes.shape[0]):
        draw_bounding_box_on_image(image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                                   COLORS[p_classes[i]], 2, ['%.2f %s' % (p_confidence[i], NAMES[p_classes[i]])])
    fig, ax = plt.subplots(1, figsize=(4, 4), dpi=80)
    ax.imshow(image)
plt.show()
