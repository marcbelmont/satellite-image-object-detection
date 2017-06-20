from preprocess import extract_crops_sw
from main import create_batch, create_loss, IMG_SIZE, create_model, LABEL_SHAPE, load_inference, load_images
from model import inference
import pandas as pd
import tensorflow as tf
import numpy as np


class ModelTest(tf.test.TestCase):

    def setUp(self, ):
        pass

    def test_inference_loader(self):
        with self.test_session() as sess:
            net, input = load_inference(sess, 'checkpoints', .1)
            path = "data/training_sliding/TQ2379_0_0/TQ2379_0_0_x1650y0.jpg"
            sess.run([net], feed_dict={input: load_images([path])})

    def test_inference(self):
        with self.test_session() as sess:
            # Create model
            net = create_model(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]), .1)
            net_ph = tf.placeholder(tf.float32, shape=net.shape)
            infer = inference(net_ph, .1)

            # Test inference results
            output = np.zeros(net.shape).astype(np.float32)
            output[0, 1, 1, :5] = [.84, .4, .68, .346, .346]
            output[0, 1, 1, 10] = .3  # class
            output[0, 2, 2, :5] = [.84, .4, .68, .346, .346]
            output[0, 2, 2, 11] = .03  # class
            result = sess.run([infer], feed_dict={net_ph: output})
            p_box, p_classes, confidence, mask = result[0]

            # Test
            self.assertEqual(mask[0, 1, 1], 1)
            self.assertEqual(p_classes, 5)
            self.assertEqual(confidence, .3 * .84)
            self.assertListEqual(
                [round(x) for x in p_box.tolist()[0]],
                [50, 60, 30, 30],)

    def test_loss(self):
        with self.test_session() as sess:
            # Use batch size of 1
            net = create_model(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]), .1)
            output = np.zeros(net.shape).astype(np.float32)
            # Confidence, x_c_rel, y_c_rel, h_sqrt, w_sqrt
            output[0, 1, 1, :5] = [1, .4, .68, .346, .346]
            output[0, 1, 1, 10] = 1  # class

            labels = np.zeros([1] + LABEL_SHAPE).astype(np.float32)
            # Mask, x_c, y_c, h, w
            labels[0, 1, 1, :5] = [1, 50, 60, 30, 30]
            labels[0, 1, 1, 10] = 1  # class

            create_loss(tf.constant(output), tf.constant(labels))
            sess.run(tf.global_variables_initializer())
            for name, loss in zip('class obj no-obj box'.split(),
                                  tf.losses.get_losses()):
                x = loss.eval()
                print("%10s %.3f" % (name, x))
                self.assertAlmostEqual(x, 0, 4)


class DataTest(tf.test.TestCase):

    def test_stats(self):
        df = pd.read_csv('data/training_cropped.csv')
        # df = pd.read_csv('data/trainingObservations.csv')
        # print(df['class_id'].value_counts())
        # df_out = extract_crops(df, 250, False)
        # df_out = df_out.sort_values(0, ascending=False)
        df.head()

    def test_batch(self):
        with self.test_session() as sess:
            df = pd.DataFrame(
                ['TQ2379_0_0_B  TQ2379_0_0.jpg      F  1776:520|1824:125'.split(),
                 'TQ2379_0_0_B  TQ2379_0_0.jpg      F  1776:500|1824:125'.split(),
                ],
                columns=['id', 'image', 'class', 'detections'])
            df = extract_crops_sw(df, 250, False, 250)
            batch = create_batch(df, False)

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            images, labels = sess.run(batch)
            self.assertListEqual(list(labels[0, 2, 3, :5]),
                                 [1., 74., 125., 30., 30.])
            self.assertTrue(labels[0, 2, 3, 5 + 5])

            coord.request_stop()
            coord.join(threads)

    def test_dataset_sw(self):
        df = pd.DataFrame(
            ['TQ2379_0_0_B  TQ2379_0_0.jpg      A  1776:520|1824:125'.split(),
             'TQ2379_0_0_B  TQ2379_0_0.jpg      C  1760:120'.split(),
             'TQ2379_0_0_B  TQ2379_0_0.jpg      B  1760:456|1760:456'.split(),
             'TQ2379_0_0_B  TQ2379_0_0.jpg      D  1060:120'.split(),
            ],
            columns=['id', 'image', 'class', 'detections'])
        df_out = extract_crops_sw(df, 250, False, 150)
        df_out.path = df_out.path.apply(lambda x: x[-20:])
        print(df_out)
        self.assertEqual(df_out.shape[0], 8)


if __name__ == '__main__':
    tf.test.main()
