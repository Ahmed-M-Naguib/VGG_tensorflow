import tensorflow as tf
import VGG
import tools
import input_data

IMG_W = 32
IMG_H = 32
N_CLASSES = 1000
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 15000
IS_PRETRAIN = True


# logits = VGG.VGG16(train_image_batch, N_CLASSES, IS_PRETRAIN)
# loss = tools.loss(logits, train_label_batch)
# accuracy = tools.accuracy(logits, train_label_batch)
# my_global_step = tf.Variable(0, trainable=False, name='global_step')
# train_op = tools.optimize(loss, learning_rate, my_global_step)
#
# x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_H, IMG_W, 3])
# y_ = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, N_CLASSES])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

with tf.name_scope('input'):
    val_image_batch, val_label_batch = input_data.read_cifar10('./cifar10/cifar-10-batches-bin',
                                                               is_train=False,
                                                               batch_size=BATCH_SIZE,
                                                               shuffle=False)

logits = VGG.VGG16(val_image_batch, N_CLASSES, IS_PRETRAIN)



# load pretrain weights
tools.load_with_skip('./VGG16_pretrain/vgg16.npy', sess, [])

# with tf.Graph().as_default():
#     log_dir = './logs2/train/'
#     test_dir = '/home/yuxin/data/cifar10_data/'
#     n_test = 10000
#
#     logits = VGG.VGG16(test_iamge_batch, N_CLASSES, IS_PRETRAIN)
#     correct = tools.num_correct_prediction(logits, test_label_batch)
#     saver = tf.train.Saver(tf.global_variables())
#
#     with tf.Session() as sess:
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         try:
#             print('\nEvaluating...')
#             num_step = int(math.ceil(n_test / BATCH_SIZE))
#             num_example = num_step * BATCH_SIZE
#             step = 0
#             total_correct = 0
#             while step < num_step and not coord.should_stop():
#                 batch_correct = sess.run(correct)
#                 total_correct += np.sum(batch_correct)
#                 step += 1
#
#             print("Total test examples: %d" % num_example)
#             print("Total correct predictions: %d" % total_correct)
#             print("Average accuracy: %.2f%%" % (100 * total_correct / num_example))
#         except Exception as e:
#             coord.request_stop(e)
#         finally:
#             coord.request_stop()
#             coord.join(threads)
