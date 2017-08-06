import tensorflow as tf
import VGG
import tools
import input_data
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

IMG_W = 32
IMG_H = 32
N_CLASSES = 1000
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 15000
IS_PRETRAIN = True

# img = misc.imread('./image/motocycle.jpg')
# img=misc.imresize(img,[224,224,3])
# img_tf = tf.Variable(img)
# testImage = sess.run(img_tf)

# filename_queue = tf.train.string_input_producer(['./image/motocycle.jpg']) #  list of files to read
# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)
# my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
# my_img2 = tf.image.resize_images(my_img,[224,224])
# # my_img = tf.reshape(my_img, [1, 224, 224, 3])
# # Start populating the filename queue.
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord, sess=sess)
# for i in range(1): #length of your filename list
#     testImage = my_img2.eval(session=sess) #here is your image Tensor :)
# coord.request_stop()
# coord.join(threads)

# print(testImage.shape)
# fig = plt.figure()
# plt.imshow(testImage)
# plt.show()


# with tf.Graph().as_default():


img = misc.imread('./image/motocycle.jpg')
img=misc.imresize(img,[224,224,3])
img_tf = tf.Variable(img)
img_tf2 = tf.reshape(img_tf,[1,224,224,3])
img_tf2 = tf.cast(img_tf2, tf.float32)
img_tf3 = tf.reshape(img_tf2,[224,224,3])


# Prepare VGG16 Model
print('Preparing VGG16 Model ...\n')
VGG.VGG16_Model()


saver = tf.train.Saver(tf.global_variables())

# load pretrain weights
print('Loading VGG16 Pretrained Params ...\n')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())

# testImage = sess.run(img_tf)
# testImage3 = sess.run(img_tf3)
# print(testImage.shape)
# print(testImage3.shape)
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(testImage)
# fig.add_subplot(1,2,2)
# plt.imshow(testImage3)
# plt.show()

tools.load_with_skip('./VGG16_pretrain/vgg16.npy', sess, [])

print('Testing Network!\n')
logits = VGG.VGG16(img_tf2, 1000, True)
print(logits.eval(session=sess), '\n')


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


print('Done!\n')