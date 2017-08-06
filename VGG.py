
import tensorflow as tf
import tools
import numpy as np


def VGG16_Model(is_pretrain=False):
    emptyImage = tf.Variable(np.zeros([1,224,224,3]), dtype=tf.float32)
    VGG16(emptyImage,1000,is_pretrain, False)


def VGG16(x, n_class, is_pretrain=True, reuse=True):

    # using the name scope, the tensorboard maybe look better
    with tf.name_scope('VGG16'):

        x = tools.conv('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        x = tools.conv('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        with tf.name_scope('pool1'):
            x = tools.pool('pool1', x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        x = tools.conv('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        with tf.name_scope('pool2'):
            x = tools.pool('pool2', x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        x = tools.conv('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        x = tools.conv('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        with tf.name_scope('pool3'):
            x = tools.pool('pool3', x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        x = tools.conv('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        x = tools.conv('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        with tf.name_scope('pool4'):
            x = tools.pool('pool4', x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        x = tools.conv('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        x = tools.conv('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain, reuse=reuse)
        with tf.name_scope('pool5'):
            x = tools.pool('pool5', x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.FC_layer('fc6', x, out_nodes=4096, reuse=reuse)
        with tf.name_scope('batch_norma1'):
            x = tools.batch_norm(x)     # batch norm can avoid overfit, more efficient than dropout
        x = tools.FC_layer('fc7', x, out_nodes=4096, reuse=reuse)
        with tf.name_scope('batch_norm2'):
            x = tools.batch_norm(x)
        x = tools.FC_layer('fc8', x, out_nodes=n_class, reuse=reuse)

        return x
