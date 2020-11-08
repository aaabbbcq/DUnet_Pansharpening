
# -*- coding:utf-8 -*-
"""
@project: Research
@file:DUnet.py.py
@author: huangyihang
@create_time: 2020/3/13 11:13
@description:
"""
import tensorflow as tf
import numpy as np
import argparse
import math
import time
import collections
import os
import json

from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.ops import control_flow_ops

# from tflearn import global_avg_pool

from utils import array2raster

parser = argparse.ArgumentParser()
parser.add_argument("--train_tfrecord", help="filename of train_tfrecord",
                    default="/home/Zhanglp/sl/train.tfrecords")
parser.add_argument("--test_tfrecord", help="filename of test_tfrecord",
                    default="/home/Zhanglp/sl/test.tfrecords")
parser.add_argument("--mode", default='train', choices=["train", "test"])
parser.add_argument("--output_dir", help="where to put output files",
                    default="/home/Zhanglp/hyh/outputs/qb_out_DUnet/")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoints")
parser.add_argument("--max_steps", type=int, help="max training steps")
parser.add_argument("--max_epochs", type=int, default=40, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images ever display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps")

parser.add_argument("--batch_size", type=int, default=32, help="number of images in batch")  # 15

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on L1 term for generator gradient")

parser.add_argument("--ndf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--train_count", type=int, default=64000, help="number of training data")
parser.add_argument("--test_count", type=int, default=384, help="number of test data")
a = parser.parse_args()

EPS = 1e-12
Examples = collections.namedtuple("Examples",
                                  "inputs1,inputs2,inputs3,inputs4,inputs5,targets,steps_per_epoch")  # ,inputs8,inputs9
Model = collections.namedtuple("Model",
                               "DUnet_loss,DUnet_grads_and_vars,outputs,train,DUnet_loss_L1"
                               )


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding='SAME')
        return network

def conv_layer_3x3(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding='SAME')
        network_1 = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=[1,3], strides=stride,
                                     padding='SAME')
        network_2 = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=[3,1], strides=stride,
                                     padding='SAME')
        network = network + network_1 + network_2
        return network

def conv_scc(input, filter, slicec_, stride=4, layer_name="scc"):
    with tf.name_scope(layer_name):
            #print(type(slicec_))
            #is_training = tf.cast(True, tf.bool)
            print('input:',input)
            print('slice:',slicec_)
            x1, x2 = tf.split(input, [slicec_//2, slicec_//2], 3)
            w = tf.shape(x1)[1]
            h = w
            identity = conv_layer(x1, filter=filter, kernel=[1, 1],
                            layer_name=layer_name + '_conv_1')
            # x=tf.nn.avg_pool(input, stride, stride)
            x = Average_pooling(x1)
            x = tf.layers.conv2d(inputs=x, use_bias=False, filters=filter, kernel_size=3, strides=1,padding='SAME')
            x = tf.layers.batch_normalization(x,  center=True, scale=True, epsilon=0.001)
            x_u = tf.image.resize_images(x, [w, h])
            out = identity + x_u
            out = tf.nn.sigmoid(out)
            k3 = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=3, strides=1,padding='SAME')
            k3 = tf.layers.batch_normalization(k3,center=True, scale=True, epsilon=0.001)
            out = tf.multiply(k3, out)
            out = tf.layers.conv2d(inputs=out, use_bias=False, filters=filter, kernel_size=3, strides=1,padding='SAME')
            out = tf.layers.batch_normalization(out, center=True, scale=True, epsilon=0.001)
            out = tf.nn.relu(out)
            x2 = conv_layer(x2, filter=filter, kernel=[1, 1],
                            layer_name=layer_name + '_conv_2')
            out = tf.concat([out, x2], 3)
            return out

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def strided_conv(input, filter, kernel=2, stride=2, layer_name="strided_conv"):
    with tf.name_scope(layer_name):
        strided_conv = tf.layers.conv2d_transpose(inputs=input, filters=filter, kernel_size=kernel, strides=stride,
                                                  padding='same')
        return strided_conv

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return batch_norm(inputs=x, is_training=training, reuse=None)



def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return lrelu(x,0.2)


def Average_pooling(x, pool_size=[4, 4], stride=4, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


# Hyperparameter
growth_k = 24
batch_size = 32

class DpnNet():
    def __init__(self, blur, pan_row, pan_col, filters, ms):
        self.filters = filters
        self.model = self.Dpn_net(blur, pan_row, pan_col, ms)


    # 128 in | 128 4 | 4 4
    def transform_layer(self, x, filters, stride, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=filters, kernel=[1,1], stride=stride, layer_name=scope+"_conv1")
            # x = Batch_Normalization(x, training=True, scope=scope+"_batch1")
            x = Relu(x)

            x = conv_layer(x, filter=filters, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
            # x = Batch_Normalization(x, training=True, scope=scope + "_batch2")
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope): # out_dim=128
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
            return x

    def D_transition_layer(self, x, scope):
        with tf.name_scope(scope):
            # x = Batch_Normalization(x, training=True, scope=scope + '_batch1')
            x = Relu(x)
            in_channel = x.shape[-1]
            x = conv_layer(x, filter=in_channel // 2, kernel=[1, 1], layer_name=scope + '_conv1')
            # x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def split_layer(self, input_x, num_3x3_b, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            #filters = num_3x3_b/32
            filters = 4
            for i in range(16):
                splits = self.transform_layer(input_x, filters=filters,stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def DualPathFactory(self, data,  num_1x1_a, num_3x3_b, num_1x1_c, inc,  layer_num, _type='normal'):
        # split + transform(bottleneck) + transition + merge
        if _type == 'normal':
            has_proj = False
        else:
            has_proj = True

        if type(data) is list:
            data_in = Concatenation(data)
            print(data_in.shape)
        else:
            data_in = data

        if has_proj:
            data_o1 , data_o2 = tf.split(data_in, [num_1x1_c,2*inc], 3)
        else:
            data_o1 = data[0]
            data_o2 = data[1]

        stride = 1

        x = conv_layer(data_in, filter=num_1x1_a, kernel=[1, 1], stride=stride)
        x = Relu(x)

        layers_split = list()
        filters = 4
        for i in range(16):
            splits = self.transform_layer(x, filters=filters,stride=stride, scope='_splitN_' + str(i))
            layers_split.append(splits)

        c1x1 = Concatenation(layers_split)

        o1 = conv_layer(c1x1, filter=num_1x1_c, kernel=[1, 1], stride=stride)
        o1 = Relu(o1)

        o2 = conv_layer(c1x1, filter=inc, kernel=[1, 1], stride=stride)
        o2 = Relu(o2)

        data_o1 = data_o1 + o1  # 64x64 32x32
        data_o2 = tf.concat([data_o2, o2], 3)

        return [data_o1, data_o2]


    def Dpn_net(self, blur, pan_row, pan_col, ms):
        layers = []

        x = conv_layer(blur, filter=32, kernel=[3, 3], stride=1, layer_name='blur_conv1')
        layers.append(x)  # 0

        x = Relu(x)
        x = conv_layer(x, filter=32, kernel=[3, 3], stride=1, layer_name='blur_conv2')
        layers.append(x) #1

        x = Relu(x)
        x = conv_layer(x, filter=64, kernel=[2, 2], stride=2, layer_name='blur_conv3')
        layers.append(x)  # 2


        pan = tf.concat([pan_row,pan_col], 3)

        x = conv_layer(pan, filter=32, kernel=[3, 3], stride=1, layer_name='blur_conv1')
        layers.append(x)  # 3

        x = Relu(x)
        x = conv_layer(x, filter=32, kernel=[3, 3], stride=1, layer_name='blur_conv2')
        layers.append(x) # 4

        x = Relu(x)
        x = conv_layer(x, filter=64, kernel=[2, 2], stride=2, layer_name='blur_conv3')
        layers.append(x)  # 5

        concat1 = tf.concat([layers[2], layers[5]], 3)

        data = self.DualPathFactory(concat1, 128, 128, 64, 32, '1',_type='proj' )
        for i in range(2,5):
            data = self.DualPathFactory(data, 128, 128, 64, 32, '1',_type='normal' )
        x = Concatenation(data)
        x = self.transition_layer(x, 128, scope='trans_layer_')
        layers.append(x)  # 6

        x = conv_layer(x, filter=64, kernel=[3, 3], stride=2, layer_name='blur_pan_conv0')

        x_ms = conv_layer(ms, filter=32, kernel=[3,3], stride=1, layer_name='ms_conv1')

        x_ms = Relu(x_ms)
        x_ms = conv_layer(x_ms,filter=64, kernel=[3,3], stride=1, layer_name='ms_conv2')
        x = tf.concat([x,x_ms],3) # channel=128

        print("x.shape:")
        print(x.shape)

        x = self.DualPathFactory(x, 128, 128, 64, 32, '1', _type='proj')
        for i in range(2, 9):
            x = self.DualPathFactory(x, 128, 128, 64, 32, '1', _type='normal')
        x = Concatenation(x)
        print('Dual_last:',x.shape)

        x = self.transition_layer(x, 128, scope='trans_layer_')
        layers.append(x)  # 7

        x_ms2 = conv_layer(ms, filter=32, kernel=[3,3], stride=1, layer_name='ms_conv3')

        x_ms2 = Relu(x_ms2)
        x_ms2 = conv_layer(x_ms2,filter=64, kernel=[3,3], stride=1, layer_name='ms_conv4')
        x = tf.concat([x,x_ms2],3)

        # -----------------stride_conv
        x = Relu(x)
        x = strided_conv(x, filter=128, kernel=2, stride=2, layer_name="strided_conv1")
        layers.append(x)  # 8

        concat2 = tf.concat([layers[6], layers[8]], 3)

        x = Relu(concat2)

        x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name='fusion_conv0')

        x = Relu(x)
        x = strided_conv(x, filter=64, kernel=2, stride=2, layer_name="strided_conv2")
        layers.append(x)  # 9

        concat3 = tf.concat([layers[1],layers[4], layers[9]], 3)

        x = Relu(concat3)

        x = conv_layer(x, filter=32, kernel=[3, 3], stride=1, layer_name='fusion_conv1')

        x = Relu(x)
        x = conv_layer(x, filter=4, kernel=3, layer_name='output')
        print(x.shape)

        return x

        # blur ,   ndvi,   ndwi,     pan     ms


def create_model(inputs1, inputs2, inputs3, inputs4, targets, inputs5):

    def getDiff(inputs):
        def getdiff_2():
            A = np.zeros([128, 128])
            for i in range(128):
                for j in range(128):
                    if i == j:
                        A[i][j] = 1
                    elif j == i + 1:
                        A[i][j] = -1
            A[127][0] = -1
            tensor_A = tf.convert_to_tensor(A)
            return tensor_A

        def getdiff_1():
            A = np.zeros([128, 128])
            for i in range(128):
                for j in range(128):
                    if i == j:
                        A[i][j] = 1
                    elif j == i + 1:
                        A[i][j] = -1
            A[127][0] = -1
            tensor_B = tf.convert_to_tensor(A)
            return tensor_B

        tensor_A = getdiff_2()
        tensor_B = getdiff_1()
        tensor_A = tf.expand_dims(tensor_A, axis=0) # 0表示第一维
        tensor_A = tf.cast(tensor_A, tf.float32)
        tensor_B = tf.expand_dims(tensor_B, axis=0)
        tensor_B = tf.cast(tensor_B, tf.float32)
        tensor_A_batch, tensor_B_batch = tf.train.shuffle_batch([tensor_A, tensor_B], batch_size=a.batch_size,
                                                                capacity=200,
                                                                min_after_dequeue=100)

        inputs2_batch = tf.reshape(inputs, [a.batch_size, 1, 128, 128])
        inputs2_h_batch = tf.matmul(tensor_B_batch, inputs2_batch)
        inputs2_v_batch = tf.matmul(inputs2_batch, tensor_A_batch)
        inputs2_h_batch = tf.reshape(inputs2_h_batch, [a.batch_size, 128, 128, 1])
        inputs2_v_batch = tf.reshape(inputs2_v_batch, [a.batch_size, 128, 128, 1])
        return inputs2_h_batch, inputs2_v_batch

    with tf.variable_scope('DUnet'):
        inputs_row, inputs_col = getDiff(inputs4)
        outputs = DpnNet(blur=inputs1, pan_row=inputs_row, pan_col=inputs_col,filters=growth_k, ms=inputs5).model

    with tf.name_scope("DUnet_loss"):
        DUnet_loss_L1 = 100 * tf.reduce_mean(tf.square(targets - outputs))
        DUnet_loss = DUnet_loss_L1

    with tf.name_scope("DUnet_train"):
        DUnet_tvars = [var for var in tf.trainable_variables() if var.name.startswith("DUnet")]
        DUnet_optim = tf.train.AdamOptimizer(a.lr, a.beta1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            DUnet_train = DUnet_optim.minimize(DUnet_loss)


    ema = tf.train.ExponentialMovingAverage(decay=0.99)

    update_losses = ema.apply([DUnet_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        DUnet_loss_L1=ema.average(DUnet_loss_L1),
        DUnet_loss=DUnet_loss,
        DUnet_grads_and_vars=0,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, DUnet_train),
    )


def load_examples():
    if a.mode == 'train':
        filename_queue = tf.train.string_input_producer([a.train_tfrecord])
    elif a.mode == 'test':
        filename_queue = tf.train.string_input_producer([a.test_tfrecord])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'im_mul_raw': tf.FixedLenFeature([], tf.string),
                                           'im_blur_raw': tf.FixedLenFeature([], tf.string),
                                           'im_lr_raw': tf.FixedLenFeature([], tf.string),
                                           'im_ndvi_raw': tf.FixedLenFeature([], tf.string),
                                           'im_ndwi_raw': tf.FixedLenFeature([], tf.string),
                                           'im_pan_raw': tf.FixedLenFeature([], tf.string),

                                       })

    im_mul_1_raw = tf.decode_raw(features['im_mul_raw'], tf.uint8)
    im_mul_1_raw = tf.reshape(im_mul_1_raw, [128, 128, 4])
    im_mul_1_raw = tf.cast(im_mul_1_raw, tf.float32)

    im_lr_1_raw = tf.decode_raw(features['im_lr_raw'], tf.uint8)
    im_lr_1_raw = tf.reshape(im_lr_1_raw, [32, 32, 4])
    im_lr_1_raw = tf.cast(im_lr_1_raw, tf.float32)

    im_ndvi_raw = tf.decode_raw(features['im_ndvi_raw'], tf.uint8)
    im_ndvi_raw = tf.reshape(im_ndvi_raw, [128, 128, 1])
    im_ndvi_raw = tf.cast(im_ndvi_raw, tf.float32)

    im_ndwi_raw = tf.decode_raw(features['im_ndwi_raw'], tf.uint8)
    im_ndwi_raw = tf.reshape(im_ndwi_raw, [128, 128, 1])
    im_ndwi_raw = tf.cast(im_ndwi_raw, tf.float32)

    im_pan_raw = tf.decode_raw(features['im_pan_raw'], tf.uint8)
    im_pan_raw = tf.reshape(im_pan_raw, [128, 128, 1])
    im_pan_raw = tf.cast(im_pan_raw, tf.float32)

    im_blur_1_raw = tf.decode_raw(features['im_blur_raw'], tf.uint8)
    im_blur_1_raw = tf.reshape(im_blur_1_raw, [128, 128, 4])
    im_blur_1_raw = tf.cast(im_blur_1_raw, tf.float32)



    if a.mode == 'train':
        inputs1_batch, inputs2_batch, inputs3_batch, inputs4_batch, targets_batch, inputs5_batch = tf.train.shuffle_batch(
            [im_blur_1_raw, im_ndvi_raw, im_ndwi_raw, im_pan_raw, im_mul_1_raw, im_lr_1_raw],
            batch_size=a.batch_size, capacity=200,
            min_after_dequeue=100)

        steps_per_epoch = int(a.train_count / a.batch_size)

    elif a.mode == 'test':
        inputs1_batch, inputs2_batch, inputs3_batch, inputs4_batch, targets_batch, inputs5_batch = tf.train.batch(
            [im_blur_1_raw, im_ndvi_raw, im_ndwi_raw, im_pan_raw, im_mul_1_raw, im_lr_1_raw],
            batch_size=a.batch_size, capacity=200)

        steps_per_epoch = int(a.test_count / a.batch_size)

    return Examples(
        inputs1=inputs1_batch,
        inputs2=inputs2_batch,
        inputs3=inputs3_batch,
        inputs4=inputs4_batch,

        inputs5=inputs5_batch,

        targets=targets_batch,
        steps_per_epoch=steps_per_epoch,
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for i in range((fetches["outputs_1"].shape[0])):
        name = '%d' % i
        for kind in ["outputs", "targets"]:
            if a.mode == "train":
                filename = "train-" + name + "-" + kind + ".tif"
                if step is not None:
                    filename = "%08d-%s" % (step, filename)
            else:
                name = '%d' % (i + a.batch_size * step)
                filename = "test-" + name + "-" + kind + ".tif"

            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            if kind in ["outputs", "targets"]:
                array2raster(out_path, [0, 0], 128, 128, contents.transpose(2, 0, 1), 4)
            # else:
            #     array2raster(out_path, [0, 0], 128, 128, contents.reshape((128, 128)), 1)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    model = create_model(examples.inputs1, examples.inputs2, examples.inputs3, examples.inputs4, examples.targets, examples.inputs5
                         )
    with tf.name_scope("images"):
        display_fetches = {
            "targets": examples.targets,
            "outputs": model.outputs,
        }
    with tf.name_scope("inputs1_summary"):
        tf.summary.image("inputs1", examples.inputs1)
    with tf.name_scope("inputs2_summary"):
        tf.summary.image("inputs2", examples.inputs2)


    tf.summary.scalar("DUnet_loss_L1", model.DUnet_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config)  as sess:
        print("parameter_count = ", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            max_steps = int(a.test_count / a.batch_size)
            for i in range(max_steps):
                results = sess.run(display_fetches)
                print(results["outputs"].shape)
                save_images(results, i)
        else:
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }
                if should(a.progress_freq):
                    fetches["DUnet_loss_L1"] = model.DUnet_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")

                    save_images(results["display"], step=results["global_step"])

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                        train_epoch, train_step, rate, remaining / 60))
                    print("DUnet_loss_L1", results["DUnet_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break

main()
