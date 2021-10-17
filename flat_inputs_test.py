import sys
from SqnNet import SqnNet
from main_S3DIS_SQN import S3DIS_SQN
from tester_S3DIS_Sqn import ModelTester
from helper_ply import read_ply
# S3DIS's configs
from helper_tool import ConfigS3DIS_Sqn as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot

import tensorflow as tf
"""
open eager execution for debugging; need next to the 'import tensorflow ...' line, otherwise possibly report ValueError: tf.enable_eager_execution must be called at program startup
COMMENT THIS LINE WHEN TRAINING/EVALUATING
"""
# tf.enable_eager_execution()

import numpy as np
import argparse, os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='vis', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    parser.add_argument('--sub_grid_size', type=float, default=0.04, help='grid-sampling size')
    parser.add_argument('--weak_label_ratio', type=float, default=0.01, help='the weakly semantic segmentation ratio')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode

    test_area = FLAGS.test_area

    # override the config with argparse's arguments
    cfg.sub_grid_size=FLAGS.sub_grid_size
    cfg.weak_label_ratio=FLAGS.weak_label_ratio
    # create S3DIS dataset object for weakly semseg using test_area as validation/test set, the rest as training set-yc
    dataset = S3DIS_SQN(test_area, cfg)
    dataset.init_input_pipeline()

    # obtain the dataset iterator's next lement
    flat_inputs = dataset.flat_inputs
    # use inputs(a dict) variable to map the flat_inputs
    with tf.variable_scope('inputs'):
        inputs = dict()
        num_layers = cfg.num_layers

        # correspond to the flat_inputs defined in get_tf_mapping2() in main_S3DIS_SQN.py
        # HACK: for encoder, it needs the original points, so add it to the first element of this array.
        inputs['original_xyz'] = flat_inputs[4 * num_layers] # features containing xyz and feature, (B,N,3+C)
        inputs['xyz'] = (inputs['original_xyz'],) + flat_inputs[:num_layers] # xyz_original plus xyz(points) of sub_pc at all the sub_sampling stages, containing num_layers items
        inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers] # neighbour id, containing num_layers items
        inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers] # sub_sampled idx, containing num_layers items
        inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers] # interpolation idx (nearest idx in the sub_pc for all raw pts), containing num_layers items
        inputs['features'] = flat_inputs[4 * num_layers + 1] # features containing xyz and feature, (B,N,3+C)
        inputs['labels'] = flat_inputs[4 * num_layers + 2]
        inputs['weak_label_masks'] = flat_inputs[4 * num_layers + 3]
        inputs['input_inds'] = flat_inputs[4 * num_layers + 4] # input_inds for each batch 's point in the sub_pc
        inputs['cloud_inds'] = flat_inputs[4 * num_layers + 5] # cloud_inds for each batch

        points = inputs['original_xyz'] # (B,N,3)
        weak_label_masks = inputs['weak_label_masks'] # weak label mask for weakly semseg, (B,N)

        # obtain weakly points, labels, mask
        # BUG: can not select the masked points ,ending up with (n=all points=40960， 3)-> resulting in OOM when performing interplolation.
        # weakly_points = points[weak_label_masks==1,:] # (n,3), each batch might have different number of weak points
        # weakly_points_labels=labels[weak_label_masks==1,:] # (n,)
        # method1 using boolean_mask
        weak_points1 = tf.boolean_mask(points,tf.cast(weak_label_masks,tf.bool))

        # method2 using the gather_nd
        selected_idx = tf.where(tf.equal(weak_label_masks,1)) # (n,2)
        weak_points2 = tf.gather_nd(points, selected_idx)

        is_training = tf.placeholder(tf.bool, shape=())
        training_step = 1
        training_epoch = 0
        correct_prediction = 0
        accuracy = 0
        mIou_list = [0]
        class_weights = DP.get_class_weights(dataset.name)
        Log_file = open('log_train_' + dataset.name + str(dataset.val_split) + '_Sqn.txt', 'a')


    c_proto = tf.ConfigProto()
    c_proto.gpu_options.allow_growth = True
    with tf.Session(config=c_proto) as sess:
        sess.run(tf.global_variables_initializer())
        # use session to start the dataset iterator
        sess.run(dataset.train_init_op)

        # for each batch of training examples, do sth
        while True:
            try:
                # BUG: the ouput should use different names, or it will result in an error like has invalid type <class 'numpy.ndarray'>, must be a string or Tensor.
                flat_inputs, weak_point1, weak_point2, idx= sess.run([dataset.flat_inputs, weak_points1, weak_points2, selected_idx])
                # print(flat_inputs) 
                # print(weak_points, idx)
                print(weak_point1.shape, weak_point2.shape)
                training_step = training_step +1
                if training_step >100:
                    break
            except tf.errors.InvalidArgumentError as e:
                print('error')

            # pc_xyz = flat_inputs[4 * cfg.num_layers] # original xyz
            # sub_pc_xyz = flat_inputs[0] # sub_pc xyz for 1st stage
            # labels = flat_inputs[4 * cfg.num_layers + 2] # sub_pc labels
            # Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :]) # only draw 1st batch's raw PC 
            # Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]]) # draw 1st batch's sub PC