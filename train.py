#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from scipy import ndimage
from collections import Counter
from collections import namedtuple

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import cPickle as pickle
import tools.readconfig as config
import configuration
import image_attribute_model
import ops.image_process as image_process
import time
from skimage import io,data,img_as_float,transform


OutputMetdata = namedtuple("OutputMetdata",
                           ["image_id", "iamge_data", "caption", "caption_id", "caption_attribute"])

OutputMetdata_new = namedtuple("OutputMetdata_new",
                           ["image_id", "iamge_data","caption_attribute"])

ImageData = []

total_data_index = 1

for i in range(total_data_index):
    data_dir = "/media/user/0005E5D80006DC02/data/coco/paper project/input_data/train-%.5d-of-00005_new.pkl"%i
    fr = open(data_dir)
    Data = pickle.load(fr)
    ImageData.extend(Data)
    fr.close()
    print("Get data form {}".format(data_dir))
    print("Total data are {}.".format(len(ImageData)))



option = config.Read_Config()
model_config = configuration.ModelConfig()
model_config.pretrain_model = option["cnn_model"]
training_config = configuration.TrainingConfig()
train_cnn = True if option["whether_train_model"] == 1 else False

batch_index_list = []
# print(self.config.batch_size/4)
n_example = len(ImageData)
step_length = model_config.batch_size // model_config.num_preprocess_threads
for start, end in zip(range(0, n_example, step_length),
                      range(step_length, n_example, step_length)):
    batch_index_list.append([start, end])


class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):

      # image = tf.image.convert_image_dtype(self._decode_jpeg, dtype=tf.float32)
      # # Resize image.
      # resize_height = 346
      # resize_width =346
      # image = tf.image.resize_images(image,size=[resize_height, resize_width],method=tf.image.ResizeMethod.BILINEAR)
      # image = tf.random_crop(image, [model_config.image_height, model_config.image_width, 3])
      image = self._sess.run(self._decode_jpeg,feed_dict={self._encoded_jpeg: encoded_jpeg})
      assert len(image.shape) == 3
      assert image.shape[2] == 3
      return image



def process_input():
    index = 0
    decoder = ImageDecoder()
    images = []
    lables = []
    # images_shape = []
    while index < len(batch_index_list):
        start, end = batch_index_list[index][0], batch_index_list[index][1]
        batch_data = ImageData[start:end]

        for image in batch_data:
            # images_tensor_length.append(len(image[1]))
            # images.append(image[1])
            encoded_image = decoder.decode_jpeg(image[1])
            images.append(encoded_image)
            lables.append(image[2])
            # images_shape.append([shape[0],shape[1]])

        yield images,lables
        images = []
        lables = []
        index += 1

def process_image(images):
    processed_image = []
    for image in images:
        float_image = img_as_float(image)
        resize_image = transform.resize(float_image,output_shape=[model_config.image_height,model_config.image_width])
        # print(resize_image.shape)
        processed_image.append(resize_image)
    return processed_image



# a = process_input()
# time_tag = time.time()
# b,c = a.next()
# d = process_image(b)
# print(np.asarray(d).shape)
# print(time.time()-time_tag)
#
# time_tag = time.time()
# b,c = a.next()
# d = process_image(b)
# print(np.asarray(d).shape)
# print(time.time()-time_tag)
#
# time_tag = time.time()
# b,c = a.next()
# d = process_image(b)
# print(np.asarray(d).shape)
# print(time.time()-time_tag)


# time_tag = time.time()
# for i in process_input():
#     process_image(i)
#     print("time : {}".format(time.time()-time_tag))
#     time_tag = time.time()




#
# # def process_input():
# #     # batch_data = ImageData[start:end]
# #     decoder = ImageDecoder()
# #     images = []
# #     lables = []
# #     for index, image in enumerate(ImageData):
# #         encoded_image = image[1]
# #         # print(type(encoded_image))
# #         encoded_image = decoder.decode_jpeg(encoded_image)
# #         # print(type(encoded_image))
# #
# #         attribute_groudtruth = image[4]
# #         for thread_id in range(4):
# #             image = image_process.process_image(encoded_image,
# #                                                 is_training=True,
# #                                                 height=model_config.image_height,
# #                                                 width=model_config.image_width,
# #                                                 thread_id=thread_id)
# #             # Enhance_Input.append([image, attribute_groudtruth])
# #
# #             images.append(image)
# #             lables.append(attribute_groudtruth)
# #         if index % 10 == 0 and index != 0:
# #             print("prefetch image -- {} / {}".format(index,len(ImageData)))
# #     with tf.Session() as sess:
# #         images = sess.run(images)
# #     return images,lables
# #
# # images,lables = process_input()
#
#
# def process_input(start,end):
#     batch_data = ImageData[start:end]
#     decoder = ImageDecoder()
#     images = []
#     lables = []
#     for image in batch_data:
#         encoded_image = image[1]
#         # print(type(encoded_image))
#         encoded_image = decoder.decode_jpeg(encoded_image)
#         # print(type(encoded_image))
#
#         attribute_groudtruth = image[4]
#         for thread_id in range(4):
#             image = image_process.process_image(encoded_image,
#                                                 is_training=True,
#                                                 height=model_config.image_height,
#                                                 width=model_config.image_width,
#                                                 thread_id=thread_id)
#             # Enhance_Input.append([image, attribute_groudtruth])
#
#             images.append(image)
#             lables.append(attribute_groudtruth)
#     # print(len(lables))
#     with tf.Session() as sess:
#         images = sess.run(images)
#     return images,lables
#
#
def main(unused_argv):

  # #  batch index的计算
  # n_example = len(ImageData)
  # batch_index_list = []
  # # print(self.config.batch_size/4)
  # step_length = model_config.batch_size // model_config.num_preprocess_threads
  # for start, end in zip(range(0, n_example, step_length),
  #                       range(step_length, n_example, step_length)):
  #     batch_index_list.append([start, end])

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
      model = image_attribute_model.ShowAndTellModel(model_config, mode="train", train_cnn=train_cnn)
      model.build()

      global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

      # global_var_name = []
      global_var = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES)
      # for var in global_var:
      #     global_var_name.append(var.op.name)
      # print("---------global_var-----------")
      # print(len(global_var_name))
      # print(global_var_name)
      #
      train_var_name = []
      train_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
      for var in train_var:
          train_var_name.append(var.op.name)
      print("---------train_var-----------")
      print(len(train_var_name))
      print(train_var_name)

      # step = tf.get_collection(key=tf.GraphKeys.GLOBAL_STEP)
      # print(step)
      # print(step[0].op.name)
      #
      #
      # model_var_name = []
      # model_var = tf.get_collection(key=tf.GraphKeys.MODEL_VARIABLES)
      # for var in model_var:
      #     model_var_name.append(var.op.name)
      # print("---------model_var-----------")
      # print(len(model_var_name))
      # print(model_var_name)
      #
      # slim_model_var_name = []
      # slim_model_var = tf.contrib.slim.get_model_variables()
      # for var in slim_model_var:
      #     slim_model_var_name.append(var.op.name)
      # print("---------slim_model_var-----------")
      # print(len(slim_model_var_name))
      # print(slim_model_var_name)

      # Set up the learning rate.
      with tf.device('/gpu:0'):
          # learning_rate_decay_fn = None
          if train_cnn:
              var_list = tf.get_collection(key=tf.GraphKeys.MODEL_VARIABLES)
              # learning_rate = tf.constant(training_config.train_model_learning_rate)
          else:
              var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
              # learning_rate = tf.constant(training_config.initial_learning_rate)
              print("variable list is {}".format(var_list))

          print(model.total_loss)

          # adam = tf.train.AdamOptimizer()
          # gradient_all = adam.compute_gradients(loss=model.total_loss)
          #
          # grads_vars = [v for (g1,v) in gradient_all if g1 is not None]
          # gradient = adam.compute_gradients(loss=model.total_loss, var_list=grads_vars)

          # print(gradient)

          # grads_holder = [(tf.placeholder(tf.float32,shape=g1.get_shape()), v) for (g1,v) in gradient]
          # print(grads_holder)
          #
          # train_op = adam.apply_gradients(grads_holder)

          # print(train_op)


          train_op = tf.train.AdamOptimizer()\
              .minimize(loss=model.total_loss,global_step=global_step,var_list=var_list,name="BackPropagation")
          # train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(model.total_loss,
          #               global_step=global_step)


  with tf.Session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())

      # Set up the Saver for saving and restoring model checkpoints.
      # print(option["train_dir"])
      # checkpoint = tf.train.latest_checkpoint(option["train_dir"])
      # if checkpoint == None:
      #     # print(option["train_dir"])
      #     # print("yes!!!")
      #     saver = tf.train.Saver(model.cnn_variables)
      #     saver.restore(sess,model_config.pretrain_model)
      #     print('Get variables from {}'.format(model_config.pretrain_model))
      # else:
      #     saver = tf.train.Saver(var_list=model.cnn_variables)
      #     saver.restore(sess,checkpoint)
      #     print('Get variables from {}'.format(option["train_dir"]))
      saver = tf.train.Saver(model.cnn_variables)
      saver.restore(sess, model_config.pretrain_model)
      print('Get variables from {}'.format(model_config.pretrain_model))

      # attribute_variables = tf.get_collection(
      #     tf.GraphKeys.GLOBAL_VARIABLES, scope="attribute_embedding/")
      # name = []
      # for var in attribute_variables:
      #     name.append(var.op.name)
      # print(name)
      saver_attribute = tf.train.Saver(model.attribute_variables)
      saver_attribute.restore(sess,option["attribute_model"])
      print("Get attribute variables from {}".format(option["attribute_model"]))


      saver2 = tf.train.Saver(var_list=global_var,
                              max_to_keep=training_config.max_checkpoints_to_keep)


      # summary
      merged_summary = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter(logdir= option["train_dir"]+'summary/')
      summary_writer.add_graph(sess.graph)
      summary_writer.flush()

      log_step = 10
      summary_step = 100
      save_step = 10000

      epoch = n_example//model_config.batch_size
      timetick = time.time()
      generate = process_input()

      tag = 0
      while  tag < 100:
          for index in range(epoch):
              # images,lables = process_input()
              # print (images)
              # print (lables)
              # print(model.input_images)
              # print(model.input_labels)
              # images_batch = images[batch_index_list[index][0]:batch_index_list[index][1]]
              # lables_batch = lables[batch_index_list[index][0]:batch_index_list[index][1]]

              images = process_image(generate.next()[0])
              lables = generate.next()[1]
              # images_shape = generate.next()[2]
              # print(images_shape)
              # print(np.asarray(images_shape).shape)
              redata = sess.run([global_step,train_op,model.total_loss],
                             feed_dict={model.input_images: np.asarray(images),
                                        model.input_labels: np.asarray(lables)})



              # print(redata)

              index_total = index + tag * epoch
              # print (index)
              if index_total % log_step == 0 and index != 0:
                print("training: iterations--{}/{}, per loss--{}, cost_time--{}/{} s".
                      format(index_total, (epoch*100),redata[2]/32, time.time()-timetick, log_step))
                timetick = time.time()
              #
              # # if index_total % save_step == 0 and index != 0:
              # #   ckpt_file = option['train_dir']+'model/'+ str(index_total)
              # #   saver2.save(sess, ckpt_file)
              # #   print("step {} : save model ".format(index_total))
              #
              # if index_total % summary_step == 0 and index != 0:
              #     summary_writer.add_summary(sess.run(merged_summary,
              #                                         feed_dict={model.input_images: np.asarray(images),
              #                                                    model.input_labels: np.asarray(lables)}),
              #                                global_step = index)
              #     summary_writer.flush()
          tag += 1
          generate = process_input()


      summary_writer.close()

      print ("Done!!!")



if __name__ == "__main__":
  tf.app.run()

