#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ops.image_process as image_process
import ops.image_embedding as image_embedding
import numpy as np


class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

class ShowAndTellModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """
  def __init__(self, config, mode, train_cnn=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    self.config = config

    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_cnn = train_cnn

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    # minval maxval bound of range
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    #image word attribute
    self.attribute_groundtruths = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # word attribute vector with shape [batch_size, padded_length, embedding_size].
    self.attribute_embeddings = None

    self.embedding_map = None


    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # Collection of variables from the inception submodel.
    self.cnn_variables = []

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"


  def build_inputs(self):
    # decoder = ImageDecoder()
    # images = []
    # lables = []
    # # 数据增强
    # n_example = len(self.ImageData)
    # batch_tag = []
    # # print(self.config.batch_size/4)
    # for start, end in zip(range(0, n_example, self.config.batch_size//4),
    #                       range(self.config.batch_size//4, n_example, self.config.batch_size//4)):
    #   batch_tag.append([start, end])
    #
    # batch_data = self.ImageData[batch_tag[0][0]:batch_tag[0][1]]
    # for i,image in enumerate(batch_data):
    #   encoded_image = image[1]
    #   # print(type(encoded_image))
    #   encoded_image = decoder.decode_jpeg(encoded_image)
    #   # print(type(encoded_image))
    #
    #   attribute_groudtruth = image[4]
    #   for thread_id in range(4):
    #     image = image_process.process_image(encoded_image,
    #                                         is_training=True,
    #                                         height=self.config.image_height,
    #                                         width=self.config.image_width,
    #                                         thread_id=thread_id)
    #     # Enhance_Input.append([image, attribute_groudtruth])
    #
    #     images.append(image)
    #     lables.append(attribute_groudtruth)
    # with tf.Session() as sess:
    #   item = sess.run(images)

    # queue_capacity = (2 *self.config.batch_size)
    # images, attribute_groundtruths= tf.train.batch_join(
    #   Enhance_Input,
    #   batch_size=self.config.batch_size,
    #   capacity=queue_capacity,
    #   name="batch_and_pad")

    self.input_images = tf.placeholder(dtype=tf.float32,shape=[self.config.step_length,self.config.image_height,self.config.image_width,3],name="input")
    self.input_labels = tf.placeholder(dtype=tf.int32,shape=[self.config.step_length,self.config.word_attribute_size],name="labels")
    # self.input_images_shape = tf.placeholder(dtype=tf.int32,shape=[self.config.step_length,2],name="shape")

    # decoder = ImageDecoder()
    batch_images = []
    batch_labels = []
    for index in range(self.config.step_length):
      encoded_image = self.input_images[index]
      label = self.input_labels[index]
      # encoded_image= decoder.decode_jpeg(image)
      for thread_id in range(self.config.num_preprocess_threads):

        image = image_process.process_image(encoded_image,
                                          is_training=True,
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id)
        batch_images.append(tf.expand_dims(image,0))
        batch_labels.append(tf.expand_dims(label,0))

    batch_images = tf.concat(batch_images,0)
    batch_labels = tf.concat(batch_labels,0)

    self.batch_images = batch_images
    self.batch_labels = batch_labels

    print(self.batch_images)
    print(self.batch_labels)


  # print(type(encoded_image))
  # encoded_image = decoder.decode_jpeg(image[1])
  # print(type(encoded_image))

  # for thread_id in range(4):
  #     image = image_process.process_image(encoded_image,
  #                                         is_training=True,
  #                                         height=model_config.image_height,
  #                                         width=model_config.image_width,
  #                                         thread_id=thread_id)
  #     # Enhance_Input.append([image, attribute_groudtruth])
  #
  #     images.append(image)
  #     lables.append(image[4])


    # self.input_iamges = input_images
    # self.label = input_labels


    # self.images = np.asarray(item)
    # self.attribute_groundtruths = np.asarray(lables)
    # with tf.Session() as sess:
    #   print(sess.run(images))
    #   print(sess.run(attribute_groundtruths))
    # print(self.images)
    # print(self.attribute_groundtruths)

  def build_image_embeddings(self):
    # inception_resnet_output = image_embedding.resnet_inception_v2(self.input_images,
    #                                                               trainable=self.train_cnn,
    #                                                               is_training=self.is_training())
    #
    # self.cnn_variables = tf.get_collection(
    #   tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionResnetV2")

    inception_output = image_embedding.inception_v3(self.batch_images,
                                                    trainable=self.train_cnn,
                                                    is_training=self.is_training())
    self.cnn_variables = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES,scope="InceptionV3")

    # Map inception output_6.1 into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings_1 = tf.contrib.layers.fully_connected(
        inputs=inception_output,
        num_outputs= 1024,
        activation_fn=None,
        weights_initializer=self.initializer,
        scope="image_embeddings_1")
      image_embeddings_2 = tf.contrib.layers.fully_connected(
        inputs=image_embeddings_1,
        num_outputs=self.config.embedding_size,
        activation_fn=None,
        weights_initializer=self.initializer,
        biases_initializer=None,
        scope="image_embeddings_2")

    self.image_embeddings = image_embeddings_2
    print(self.image_embeddings)

    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   print(sess.run(image_embeddings,
    #                  feed_dict={self.input:self.images,self.label:self.attribute_groundtruths}))

  def build_attribute_embeddings(self):
    with tf.variable_scope("attribute_embedding") as scope:
      embedding_map = tf.get_variable(
        name="attribute_map",
        shape=[self.config.word_attribute_size, self.config.embedding_size],
        initializer=self.initializer,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES],
        trainable=False)

      attribute_variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope="attribute_embedding/")
    self.embedding_map = embedding_map
    self.attribute_variables = attribute_variables
    print(self.embedding_map)



  def build_attribute_model(self):

    with tf.variable_scope("build_attribute_model"):
      print(self.image_embeddings)
      print(self.embedding_map)

      prod = tf.matmul(self.image_embeddings, self.embedding_map, transpose_b=True)
      x_len = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.multiply(self.image_embeddings, self.image_embeddings), axis=1)), 0)
      y_len = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.multiply(self.embedding_map, self.embedding_map), axis=1)), 0)

      x_y_len = tf.matmul(x_len, y_len, transpose_a=True)

      self.cosine_dist = prod / x_y_len

      loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.batch_labels, logits=self.cosine_dist)
      total_loss = tf.reduce_sum(loss)
      tf.summary.scalar("loss", loss)
      self.total_loss = total_loss



      # predict = []
      # for x in range(self.config.batch_size):
      #   # feature = tf.slice(self.image_embeddings,[x,0],[1,self.config.embedding_size])
      #   #     vector = tf.ones(shape=[self.config.word_attribute_size,1], dtype=tf.float32)
      #   #     image_embeddings_clone = tf.matmul(vector,feature)
      #   #     # print(image_embeddings_clone)
      #   #     print(self.embedding_map)
      #   #     print(image_embeddings_clone)
      #   #     vector_distance = tf.contrib.losses.cosine_distance(self.embedding_map, image_embeddings_clone, dim=1)
      #   #     print(vector_distance)
      #   #     predict.append(tf.expand_dims(vector_distance,0))
      #   # predict = tf.concat(predict,0)
      #   # print(predict)
      #
      #
      #   feature = tf.nn.embedding_lookup(self.image_embeddings, x)
      #   one = []
      #   for y in range(self.config.word_attribute_size):
      #     word_vector = tf.nn.embedding_lookup(self.embedding_map, y)
      #     # word_vector = tf.expand_dims(word_vector,0)
      #     # print(feature)
      #     # print(word_vector)
      #     mean_feature, variance_feature = tf.nn.moments(x=feature, axes=[0])
      #     mean_word_vector, variance_word_vector = tf.nn.moments(x=word_vector, axes=[0])
      #     normal_feature = tf.nn.batch_normalization(x=feature, mean=mean_feature, variance=variance_feature, offset=0,
      #                                                scale=1,
      #                                                variance_epsilon=0)
      #     normal_word_vector = tf.nn.batch_normalization(x=word_vector, mean=mean_word_vector,
      #                                                    variance=variance_word_vector, offset=0,
      #                                                    scale=1, variance_epsilon=0)
      #     vector_distance = tf.losses.cosine_distance(normal_feature, normal_word_vector, dim=0)
      #     one.append(tf.expand_dims(vector_distance, 0))
      #   print("process: {}".format(x))
      #   predict.append(tf.expand_dims(tf.concat(one, 0), 0))
      # predict = tf.concat(predict, 0)
      #
      # print(predict)
    #
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=self.input_labels, logits=predict)
    # tf.summary.scalar("batch loss", loss)
    # self.total_loss = loss


  # def setup_inception_initializer(self):
  #   """Sets up the function to restore inception variables from checkpoint."""
  #   # Restore inception variables only.
  #   saver = tf.train.Saver(self.inception_variables)
  #
  #   def restore_fn(sess):
  #     tf.logging.info("Restoring Inception variables from checkpoint file %s",
  #                     self.config.pretrain_model)
  #     saver.restore(sess, self.config.pretrain_model)
  #
  #   self.init_fn = restore_fn


  # def setup_global_step(self):
  #   """Sets up the global step Tensor."""
  #   global_step = tf.Variable(
  #       initial_value=0,
  #       name="global_step",
  #       trainable=False,
  #       collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
  #
  #   self.global_step = global_step

  # def bulid_mapping_model(self):
  #
  #   inception_output = image_embedding.inception_v3(self.batch_images,
  #                                                   trainable=self.train_cnn,
  #                                                   is_training=self.is_training())
  #   self.cnn_variables = tf.get_collection(
  #     tf.GraphKeys.GLOBAL_VARIABLES,scope="InceptionV3")
  #   #
  #   # inception_resnet_output = image_embedding.resnet_inception_v2(self.input_images,
  #   #                                                               trainable=self.train_cnn,
  #   #                                                               is_training=self.is_training())
  #   #
  #   # self.cnn_variables = tf.get_collection(
  #   #   tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionResnetV2")
  #
  #   with tf.variable_scope("image_embedding") as scope:
  #     image_embeddings_1 = tf.contrib.layers.fully_connected(
  #       inputs=inception_output,
  #       num_outputs= 1024,
  #       activation_fn=None,
  #       weights_initializer=self.initializer,
  #       scope="image_embeddings_1")
  #     image_embeddings_2 = tf.contrib.layers.fully_connected(
  #       inputs=image_embeddings_1,
  #       num_outputs=self.config.word_attribute_size,
  #       activation_fn=None,
  #       weights_initializer=self.initializer,
  #       biases_initializer=None,
  #       scope="image_embeddings_2")
  #
  #   self.image_embeddings = image_embeddings_2
  #   print(self.image_embeddings)
  #     # tf.summary.histogram("bias", b)
  #     # tf.summary.histogram("image_mapping",image_mapping)
  #
  #   loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.batch_labels, logits=self.image_embeddings )
  #   tf.summary.scalar("batch loss",loss)
  #   # tf.summary.scalar("accuracy", accuracy)
  #   self.total_loss = loss

  def build(self):
    self.build_inputs()
    # self.bulid_mapping_model()
    self.build_image_embeddings()
    self.build_attribute_embeddings()
    self.build_attribute_model()
    # # self.build_attribute_embeddings()
    # # self.build_seq_embeddings()
    # # self.build_model()
    # self.setup_inception_initializer()
    # self.setup_global_step()
