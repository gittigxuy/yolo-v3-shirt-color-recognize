# -*- coding:utf-8 -*- 

# import the necessary packages
from keras.models import Model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf

class FashionNet:
	@staticmethod
	def build_category_branch(inputs, numCategories,
		finalAct="softmax", chanDim=-1):
		# utilize a lambda layer to convert the 3 channel input to a
		# grayscale representation
		x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

		# CONV => RELU => POOL
		x = Conv2D(32, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(3, 3))(x)
		x = Dropout(0.25)(x)

		# (CONV => RELU) * 2 => POOL
		x = Conv2D(64, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Conv2D(64, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

		# (CONV => RELU) * 2 => POOL
		x = Conv2D(128, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Conv2D(128, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

		# define a branch of output layers for the number of different
		# clothing categories (i.e., shirts, jeans, dresses, etc.)
		x = Flatten()(x)
		x = Dense(256)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(numCategories)(x)
		x = Activation(finalAct, name="category_output")(x)

		# return the category prediction sub-network
		return x

	@staticmethod
	def build_color_branch_alexnet(inputs, numColors, finalAct="softmax",
		chanDim=-1):
		# CONV => RELU => POOL
		x = Conv2D(16, (3, 3), padding="same")(inputs)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(3, 3))(x)
		x = Dropout(0.25)(x)

		# CONV => RELU => POOL
		x = Conv2D(32, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

		# CONV => RELU => POOL
		x = Conv2D(32, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

		# define a branch of output layers for the number of different
		# colors (i.e., red, black, blue, etc.)
		x = Flatten()(x)
		x = Dense(128)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(numColors)(x)
		x = Activation(finalAct, name="color_output")(x)

		# return the color prediction sub-network
		return x

	@staticmethod
	def build_color_branch_vgg(inputs, numColors, finalAct="softmax",
								   chanDim=-1):
		# Block 1
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(
			inputs)
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

		# Block 2
		x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
			x)
		x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
			x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

		# Block 3
		x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
			x)
		x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
			x)
		x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
			x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

		# Block 4
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
			x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
			x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
			x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

		# Block 5
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
			x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
			x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
			x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

		# Classification block
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = Activation('relu')(x)
		x = Dense(4096)(x)
		x = Activation('relu')(x)
		x = Dense(numColors)(x)
		x = Activation('softmax', name='color_output')(x)
		return x

	@staticmethod
	def build(width, height, numColors,
		finalAct="softmax"):
		# initialize the input shape and channel dimension (this code
		# assumes you are using TensorFlow which utilizes channels
		# last ordering)
		inputShape = (height, width, 3)
		chanDim = -1

		# construct both the "category" and "color" sub-networks
		inputs = Input(shape=inputShape)
		# categoryBranch = FashionNet.build_category_branch(inputs,
		# 	numCategories, finalAct=finalAct, chanDim=chanDim)
		# colorBranch = FashionNet.build_color_branch(inputs,
		# 	numColors, finalAct=finalAct, chanDim=chanDim)


		colorBranch = FashionNet.build_color_branch_vgg(inputs,
			numColors, finalAct=finalAct, chanDim=chanDim)

		# create the model using our input (the batch of images) and
		# two separate outputs -- one for the clothing category
		# branch and another for the color branch, respectively
		model = Model(
			inputs=inputs,
			outputs=[colorBranch],
			name="fashionnet")

		# return the constructed network architecture
		return model
