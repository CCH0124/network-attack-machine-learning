import tensorflow as tf
from tensorflow.keras import layers
class MyLstmModel(tf.keras.Model):
    def __init__(self):
        super(MyLstmModel, self).__init__()
        self.forward_layer_one = layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10), bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10), dropout=0.3, return_sequences=True)
        self.backward_layer_one = layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10), bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10) , dropout=0.3, return_sequences=True , go_backwards=True)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')

        self.forward_layer_two = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10), bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10), dropout=0.3)
        self.backward_layer_two = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10), bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10) , dropout=0.3 , go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')

        self.forward_layer_three = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10), bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10), dropout=0.3)
        self.backward_layer_three = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10), bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06,l2=1e-08), recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10) , dropout=0.3 , go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_three')

        self.dense_one = layers.Dense(8, kernel_regularizer=tf.keras.regularizers.l2(1e-05), bias_regularizer=tf.keras.regularizers.l2(1e-05), activity_regularizer=tf.keras.regularizers.l2(1e-05), name='dense_one')
        self.avtivation = layers.Activation(tf.nn.relu, name='dense_three_activation')

        self.dense = layers.Dense(3, kernel_regularizer=tf.keras.regularizers.l2(1e-05), bias_regularizer=tf.keras.regularizers.l2(1e-05), activity_regularizer=tf.keras.regularizers.l2(1e-05), name='classification')
        self.output_res = layers.Activation(tf.nn.sigmoid, name='classifi')
    
    
    def call(self, inputs, training=None):
        x = self.bi_one(inputs)
        x = self.bi_two(x)
        x = self.bi_three(x)
        x = self.dense_one(x)
        x = self.classification(x)
        return x