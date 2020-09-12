import tensorflow as tf
from tensorflow.keras import layers
class MyLstmModel(tf.keras.Model):
    def __init__(self):
        super(MyLstmModel, self).__init__()
        self.forward_layer_one = layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(1e-06), bias_regularizer=tf.keras.regularizers.l2(1e-06),activity_regularizer=tf.keras.regularizers.l2(1e-06), recurrent_regularizer=tf.keras.regularizers.l2(1e-06), dropout=0.3, recurrent_dropout=0.3, return_sequences=True)
        self.backward_layer_one = layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(1e-06), bias_regularizer=tf.keras.regularizers.l2(1e-06), activity_regularizer=tf.keras.regularizers.l2(1e-06), recurrent_regularizer=tf.keras.regularizers.l2(1e-06) , dropout=0.3, recurrent_dropout=0.3, return_sequences=True , go_backwards=True)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')
        
        self.forward_layer_two = layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(1e-06), bias_regularizer=tf.keras.regularizers.l2(1e-06), activity_regularizer=tf.keras.regularizers.l2(1e-06), recurrent_regularizer=tf.keras.regularizers.l2(1e-06), dropout=0.3, recurrent_dropout=0.3, return_sequences=True)
        self.backward_layer_two = layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(1e-06), bias_regularizer=tf.keras.regularizers.l2(1e-06), activity_regularizer=tf.keras.regularizers.l2(1e-06), recurrent_regularizer=tf.keras.regularizers.l2(1e-06), dropout=0.3, recurrent_dropout=0.3, return_sequences=True , go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')
        
        self.forward_layer_three = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(1e-06), bias_regularizer=tf.keras.regularizers.l2(1e-06), activity_regularizer=tf.keras.regularizers.l2(1e-06), recurrent_regularizer=tf.keras.regularizers.l2(1e-06), dropout=0.3, recurrent_dropout=0.3, return_sequences=True)
        self.backward_layer_three = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(1e-06), bias_regularizer=tf.keras.regularizers.l2(1e-06), activity_regularizer=tf.keras.regularizers.l2(1e-06), recurrent_regularizer=tf.keras.regularizers.l2(1e-06), dropout=0.3, recurrent_dropout=0.3, return_sequences=True, go_backwards=True)
        self.bi_three = layers.Bidirectional(self.forward_layer_three, backward_layer=self.backward_layer_three, name='bi_three')
        
        self.forward_layer_four = layers.LSTM(4, kernel_regularizer=tf.keras.regularizers.l2(1e-06), bias_regularizer=tf.keras.regularizers.l2(1e-06), activity_regularizer=tf.keras.regularizers.l2(1e-06), recurrent_regularizer=tf.keras.regularizers.l2(1e-06), dropout=0.1, recurrent_dropout=0.1)
        self.backward_layer_four = layers.LSTM(4, kernel_regularizer=tf.keras.regularizers.l2(1e-06), bias_regularizer=tf.keras.regularizers.l2(1e-06), activity_regularizer=tf.keras.regularizers.l2(1e-06), recurrent_regularizer=tf.keras.regularizers.l2(1e-06), dropout=0.1, recurrent_dropout=0.1, go_backwards=True)
        self.bi_four = layers.Bidirectional(self.forward_layer_four, backward_layer=self.backward_layer_four, name='bi_four')
        
        self.dense = layers.Dense(3, kernel_regularizer=tf.keras.regularizers.l2(1e-03), bias_regularizer=tf.keras.regularizers.l2(1e-03), activity_regularizer=tf.keras.regularizers.l2(1e-03), name='classification')
        self.output_res = layers.Activation(tf.nn.sigmoid, name='classifi')
    
    def call(self, inputs, training=None):
        x = self.bi_one(inputs)
        x = self.bi_two(x)
        x = self.bi_three(x)
        x = self.bi_four(x)
        x = self.dense(x)
        x = self.output_res(x)
        return x