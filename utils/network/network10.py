import tensorflow as tf
from tensorflow.keras import layers
# use elastic net
class MyLstmModel(tf.keras.Model):
    def __init__(self):
        super(MyLstmModel, self).__init__()
        self.forward_layer_one = layers.LSTM(32, dropout=0.5, return_sequences=True) # , kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.backward_layer_one = layers.LSTM(32, dropout=0.3, return_sequences=True , go_backwards=True) # , kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')

        self.forward_layer_two = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.5)
        self.backward_layer_two = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.3, go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')

        self.dense_three = layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03) ,name='dense_three') # bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-03, l2=1e-03), activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-10, l2=1e-10)
        self.avtivation_three = layers.Activation(tf.nn.swish, name='dense_three_activation')
        self.droup_three = layers.GaussianDropout(0.5)

        self.dense_four = layers.Dense(8, kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03) ,name='dense_four') # bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-03, l2=1e-03), activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-10, l2=1e-10)
        self.avtivation_four = layers.Activation(tf.nn.swish, name='dense_four_activation')
        self.droup_four = layers.GaussianDropout(0.5)

        self.dense = layers.Dense(2, name='classification') # , kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03)
        self.output_res = layers.Activation(tf.nn.sigmoid, name='classifi')
    
    def call(self, inputs, training=None):
        x = self.bi_one(inputs)
        x = self.bi_two(x)
        x = self.dense_three(x)
        x = self.avtivation_three(x)
        x = self.droup_three(x)
        x = self.dense_four(x)
        x = self.avtivation_four(x)
        x = self.droup_four(x)
        x = self.dense(x)
        x = self.output_res(x)
        return x