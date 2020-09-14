import tensorflow as tf
from tensorflow.keras import layers
# use elastic net
class MyLstmModel(tf.keras.Model):
    def __init__(self):
        super(MyLstmModel, self).__init__()
        # recurrent_dropout=0.3 
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU#used-in-the-notebooks_1
        self.forward_layer_one = layers.LSTM(64, dropout=0.3, return_sequences=True) # bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-10),activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08,l2=1e-08)
        self.backward_layer_one = layers.LSTM(64, dropout=0.3, return_sequences=True , go_backwards=True)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')

        self.forward_layer_two = layers.LSTM(32, dropout=0.3, return_sequences=True)
        self.backward_layer_two = layers.LSTM(32, dropout=0.3, return_sequences=True , go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')

        self.forward_layer_one = layers.LSTM(16, dropout=0.3) 
        self.backward_layer_one = layers.LSTM(16, dropout=0.3, go_backwards=True)
        self.bi_three = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_three')

        self.dense_three = layers.Dense(8, name='dense_three') # bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-06, l2=1e-10), activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-08, l2=1e-10)
        self.avtivation = layers.Activation(tf.nn.relu, name='dense_three_activation')
        self.drop = layers.Dropout(0.5)

        self.dense = layers.Dense(3, name='classification')
        self.output_res = layers.Activation(tf.nn.sigmoid, name='classifi')
    
    def call(self, inputs, training=None):
        x = self.bi_one(inputs)
        x = self.bi_two(x)
        x = self.bi_three(x)
        x = self.dense_three(x)
        x = self.avtivation(x)
        x = self.drop(x)
        x = self.dense(x)
        x = self.output_res(x)
        return x