## Network Arch
```python
self.forward_layer_one = layers.LSTM(64, dropout=0.5, return_sequences=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.backward_layer_one = layers.LSTM(64, dropout=0.5, return_sequences=True , go_backwards=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')

        self.forward_layer_two = layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.5 , return_sequences=True)
        self.backward_layer_two = layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.5, return_sequences=True, go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')

        self.forward_layer_three = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.5)
        self.backward_layer_three = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.5, go_backwards=True)
        self.bi_three = layers.Bidirectional(self.forward_layer_three, backward_layer=self.backward_layer_three, name='bi_three')

        self.drop_two = layers.BatchNormalization()

        self.flatten_one = layers.Flatten()

        self.dense_four = layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03) ,name='dense_three')
        self.avtivation_four = layers.Activation(tf.nn.relu6, name='dense_four_activation')

        self.drop_three = layers.BatchNormalization()

        self.dense = layers.Dense(2, name='classification') # , kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03)
        self.output_res = layers.Activation(tf.nn.softmax, name='classifi')
```

### 20200922-220602-network-RMSprop
- Optimizer
    - learning_rate=0.001
    - momentum=0.9
    - decay= 1e-06
- epochs=40
- batch_size=512
- validation_split=0.3

##### 評估
```
loss :  0.6204817295074463
tp :  184882.0
fp :  22929.0
tn :  184882.0
fn :  22929.0
acc :  0.0
precision :  0.8896641731262207
recall :  0.8896641731262207
auc :  0.8753490447998047
binary_accuracy :  0.8896641731262207
binary_crossentropy :  0.6158796548843384
```

##### 預測
```
TrueNegatives result:  132106.0
TruePositives result:  52776.0
FalseNegatives result:  22841.0
FalsePositives result:  88.0
Recall result:  0.6979383
Precision result:  0.99833536
```

##### 圖片
![](../model_record/20200922-220602-network-RMSprop/cross_entropy_graph_decay.png)
![](../model_record/20200922-220602-network-RMSprop/loss.png)
![](../model_record/20200922-220602-network-RMSprop/precision.png)
![](../model_record/20200922-220602-network-RMSprop/recall.png)