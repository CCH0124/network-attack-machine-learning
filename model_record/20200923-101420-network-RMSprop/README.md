## Network Arch
```python
self.forward_layer_one = layers.LSTM(64, dropout=0.5, return_sequences=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.backward_layer_one = layers.LSTM(64, dropout=0.5, return_sequences=True , go_backwards=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')

        self.drop_one = layers.GaussianDropout(0.5)

        self.forward_layer_two = layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.5 , return_sequences=True)
        self.backward_layer_two = layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.5, return_sequences=True, go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')

        self.drop_two = layers.GaussianDropout(0.5)

        self.forward_layer_three = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.5)
        self.backward_layer_three = layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(10e-06), dropout=0.5, go_backwards=True)
        self.bi_three = layers.Bidirectional(self.forward_layer_three, backward_layer=self.backward_layer_three, name='bi_three')

        self.drop_three = layers.BatchNormalization()

        self.flatten_one = layers.Flatten()

        self.dense_four = layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03) ,name='dense_three')
        self.avtivation_four = layers.Activation(tf.nn.relu6, name='dense_four_activation')

        self.drop_four = layers.BatchNormalization()

        self.dense = layers.Dense(2, name='classification') # , kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03)
        self.output_res = layers.Activation(tf.nn.softmax, name='classifi')
```

### 20200923-101420-network-RMSprop

在模型方面嘗試將 drop_one、drop_two 從 `BatchNormalization` 變成 `GaussianDropout`

- Optimizer
    - learning_rate=0.001
    - momentum=0.9
    - decay= 1e-06
    - clipnorm=0.9
- epochs=40
- batch_size=512
- validation_split=0.3

##### 評估
以評估來看改善了 loss，結果有比較好。

```
loss :  0.24785716831684113
tp :  187288.0
fp :  20523.0
tn :  187288.0
fn :  20523.0
acc :  0.0
precision :  0.9012420177459717
recall :  0.9012420177459717
auc :  0.9840225577354431
binary_accuracy :  0.9012420177459717
binary_crossentropy :  0.24251492321491241
```

##### 預測
```
TrueNegatives result:  132141.0
TruePositives result:  55147.0
FalseNegatives result:  20149.0
FalsePositives result:  374.0
Recall result:  0.7324028
Precision result:  0.9932638
```

##### 圖片
![](../model_record/20200923-101420-network-RMSprop/cross_entropy_graph_decay.png)
![](../model_record/20200923-101420-network-RMSprop/loss.png)
![](../model_record/20200923-101420-network-RMSprop/precision.png)
![](../model_record/20200923-101420-network-RMSprop/recall.png)
