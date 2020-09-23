## Network Arch
```python
        self.forward_layer_one = layers.LSTM(64, return_sequences=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.backward_layer_one = layers.LSTM(64, return_sequences=True , go_backwards=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')

        self.drop_one = layers.GaussianDropout(0.5)

        self.forward_layer_two = layers.LSTM(32 , return_sequences=True)
        self.backward_layer_two = layers.LSTM(32, return_sequences=True, go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')

        self.drop_two = layers.GaussianDropout(0.5)

        self.forward_layer_three = layers.LSTM(16)
        self.backward_layer_three = layers.LSTM(16, go_backwards=True)
        self.bi_three = layers.Bidirectional(self.forward_layer_three, backward_layer=self.backward_layer_three, name='bi_three')

        self.drop_three = layers.GaussianDropout(0.5)

        self.flatten_one = layers.Flatten()

        self.dense_four = layers.Dense(16, name='dense_three')
        self.avtivation_four = layers.Activation(tf.nn.relu6, name='dense_four_activation')

        self.drop_four = layers.GaussianDropout(0.5)

        self.dense = layers.Dense(2, name='classification') # , kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03)
        self.output_res = layers.Activation(tf.nn.softmax, name='classifi')
```

### 20200923-104825-network-RMSprop

在模型方面嘗試將正規化拿掉，直接用 Dropout 方式去實現正規化，效果滿好的。從結果圖的曲線來看，原因有可能是 Dropout 有正規化效果如果再加上 `kernel_regularizer` 這些的話導致 `underfitting`。

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
loss :  0.04121885821223259
tp :  204839.0
fp :  2972.0
tn :  204839.0
fn :  2972.0
acc :  0.3078398108482361
precision :  0.9856985211372375
recall :  0.9856985211372375
auc :  0.9987913370132446
binary_accuracy :  0.9856985211372375
binary_crossentropy :  0.04121885821223259
```

##### 預測
```
TrueNegatives result:  129346.0
TruePositives result:  75493.0
FalseNegatives result:  137.0
FalsePositives result:  2835.0
Recall result:  0.99818856
Precision result:  0.96380603
```

##### 圖片
![](../model_record/20200923-104825-network-RMSprop/cross_entropy_graph_decay.png)
![](../model_record/20200923-104825-network-RMSprop/loss.png)
![](../model_record/20200923-104825-network-RMSprop/precision.png)
![](../model_record/20200923-104825-network-RMSprop/recall.png)
