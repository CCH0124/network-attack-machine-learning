## Network Arch
```python
        self.forward_layer_one = layers.LSTM(64, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3) , recurrent_constraint=tf.keras.constraints.MaxNorm(max_value=3.5), dropout=0.5, return_sequences=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.backward_layer_one = layers.LSTM(64, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3) , recurrent_constraint=tf.keras.constraints.MaxNorm(max_value=3.5), dropout=0.5, return_sequences=True , go_backwards=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')

        self.drop_one = layers.GaussianDropout(0.5)

        self.forward_layer_two = layers.LSTM(32, return_sequences=True)
        self.backward_layer_two = layers.LSTM(32, return_sequences=True, go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')

        self.drop_two = layers.GaussianDropout(0.5)

        self.forward_layer_three = layers.LSTM(16,)
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

### 20200923-142954-network-RMSprop

在模型方面嘗試只有第一層 LSTM 增加 kernel_constraint、和 dropout，同樣的有 `GaussianDropout`。

- Optimizer
    - learning_rate=0.001
    - momentum=0.9
    - decay= 1e-06
    - clipnorm=0.9
- epochs=40
- batch_size=512
- validation_split=0.3

##### 評估
以評估來看 loss 不錯，但感覺在驗證集的 loss 有點升高。

```
loss :  0.0851297676563263
tp :  200727.0
fp :  7084.0
tn :  200727.0
fn :  7084.0
acc :  0.1904735565185547
precision :  0.965911328792572
recall :  0.965911328792572
auc :  0.9956796169281006
binary_accuracy :  0.965911328792572
binary_crossentropy :  0.0851297676563263
```

##### 預測
```
TrueNegatives result:  128266.0
TruePositives result:  72461.0
FalseNegatives result:  3334.0
FalsePositives result:  3750.0
Recall result:  0.9560129
Precision result:  0.9507945
```

##### 圖片
![](cross_entropy_graph_decay.png)
![](loss.png)
![](precision.png)
![](recall.png)
