## Network Arch
```python
        self.forward_layer_one = layers.LSTM(64, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=4) , recurrent_constraint=tf.keras.constraints.MaxNorm(max_value=4), return_sequences=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.backward_layer_one = layers.LSTM(64, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=4) , recurrent_constraint=tf.keras.constraints.MaxNorm(max_value=4), return_sequences=True , go_backwards=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')

        self.noise_one = layers.GaussianNoise(0.5)

        self.drop_one = layers.BatchNormalization()

        self.forward_layer_two = layers.LSTM(32, return_sequences=True)
        self.backward_layer_two = layers.LSTM(32, return_sequences=True, go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')

        self.drop_two = layers.BatchNormalization()

        self.forward_layer_three = layers.LSTM(16,)
        self.backward_layer_three = layers.LSTM(16, go_backwards=True)
        self.bi_three = layers.Bidirectional(self.forward_layer_three, backward_layer=self.backward_layer_three, name='bi_three')

        self.drop_three = layers.BatchNormalization()

        self.flatten_one = layers.Flatten()

        self.dense_four = layers.Dense(16, name='dense_three')
        self.noise_two = layers.GaussianNoise(0.2)
        self.avtivation_four = layers.Activation(tf.nn.relu6, name='dense_four_activation')

        self.drop_four = layers.BatchNormalization()

        self.dense = layers.Dense(2, name='classification') # , kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03)
        self.output_res = layers.Activation(tf.nn.softmax, name='classifi')
```

### 20200923-174819-network-RMSprop

這次使用基於 `20200923-162245-network-RMSprop` 的架構改進，同樣的第一層 BI-LSTM 有增加 `constraint` 進行數據的正規化，替換 dropout 為 `BatchNormalization`。

- Optimizer
    - learning_rate=0.001
    - momentum=0.9
    - decay= 1e-06
    - clipnorm=0.9
- epochs=40
- batch_size=512
- validation_split=0.3

##### 評估
以評估來看 loss 不錯，但驗證集的 loss 慢慢的升高。

```
loss :  0.05775073170661926
tp :  203528.0
fp :  4283.0
tn :  203528.0
fn :  4283.0
acc :  0.0
precision :  0.9793899059295654
recall :  0.9793899059295654
auc :  0.9981651902198792
binary_accuracy :  0.9793899059295654
binary_crossentropy :  0.05775073170661926
```

##### 預測
還可以在改進模型，Precision 部分有提升。
```
TrueNegatives result:  129259.0
TruePositives result:  74269.0
FalseNegatives result:  1595.0
FalsePositives result:  2688.0
Recall result:  0.97897553
Precision result:  0.9650714
```

##### 圖片
![](cross_entropy_graph_decay.png)
![](loss.png)
![](../20200923-104825-network-RMSprop/loss.png) 20200923-104825-network-RMSprop 的 loss
![](precision.png)
![](recall.png)
