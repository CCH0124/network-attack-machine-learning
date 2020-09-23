## Network Arch
```python
        self.forward_layer_one = layers.LSTM(64, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=4) , recurrent_constraint=tf.keras.constraints.MaxNorm(max_value=4), return_sequences=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.backward_layer_one = layers.LSTM(64, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=4) , recurrent_constraint=tf.keras.constraints.MaxNorm(max_value=4), return_sequences=True , go_backwards=True) # kernel_regularizer=tf.keras.regularizers.l2(10e-06)
        self.bi_one = layers.Bidirectional(self.forward_layer_one, backward_layer=self.backward_layer_one, name='bi_one')

        self.noise_one = layers.GaussianNoise(0.5)

        self.bn_one = layers.BatchNormalization()

        self.forward_layer_two = layers.LSTM(32, return_sequences=True)
        self.backward_layer_two = layers.LSTM(32, return_sequences=True, go_backwards=True)
        self.bi_two = layers.Bidirectional(self.forward_layer_two, backward_layer=self.backward_layer_two, name='bi_two')

        self.bn_two = layers.BatchNormalization()

        self.forward_layer_three = layers.LSTM(16,)
        self.backward_layer_three = layers.LSTM(16, go_backwards=True)
        self.bi_three = layers.Bidirectional(self.forward_layer_three, backward_layer=self.backward_layer_three, name='bi_three')

        self.bn_three = layers.BatchNormalization()

        self.flatten_one = layers.Flatten()

        self.drop_one = layers.GaussianDropout(0.5)

        self.dense_four = layers.Dense(16, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=4), name='dense_three')
        self.noise_two = layers.GaussianNoise(0.5)
        self.avtivation_four = layers.Activation(tf.nn.relu6, name='dense_four_activation')

        self.bn_four = layers.BatchNormalization()

        self.dense = layers.Dense(2, name='classification') # , kernel_regularizer=tf.keras.regularizers.l2(1e-01), activity_regularizer=tf.keras.regularizers.l1(1e-03)
        self.output_res = layers.Activation(tf.nn.softmax, name='classifi')    
```

### 20200923-211443-network-RMSprop

這次使用基於 `20200923-193753-network-RMSprop` 的架構，新增  `self.drop_one = layers.GaussianDropout(0.5)` 這一層，無調整 `learning rate`，`momentum` 調整至 0.95

- Optimizer
    - learning_rate=0.001
    - momentum=0.95
    - decay= 1e-06
    - clipnorm=0.9
- epochs=40
- batch_size=512
- validation_split=0.3

##### 評估
loss 慢慢升高，有可能是新增的 `GaussianDropout` 有所影響。

```
loss :  0.06999453157186508
tp :  202839.0
fp :  4972.0
tn :  202839.0
fn :  4972.0
acc :  0.0
precision :  0.9760743975639343
recall :  0.9760743975639343
auc :  0.9957402944564819
binary_accuracy :  0.9760743975639343
binary_crossentropy :  0.06999453157186508
```

##### 預測

```
TrueNegatives result:  127222.0
TruePositives result:  75617.0
FalseNegatives result:  399.0
FalsePositives result:  4573.0
Recall result:  0.9947511
Precision result:  0.94297296
```

##### 圖片
![](cross_entropy_graph_decay.png)
![](loss.png)
![](precision.png)
![](recall.png)
