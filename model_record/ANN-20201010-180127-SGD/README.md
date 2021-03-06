特徵部份把欄位只有一個值得欄位刪除。實作在 `data_value` 的 `column_del_one_value` 方法中。
```python
from tensorflow.keras import layers, Input, constraints
from tensorflow.keras import Model
dos_input = Input(shape=(X_train.shape[1],))

dense = layers.Dense(256, name='nn1', kernel_constraint=constraints.MaxNorm(max_value=4))(dos_input)
activation = layers.Activation(tf.nn.relu, name='nn1_relu')(dense)
drop = layers.GaussianDropout(0.5)(dense)
dense = layers.Dense(128, name='nn2')(drop)
activation = layers.Activation(tf.nn.relu, name='nn2_relu')(dense)
noise = layers.GaussianNoise(0.5)(activation)
dense = layers.Dense(64, name='nn3')(noise)
activation = layers.Activation(tf.nn.relu, name='nn3_relu')(dense)
noise = layers.GaussianNoise(0.5)(activation)
dense = layers.Dense(32, name='nn4')(noise)
activation = layers.Activation(tf.nn.relu, name='nn4_relu')(dense)
noise = layers.GaussianNoise(0.5)(activation)
dense = layers.Dense(16, name='nn5')(noise)
activation = layers.Activation(tf.nn.relu, name='nn5_relu')(dense)
dense = layers.Dense(8, name='nn6')(activation)
activation = layers.Activation(tf.nn.relu, name='nn6_relu')(dense)
dense = layers.Dense(2, name='nn7')(activation)
output = layers.Activation(tf.nn.softmax, name='output')(dense)


model = Model(inputs=dos_input, outputs=output)
algorithm = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.99, clipvalue=0.2, nesterov=True, name='SGD')

model.compile(optimizer=algorithm, loss='binary_crossentropy', metrics=modelmetric.metric('binary_crossentropy')) 
```

- epoch=100
- batchsize=512


## evaluate
```
loss :  0.03839242458343506
tp :  204689.0
fp :  2733.0
tn :  204689.0
fn :  2733.0
acc :  0.31808823347091675
precision :  0.9868239760398865
recall :  0.9868239760398865
auc :  0.9989969730377197
binary_accuracy :  0.9868239760398865
binary_crossentropy :  0.03839242458343506
```

## predict
```
TrueNegatives result:  128989.0
TruePositives result:  75700.0
FalseNegatives result:  117.0
FalsePositives result:  2616.0
Recall result:  0.99845684
Precision result:  0.96659684
```

## fig
![](cross_entropy_graph_decay.png)
![](loss.png)
![](precision.png)
![](recall.png)
