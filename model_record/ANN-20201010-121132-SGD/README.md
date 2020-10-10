特徵部份把欄位只有一個值得欄位刪除。實作在 `data_value` 的 `column_del_one_value` 方法中。
```python
from tensorflow.keras import layers, Input, constraints
from tensorflow.keras import Model
dos_input = Input(shape=(X_train.shape[1],))

dense = layers.Dense(64, kernel_constraint=constraints.MaxNorm(max_value=4), name='nn3')(dos_input)
activation = layers.Activation(tf.nn.relu, name='nn3_relu')(dense)
dense = layers.Dense(32, name='nn4')(activation)
drop = layers.GaussianDropout(0.5)(dense)
activation = layers.Activation(tf.nn.relu, name='nn4_relu')(drop)
dense = layers.Dense(16, name='nn5')(activation)
noise = layers.GaussianNoise(0.5)(dense)
activation = layers.Activation(tf.nn.relu, name='nn5_relu')(noise)
dense = layers.Dense(8, name='nn6')(activation)
noise = layers.GaussianNoise(0.2)(dense)
activation = layers.Activation(tf.nn.relu, name='nn6_relu')(noise)
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
loss :  0.04131507873535156
tp :  204484.0
fp :  2938.0
tn :  204484.0
fn :  2938.0
acc :  0.5797600746154785
precision :  0.985835611820221
recall :  0.985835611820221
auc :  0.9987465739250183
binary_accuracy :  0.985835611820221
binary_crossentropy :  0.04131507873535156
```

## predict
```
TrueNegatives result:  128898.0
TruePositives result:  75586.0
FalseNegatives result:  189.0
FalsePositives result:  2749.0
Recall result:  0.9975058
Precision result:  0.9649071
```

## fig
![](cross_entropy_graph_decay.png)
![](loss.png)
![](precision.png)
![](recall.png)
