因為儲存 `model` 問題因此直接使用 Keras 中 Function API 方式建立網路，架構與 `20200923-181126-network-RMSprop` 的架構一致。大致上都沒有改變，只是這邊因為資料前處裡部分有做些正

```python
def inf_and_na_drop(dataframe):
    dataframe = dataframe.replace(['Infinity', np.inf, -np.inf], np.nan)
    return dataframe.dropna(axis=0) # 這邊改為 0 原本是 1
```

- Optimizer
    - learning_rate=0.05
        - step_decay
    - momentum=0.99
    - clipvalue=0.2
    - decay= 1e-06
- epochs=100
- batch_size=512
- validation_split=0.3

##### 評估

```
loss :  0.04151104390621185
tp :  204281.0
fp :  3141.0
tn :  204281.0
fn :  3141.0
acc :  0.017331816256046295
precision :  0.9848569631576538
recall :  0.9848569631576538
auc :  0.9988895058631897
binary_accuracy :  0.9848569631576538
binary_crossentropy :  0.04151104390621185
```

##### 預測

```
TrueNegatives result:  128768.0
TruePositives result:  75513.0
FalseNegatives result:  131.0
FalsePositives result:  3010.0
Recall result:  0.9982682
Precision result:  0.9616673
```

##### 圖片
![](cross_entropy_graph_decay.png)
![](loss.png)
![](precision.png)
![](recall.png)
![](lr_decay.png)
