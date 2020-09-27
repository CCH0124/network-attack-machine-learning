因為儲存 `model` 問題因此直接使用 Keras 中 Function API 方式建立網路，架構與 `20200923-181126-network-RMSprop` 的架構一致。大致上都沒有改變，只是這邊因為資料前處裡部分有做些正

```python
def inf_and_na_drop(dataframe):
    dataframe = dataframe.replace(['Infinity', np.inf, -np.inf], np.nan)
    return dataframe.dropna(axis=0) # 這邊改為 0 原本是 1
```

- Optimizer
    - learning_rate=0.0129
        - step_decay
    - momentum=0.99
    - clipvalue=0.3
    - decay= 1e-06
- epochs=100
- batch_size=512
- validation_split=0.3

##### 評估

```
loss :  0.040394630283117294
tp :  204341.0
fp :  3081.0
tn :  204341.0
fn :  3081.0
acc :  0.001053408021107316
precision :  0.9851462244987488
recall :  0.9851462244987488
auc :  0.9989821314811707
binary_accuracy :  0.9851462244987488
binary_crossentropy :  0.040394630283117294
```

##### 預測

```
TrueNegatives result:  129259.0
TruePositives result:  75082.0
FalseNegatives result:  149.0
FalsePositives result:  2932.0
Recall result:  0.99801946
Precision result:  0.962417
```

##### 圖片
![](cross_entropy_graph_decay.png)
![](loss.png)
![](precision.png)
![](recall.png)
![](lr_decay.png)
