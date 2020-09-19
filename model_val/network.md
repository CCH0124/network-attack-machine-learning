## network
三層 `BI-LSTM`，神經元分別是，64、32、16，再接神經網路分別是 8 和 3 層，同時間也有使用正歸化方式去約束 `weight`、`bias` 等。

從以下訓練結果來看，`optimizer` 和 `batch_size` 沒有變化，設定如下

- optimizer
    - learning_rate=0.001
    - momentum=0.89
- train
    - batch_size=512

針對不同 `epoch` 分別為 10、20、40 來做比較。發現說 `predict` 出來的結果很相近。`recall` 大致上約在 0.95 而 `Precision` 也是相同。分別拿以下 `loss` 來觀察，抖動很大應該是尚為擬合。

![](../figure/20200915-111015-network-RMSprop/loss.png)
![](../figure/20200915-113013-network-RMSprop/loss.png)
![](../figure/20200915-114805-network-RMSprop/loss.png)

##### 20200915-111015-network-RMSprop

- epochs=10      
- predict
```
TrueNegatives result:  128082.0
TruePositives result:  72690.0
FalseNegatives result:  3072.0
FalsePositives result:  3578.0
Recall result:  0.959452
Precision result:  0.9530865
```
- fig
    - figure/20200915-111015-network-RMSprop

##### 20200915-113013-network-RMSprop
- epochs=20       
- predict
```
TrueNegatives result:  128204.0
TruePositives result:  72579.0
FalseNegatives result:  3183.0
FalsePositives result:  3456.0
Recall result:  0.95798683
Precision result:  0.9545472
```
- fig
    - figure/20200915-113013-network-RMSprop

##### 20200915-114805-network-RMSprop
- epochs=40
- predict
```
TrueNegatives result:  128172.0
TruePositives result:  72715.0
FalseNegatives result:  3047.0
FalsePositives result:  3488.0
Recall result:  0.95978194
Precision result:  0.9542275
```
- fig
    - figure/20200915-114805-network-RMSprop
