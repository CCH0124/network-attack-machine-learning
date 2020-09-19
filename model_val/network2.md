## network2
兩層 `BI-LSTM`，神經元分別是，32、16，再接神經網路分別是 8 和 3 層，同時間也有使用正歸化方式去約束 `weight`、`bias` 等。

從以下訓練結果來看，`optimizer` 和 `batch_size` 沒有變化，設定如下
- optimizer
    - learning_rate=0.001
    - momentum=0.89
- train
    - batch_size=512

針對不同 `epoch` 分別為 10、20、40 來做比較。發現說 `predict` 出來的結果很相近。`recall` 大致上約在 0.95 而 `Precision` 也是相同，但在 `epoch` 為 20 時約為 0.96。分別拿以下 `loss` 來觀察，抖動很大應該是尚為擬合。

![](../figure/20200915-120725-network2-RMSprop/loss.png)
![](../figure/20200915-122026-network2-RMSprop/loss.png)
![](../figure/20200915-124032-network2-RMSprop/loss.png)


從 `network` 和 `network2` 的架構來看，`network` 多了一層 64 個神經元的 `BI_LSTM` 層，結果是相去不遠的。也就是說疊樂多網路層並非是最好的。

##### 20200915-120725-network2-RMSprop
- epochs=10        
- predict
```
TrueNegatives result:  128022.0
TruePositives result:  72722.0
FalseNegatives result:  3040.0
FalsePositives result:  3638.0
Recall result:  0.95987433
Precision result:  0.95235723
```
- fig
    - figure/20200915-120725-network2-RMSprop

##### 20200915-122026-network2-RMSprop
- epochs=20        
- predict
```
TrueNegatives result:  129010.0
TruePositives result:  72241.0
FalseNegatives result:  3254.0
FalsePositives result:  2917.0
Recall result:  0.9568978
Precision result:  0.96118844
```
- fig
    - figure/20200915-122026-network2-RMSprop


##### 20200915-124032-network2-RMSprop
- epochs=40        
- predict
```
TrueNegatives result:  128550.0
TruePositives result:  72411.0
FalseNegatives result:  3084.0
FalsePositives result:  3377.0
Recall result:  0.9591496
Precision result:  0.9554415
```
- fig
    - figure/20200915-124032-network2-RMSprop
