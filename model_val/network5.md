## network5
這邊使用兩層 `BI-LSTM` 神經元分別為 `32`、`16` 和兩層神經網路分別是 8 和 3，帶有正規化。

從以下訓練結果來看，`optimizer` 和 `batch_size` 沒有變化，設定如下
- optimizer
    - learning_rate=0.01
    - momentum=0.89
- train
    - batch_size=512

`epoch` 同樣以 10、20 和 40 做觀察，發現越高的 `epoch` 讓 `recall` 從 0.98 調到 0.95，但 `precision` 都是保持 0.95。我們觀察 `loss`，相較於先前它更加擬合，但是相對的還不穩定，後續將基於這個架構繼續開發。

![](../figure/20200916-143949-network5-RMSprop/loss.png)
![](../figure/20200916-144449-network5-RMSprop/loss.png)
![](../figure/20200916-145155-network5-RMSprop/loss.png)

##### 20200916-143949-network5-RMSprop
- epochs=10
        - predict
```
TrueNegatives result:  127750.0
TruePositives result:  74929.0
FalseNegatives result:  852.0
FalsePositives result:  3891.0
Recall result:  0.9887571
Precision result:  0.95063436
```
- fig
    - figure/20200916-143949-network5-RMSprop

##### 20200916-144449-network5-RMSprop
- epochs=20
- predict
```
TrueNegatives result:  127841.0
TruePositives result:  73831.0
FalseNegatives result:  1950.0
FalsePositives result:  3800.0
Recall result:  0.97426796
Precision result:  0.95105046
```
- fig
    - figure/20200916-144449-network5-RMSprop


##### 20200916-145155-network5-RMSprop
- epochs=40
- predict
```
TrueNegatives result:  128344.0
TruePositives result:  72579.0
FalseNegatives result:  3202.0
FalsePositives result:  3297.0
Recall result:  0.9577467
Precision result:  0.9565475
```
- fig
    - figure/20200916-145155-network5-RMSprop
