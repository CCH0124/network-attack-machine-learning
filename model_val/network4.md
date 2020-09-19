## Network4
三層 `BI-LSTM`，神經元分別是，64、32、16，再接神經網路分別是 8 和 3 層，無正規化。同樣的 `epoch` 使用 10、20 和 40 觀察。以結果來看 `recall` 大致上約在 0.95 而 `Precision` 也是相同，這和先前一樣。

從以下訓練結果來看，`optimizer` 和 `batch_size` 沒有變化，設定如下

- optimizer
    - learning_rate=0.001
    - momentum=0.89
- train
    - batch_size=512

這邊同樣觀察 `loss`，發現說 `network3` 和 `network4` 在沒正規化下抖動沒很大。但結過還是一樣，不是理想。

![](../figure/20200915-213526-network4-RMSprop/loss.png)
![](../figure/20200915-214749-network4-RMSprop/loss.png)
![](../figure/20200915-224224-network4-RMSprop/loss.png)

##### 20200915-213526-network4-RMSprop
- epochs=10        
- predict
```
TrueNegatives result:  128340.0
TruePositives result:  72456.0
FalseNegatives result:  3124.0
FalsePositives result:  3502.0
Recall result:  0.9586663
Precision result:  0.95389557
```
- fig
    - figure/20200915-213526-network4-RMSprop


##### 20200915-214749-network4-RMSprop
- epochs=20
- predict
```
TrueNegatives result:  128506.0
TruePositives result:  72435.0
FalseNegatives result:  3145.0
FalsePositives result:  3336.0
Recall result:  0.95838845
Precision result:  0.9559726
```
- fig
    - figure/20200915-214749-network4-RMSprop


##### 20200915-224224-network4-RMSprop
- epochs=40
- predict
```
TrueNegatives result:  128419.0
TruePositives result:  72509.0
FalseNegatives result:  3071.0
FalsePositives result:  3423.0
Recall result:  0.9593676
Precision result:  0.9549202
```
- fig
    - figure/20200915-214749-network4-RMSprop
