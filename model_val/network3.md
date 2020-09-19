## Network3
兩層 `BI-LSTM`，神經元分別是，32、16，再接神經網路分別是 8 和 3 層，無正規化。

針對不同 `epoch` 分別為 10、20、40 來做比較。發現說 `recall` 在 `epoch` 為 10 時在 0.98，在 `epoch` 為 40 時在 0.97 左右，但 `precision` 也維持在 0.95 左右和前面介紹的神經網路架構沒什麼改變。

同樣的來觀察 `loss`，發現抖動沒先前大，在 `epoch` 為 20 時，以為要擬合了，於是將它調整為 40，但發現並沒有，這讓我傷了一下腦筋，於是又想是不是未擬合，因此在 `network4` 上加了一層 64 個神經元的 `BI-LSTM`。在下面，有嘗試調整 `learning rate`，因為有可能會是局部最小值關係。

![](../figure/20200915-192326-network3-RMSprop/loss.png)
![](../figure/20200915-202658-network3-RMSprop/loss.png)
![](../figure/20200915-211922-network3-RMSprop/loss.png)
##### 20200915-192326-network3-RMSprop
- epochs=40        
- predict
```
TrueNegatives result:  128504.0
TruePositives result:  73336.0
FalseNegatives result:  1973.0
FalsePositives result:  3609.0
Recall result:  0.97380126
Precision result:  0.9530964
```
- fig
    - figure/20200915-192326-network3-RMSprop

##### 20200915-202658-network3-RMSprop
- epochs=20        
- predict
```
TrueNegatives result:  128801.0
TruePositives result:  71983.0
FalseNegatives result:  3326.0
FalsePositives result:  3312.0
Recall result:  0.9558353
Precision result:  0.956013
```
- fig
    - figure/20200915-202658-network3-RMSprop

##### 20200915-211922-network3-RMSprop
- epochs=10
- predict
```
TrueNegatives result:  128153.0
TruePositives result:  74715.0
FalseNegatives result:  865.0
FalsePositives result:  3689.0
Recall result:  0.9885552
Precision result:  0.9529488
```
- fig
    - figure/20200915-211922-network3-RMSprop
## network3 調整 learning rate

下面分別針對 `epoch` 為 10 和 20 來觀察，發現 `recall` 在 `epoch` 為 10 時 `recall` 高達 0.99，在 40 則有 0.98，但是 `precision` 則是調到 0.94。從下面 `loss` 來看調整 `learning rate` 確實有抑制到抖動。

![](../figure/20200915-233149-network3-RMSprop/loss.png)
![](../figure/20200915-234414-network3-RMSprop/loss.png)

##### 20200915-233149-network3-RMSprop
- learning_rate=0.01
- epochs=10
        
- predict
```
TrueNegatives result:  127085.0
TruePositives result:  75189.0
FalseNegatives result:  391.0
FalsePositives result:  4757.0
Recall result:  0.9948267
Precision result:  0.94049734
```
- fig
    - figure/20200915-233149-network3-RMSprop

##### 20200915-234414-network3-RMSprop

- learning_rate=0.01
- epochs=40
        
- predict
```
TrueNegatives result:  127878.0
TruePositives result:  74657.0
FalseNegatives result:  923.0
FalsePositives result:  3964.0
Recall result:  0.9877878
Precision result:  0.9495809
```
- fig
    - figure/20200915-234414-network3-RMSprop
