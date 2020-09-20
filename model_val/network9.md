## Network9
優化器參數和訓練參數的值

- optimizer
    - learning_rate=0.01
    - momentum=0.89
- train
    - batch_size=512
##### 20200918-100919-network9-RMSprop
- epoch=40
- predict
```
TrueNegatives result:  127296.0
TruePositives result:  74677.0
FalseNegatives result:  1301.0
FalsePositives result:  4537.0
Recall result:  0.9828766
Precision result:  0.94272476
```
- fig
    - figure/20200918-100919-network9-RMSprop

##### 20200918-104440-network9-RMSprop
- epoch=40
- clipnorm=0.9
- predict
```
TrueNegatives result:  126700.0
TruePositives result:  75304.0
FalseNegatives result:  689.0
FalsePositives result:  5118.0
Recall result:  0.99093336
Precision result:  0.9363607
```
- fig
    - figure/20200918-104440-network9-RMSprop