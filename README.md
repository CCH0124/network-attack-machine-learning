## LSTM
輸入定義為以下
- Samples
    - 一個序列就是一個樣本。一個批次包含一個或多個樣品
- Time Steps
    - 一個時間步是樣本中的一個觀察點
- Features
    - 一個特徵一次觀察一次

- `LSTM` 輸入層必須為 3 維 (Samples, TimeSteps, Feature)
- 在定義時使用 `Input_shape`，其參數是設置兩個值，它們定義了 `Time Steps` 和 `Feature` 數量
- `Samples` 設置為 1 個或更多

本實驗是使用 one-to-one，輸入格式是 (ALL_DATA samples, 1 time step, DATA_FEATURE feature)

## Batch Size
我們都知道深度學習使用 `gradient descent` 來訓練神經網路，其中基於訓練數據集的子集合計算用於更新權重的誤差估計。而梯度的評估使用訓練數據集中的小樣本，可稱為`batch size`，它會影響超參數(hyperparameter)像是 `learning rate`等。方式有以下

- Batch Gradient Descent
    - `Batch Size`設置為訓練數據集中的樣本數量，可以準確估計錯誤，但權重更新之間的時間更長
- Stochastic Gradient Descent
    - `Batch Size`設置為 1，錯誤估計影響很大，原因是經常更新權重
- Minibatch Gradient Descent
    - `Batch Size` 設置為大於 1 且小於訓練樣本數的值，這是 `Batch` 和 `Stochastic Gradient Descent` 之間的權衡
## Drop

它是一個正規化的功能

## Batch Normalization
一種在自動標準化深度學習神經網路中某一層輸入技術。其苦以加速神經網路的訓練過程，在某些情況下適度的正則化效果可以提高模型的性能。在接收上一層的資訊後訓練期間，將追蹤每個輸入變量的統計訊息，並使用它們對數據進行標準化。當中可使用 `Beta` 和 `Gamma` 參數對標準化輸出進行縮放，這些參數定義了轉換輸出新均值和標準差。

在[這個](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)研究中顯示出，在 `activation` 後進行 Batch Normalization 效果會更好。

## GaussianNoise
添加一些聲噪來提升模型的可用性，

## Weight Constraints

減少 Overfitting，可用於全部網路或者標準化輸入資料。

- Maximum norm (max_norm)
    - 強制權重達到或低於給定值
- Non-negative norm (non_neg)
    - 使權重具有正值
- Unit norm (unit_norm)
    - 強制權重的大小為 1.0
- Min-Max norm (min_max_norm)
    - 強制權重在一個範圍內

## Learning Rate

### Learning Rate Schedule
在訓練過程中改變 `Learning Rate`，在學習過程一開始時以較大的權重進行訓練，並在學習過程結束時進行較小的更改或微調。


### Momentum

可防止局部值，在更新過程中增加了*慣性*，從而讓一個方向上的許多過去更新在將來繼續朝該方向發展，具有平滑優化過程，減慢更新速度以繼續前一個方向的作用，而不會卡住或振盪。

### 適應適 Learning Rates
模型在訓練數據集上的性能藉由學習演算法進行監控，且可以相應調整學習率。`AdaGrad`、`RMSProp` 和 `Adam`，都針對模型中的每個權重維持並調整學習率。

### Clipping
給定 loss function、learning rate 甚至目標變量的規模，訓練時神經網路可能變得不穩定。訓練期間權重較大更新會導致數值震盪很大，可能突然衝高或著是負值，這可稱為**exploding gradients**。解決方式可以透過給定選定的`vector norm`以重新縮放梯度或者`clipping`超出範圍的梯度值，這稱為 `gradient clipping`。

會發生梯度爆炸可能是以下原因
- `Learning Rates` 設定不當導致權重更新很大
- 數據前處理選擇不當，導致目標變量差異很大
- `loss function` 選擇不當，導致計算較大的誤差值

在 Keras 中有以下設定
- clipnorm
    - `Gradient` 的值超過設定的值，則值將重新縮放
- clipvalue
    - `Gradient` 小於負閾值或大於正閾值，將梯度值裁減及將損失函數的導數裁減為給定值

## 參考資源

- [drop and regularizers](https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275)
- [kernel-bias-and-activity-regulizers](https://stats.stackexchange.com/questions/383310/what-is-the-difference-between-kernel-bias-and-activity-regulizers-and-when-t)
- [Weight Constraints](https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-neural-networks-with-weight-constraints-in-keras/)