# 資料前處理
流程大致可分做

1. 選擇數據
2. 前處理數據
3. 轉換數據中資料

## 選擇數據
- 取得的數據範圍是多少
- 那些數據是不可用
- 不需要哪些數據來解決問題

## 前處理數據
- 格式化
- 清理
- 採樣
##### 缺失值處理
- 刪除該列或欄位
    - pandas 提供 `dropna`
- 評估缺失值
    - 可能是該屬性的平均值或眾數等
    - pandas 提供 `fillna`

## 轉換數據中資料
- 縮放數據資料
- 分解數據資料
- 聚合數據資料

### 縮放數據資料
##### sklearn 中 normalize
重新將數據縮放為 0 和 1 的範圍

##### sklearn 中 scale
將數據中每個特徵欄位的分佈偏移為平均值為 0 和標準偏差為 1

##### sklearn 中 MinMaxScaler
將數值重新縮放為 0 至 1 範圍，此範圍值可以做設定

##### sklearn 中 StandardScaler
重新縮放值的分佈，使觀測值的平均值為 0，標準偏差為 1

### Normalize 或 Standardize ?
分布為正常的應當使用 `Standardize`，否則就 `Normalize`。

## 特徵選取

特徵選取是指在開發一個機器學習模型時，減少輸入特徵數量的過程。這過程不但能減少計算上的成本，有時還能因為特徵選取減少了聲噪的影響因而建構出一個良好的模型。特徵選擇可分為以下

- Unsupervised
    - 移除多餘的特徵
    - Correlation
- Supervised
    - 移除無關連特徵
    - Wrapper
        - RFE
    - Filter
        - 依照特徵集合和目標的關係選擇特徵集合
        - Statistical 方法
            - SelectKBest
            - SelectPercentil
        - Feature Importance 方法
    - Intrinsic
        - 訓練過程中執行自動特徵選取的演算法
        - Decision Tree
- Dimensionality Reduction
    - 將數據投影到低維度的特徵空間中


### 統計的特徵選取方法
通常在輸入和輸出變量之間使用 `correlation` 統計作為過濾器特徵選擇的基礎。統計量測選擇高度依賴於可變數據類型，如下
- 數值
    - Integer 
    - Floating
- 分類
    - Boolean
    - Ordinal
    - Nominal

從數據類型來看的話數值是屬於 `Regression` 問題，分類是 `Classification` 問題。通常過濾器特徵選擇中使用的統計測量與目標變數一次計算一個輸入變數。因此，它們被稱為單變量統計(univariate statistical)測量。

以下是基於過濾器特徵選擇的單變量統計測量方法
![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/11/How-to-Choose-Feature-Selection-Methods-For-Machine-Learning.png) from https://machinelearningmastery.com/

##### 數值輸入與數值輸出
- Pearson’s correlation coefficient (linear)
- Spearman’s rank coefficient (nonlinear)

##### 數值輸入與分類輸出
- ANOVA correlation coefficient (linear).
- Kendall’s rank coefficient (nonlinear).

>Kendall 假設分類變數是 `Ordinal`

##### 分類輸入與分類輸出
- Chi-Squared test (contingency tables).
- Mutual Information.

### scikit-learn 的特徵選取
在 `scikit-learn` 中提供了許多的[統計測量](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)，如下

- Pearson’s Correlation Coefficient: `f_regression()`
- ANOVA: `f_classif()`
- Chi-Squared: `chi2()`
- Mutual Information: `mutual_info_classif()` and `mutual_info_regression()`