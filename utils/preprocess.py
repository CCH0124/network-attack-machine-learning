import pandas as pd
import numpy as np
import utils.feature_list as feature
class preprocess():
    __filePath = None
    __labelColumnName = None
    __normalValue = 'BENIGN'
    __dataFrame = None
    __LabelDataFrame = None
    __test_size = None
    def __init__(self, filePath, labelColumnName, test_size=0.3):
        super().__init__()
        self.__filePath = filePath
        self.__labelColumnName = labelColumnName
        self.__test_size = test_size
    
    def get_dataFrame(self):
        self.__dataFrame = pd.read_csv(self.__filePath)
        self.column_trim()
        self.__dataFrame = self.__dataFrame[feature.get_feature()]

    def column_trim(self):
        self.__dataFrame.columns = self.__dataFrame.columns.str.strip()

    def drop_nan(self):
        self.__dataFrame = self.__dataFrame.replace([np.inf, -np.inf], np.nan)
        self.__dataFrame.dropna(inplace=True)
    
    def inf_replace(self):
        replaceValue = 9999999.9
        self.__dataFrame = self.__dataFrame.replace(['Infinity',np.inf, -np.inf], replaceValue)
        self.__dataFrame = self.__dataFrame.fillna(method='ffill')

    def __get_Label(self):
        label_list = self.__getAttackList()
        label = self.__dataFrame.pop(self.__labelColumnName)
        for i in label_list:
            label = np.where((label==i), 1, label)
        label = np.where(label==self.__normalValue, 0, label)
        label = label.astype('int8')
        return label

    def one_hot_encode(self):
        import tensorflow as tf
        self.__LabelDataFrame = tf.one_hot(self.__get_Label(), depth=3)
        
    def __getAttackList(self):
        list_ = list(set((self.__dataFrame[self.__labelColumnName])))
        list_.remove(self.__normalValue)
        return list_

    def normalization(self):
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        self.__dataFrame = scaler.fit_transform(self.__dataFrame)
        
    def splite_train_test(self):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.__dataFrame, self.__LabelDataFrame.numpy(), test_size=self.__test_size)
        return X_train, X_test, y_train, y_test
    
    def splite_train_test_val(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.__dataFrame, self.__LabelDataFrame))
        train_size = int(0.7 * self.__dataFrame.shape[0])
        val_size = int(0.15 * self.__dataFrame.shape[0])
        test_size = int(0.15 * self.__dataFrame.shape[0])

        full_dataset = dataset.shuffle(buffer_size=self.__dataFrame.shape[0], reshuffle_each_iteration = False )
        train_dataset = full_dataset.take(train_size)
        test_dataset = full_dataset.skip(train_size)
        val_dataset = test_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)
        return train_dataset, test_dataset, val_dataset
        
    def preprocessing(self):
        self.get_dataFrame()
        self.drop_nan()
        self.inf_replace()   
        self.one_hot_encode()
        self.normalization()     
        X_train, X_test, y_train, y_test = self.splite_train_test()
        # print(X_train)
        # return self.splite_train_test()
        return X_train, X_test, y_train, y_test
        