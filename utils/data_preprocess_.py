import tensorflow as tf

def splite_train_test_val(self, train_size=0.7, val_size=0.15, test_size=0.15, reshuffle_each_iteration=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.__dataFrame, self.__LabelDataFrame))
        train_size_ = int(train_size * self.__dataFrame.shape[0])
        val_size_ = int(val_size * self.__dataFrame.shape[0])
        test_size_ = int(test_size * self.__dataFrame.shape[0])

        full_dataset = dataset.shuffle(buffer_size=self.__dataFrame.shape[0], reshuffle_each_iteration = reshuffle_each_iteration )
        train_dataset = full_dataset.take(train_size_)
        test_dataset = full_dataset.skip(train_size_)
        val_dataset = test_dataset.skip(val_size_)
        test_dataset = test_dataset.take(test_size_)
        return train_dataset, test_dataset, val_dataset