
import tensorflow as tf
class preprocess():
    __LABEL_COLUMN = None
    __UNWANTED_COLS = None
    __CSV_COLUMNS = None
    __DEFAULTS = None
    def __init(self, LABEL_COLUMN, UNWANTED_COLS, CSV_COLUMNS, DEFAULTS):
        self.__LABEL_COLUMN = LABEL_COLUMN
        self.__UNWANTED_COLS = UNWANTED_COLS
        self.__CSV_COLUMNS = CSV_COLUMNS
        self.__DEFAULTS = DEFAULTS


    def features_and_labels(self, row_data):
    # The .pop() method will return item and drop from frame. 
        label = row_data.pop(self.LABEL_COLUMN)
        features = row_data
        
        for unwanted_col in self.UNWANTED_COLS:
            features.pop(unwanted_col)

        return features, label

    def create_dataset(self, pattern, batch_size=1, mode='eval'):
        dataset = tf.data.experimental.make_csv_dataset(
            pattern, batch_size, self.CSV_COLUMNS, self.DEFAULTS)

    # The map() function executes a specified function for each item in an iterable.
    # The item is sent to the function as a parameter.
        dataset = dataset.map(features_and_labels)

        if mode == 'train':
    # The shuffle() method takes a sequence (list, string, or tuple) and reorganize the order of the items.
            dataset = dataset.shuffle(buffer_size=1000).repeat()

        # take advantage of multi-threading; 1=AUTOTUNE
        dataset = dataset.prefetch(1)
        return dataset