from tensorflow.keras import metrics
class ModelMetric():
    __name = None
    def __init__(self, name):
        super().__init__()
        self.__name = name

    def confusion_matrix_metric(self):
        return [metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn')]

    def confusion_matrix_other_metric(self):
        return [
            metrics.Accuracy(name='acc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(name='auc'),
        ]
    def multi_category(self):
        return [
            metrics.CategoricalAccuracy(name='categorical_accuracy'),
            metrics.CategoricalCrossentropy(name='categorical_crossentropy'),
        ]
    def binary_category(self):
        return [
            metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5),
            metrics.BinaryCrossentropy(name='binary_crossentropy', dtype=None, from_logits=False, label_smoothing=0)
        ]

    def metric(self):
        metrics_ = self.confusion_matrix_metric() + self.confusion_matrix_other_metric()
        if self.__name == 'categorical_crossentropy':
            return metrics_ + self.multi_category()
        if self.__name == 'binary_crossentropy':
            return metrics_ + self.binary_category()
    
    def confusion_matrix(self, y_label, y_class):
        tn = metrics.TrueNegatives()
        tn.update_state(y_label, y_class) 
        print('TrueNegatives result: ', tn.result().numpy())
        tp =metrics.TruePositives()
        tp.update_state(y_label, y_class) 
        print('TruePositives result: ', tp.result().numpy())
        fn = metrics.FalseNegatives()
        fn.update_state(y_label, y_class) 
        print('FalseNegatives result: ', fn.result().numpy())
        fp = metrics.FalsePositives()
        fp.update_state(y_label, y_class) 
        print('FalsePositives result: ', fp.result().numpy())

    def eva_metric(self, y_label, y_class):
        recall = metrics.Recall()
        recall.update_state(y_label, y_class)
        print('Recall result: ', recall.result().numpy())
        pre = metrics.Precision() 
        pre.update_state(y_label, y_class) 
        print('Precision result: ', pre.result().numpy())

    def getTrainMetricValue(self, model, evl_result):
        for name, value in zip(model.metrics_names, evl_result):
            print(name, ': ', value)