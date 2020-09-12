from tensorflow.keras import metrics

def metric():
    metric_list = [
        metrics.Accuracy(name='acc'),
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'), 
        metrics.CategoricalAccuracy(name='categorical_accuracy'),
        metrics.CategoricalCrossentropy(name='categorical_crossentropy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc'),
    ]  
    return metric_list      

