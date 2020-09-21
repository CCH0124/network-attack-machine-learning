from tensorflow.keras import metrics

def confusion_matrix(y_label, y_class):
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
def eva_metric(y_label, y_class):
    recall = metrics.Recall()
    recall.update_state(y_label, y_class)
    print('Recall result: ', recall.result().numpy())
    pre = metrics.Precision() 
    pre.update_state(y_label, y_class) 
    print('Precision result: ', pre.result().numpy())

def getTrainMetricValue(model, evl_result):
    for name, value in zip(model.metrics_names, evl_result):
        print(name, ': ', value)