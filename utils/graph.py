import matplotlib.pyplot as plt
import os

def add_dir(name): 
    path = "../figure/"+name
    if not os.path.isdir(path):
        os.mkdir(path)

def loss_graph(model_, name):
    plt.plot(model_.history['loss']) 
    plt.plot(model_.history['val_loss']) 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()
    plt.savefig("../figure/loss.png")
def recall_graph(model_, name):
    plt.plot(model_.history['recall']) 
    plt.plot(model_.history['val_recall']) 
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()
    plt.savefig("../figure/recall.png")
def precision_graph(model_, name):
    plt.plot(model_.history['precision']) 
    plt.plot(model_.history['val_precision']) 
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()
    plt.savefig("../figure/precision.png")
def lr_graph(model_, name):
    plt.plot(model_.history['lr']) 
    plt.title('model Learning Rate')
    plt.ylabel('lr')
    plt.xlabel('epoch')
    plt.legend(['lr'], loc='upper left') 
    plt.show()
    plt.savefig("../figure/"+name+"lr_decay.png")