import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import plot_model

class ModelFigre():
    __HOME_DIR = None
    __NAME = None
    __MODEL = None
    def __init__(self, path, filename, model):
        self.__HOME_DIR = path
        self.__add_dir(self.__HOME_DIR)
        self.__NAME = filename
        self.__MODEL = model

    def __add_dir(self, path): 
        if not os.path.isdir(path):
            os.mkdir(path)

    def loss_graph(self):
        plt.plot(self.__MODEL.history['loss']) 
        plt.plot(self.__MODEL.history['val_loss']) 
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left') 
        plt.savefig(self.__HOME_DIR+"/"+self.__NAME+"/loss.png")
        plt.show()
    def recall_graph(self):
        plt.plot(self.__MODEL.history['recall']) 
        plt.plot(self.__MODEL.history['val_recall']) 
        plt.title('model recall')
        plt.ylabel('recall')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left') 
        plt.savefig(self.__HOME_DIR+"/"+self.__NAME+"/recall.png")
        plt.show()
        
    def precision_graph(self):
        plt.plot(self.__MODEL.history['precision']) 
        plt.plot(self.__MODEL.history['val_precision']) 
        plt.title('model precision')
        plt.ylabel('precision')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left') 
        plt.savefig(self.__HOME_DIR+"/"+self.__NAME+"/precision.png")
        plt.show()
        
    def lr_graph(self):
        plt.plot(self.__MODEL.history['lr']) 
        plt.title('model Learning Rate')
        plt.ylabel('lr')
        plt.xlabel('epoch')
        plt.legend(['lr'], loc='upper left') 
        plt.savefig(self.__HOME_DIR+"/"+self.__NAME+"/lr_decay.png")
        plt.show()
        
    def cross_entropy_graph(self):
        # plot learning curves
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy')
        plt.plot(self.__MODEL.history['loss'], label='train')
        plt.plot(self.__MODEL.history['val_loss'], label='val')
        plt.legend()
        plt.savefig(self.__HOME_DIR+"/"+self.__NAME+"/cross_entropy_graph_decay.png")
        plt.show()

    def save_model(self):
        dot_img_file = self.__HOME_DIR+"/"+self.__NAME+"/model.png"
        plot_model(self.__MODEL, to_file=dot_img_file, rankdir='LR', show_shapes=True)