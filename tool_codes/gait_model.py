import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import tempfile
import os

from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import ClassVar
from dataclasses import dataclass
from io import StringIO

from .gait_data import Gait_data

class Gait_model:
    @dataclass
    class TestResult:
        class_predicted: np.ndarray
        class_original: np.ndarray
        probability_predicted: np.ndarray
        accuracy: float
        length: int
        correct: int
        incorrect: int
        timestamp: str
        
        @classmethod
        def create(cls, class_predicted: np.ndarray, class_original: np.ndarray, probability_predicted: np.ndarray,) -> 'Gait_model.TestResult':
            correct, length, accuracy = measure_accuracy(class_predicted, class_original)
            incorrect = length - correct
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return cls(
                class_predicted=class_predicted,
                class_original=class_original,
                probability_predicted=probability_predicted,
                accuracy=accuracy,
                length=length,
                correct=correct,
                incorrect=incorrect,
                timestamp=timestamp
            )
            
        def show_accuracy(self) -> None:
            print(f"Accuracy: {self.accuracy:.2f}% ({self.correct}/{self.length})")
        
        def plot(self, period:int=10) -> None:
            sampled_plot(self.class_predicted, self.class_original, period=period)
            
    # region Constants
    # 0: sit , 1: standing, 2: stand, 3: sitting, 4: walking, 5: turning
    Class_causality:ClassVar[np.ndarray]=np.array(
        [[1,1,0,0,0,0],
         [0,1,1,1,1,0],
         [0,0,1,1,1,1],
         [1,1,0,1,0,0],
         [0,0,1,1,1,1],
         [0,0,1,1,1,1]],
        dtype=int)
    # endregion Constants
    
    # region Attributes
    model:tf.keras.Model
    record:tf.keras.callbacks.History
    # endregion Attributes
    
    # region Initialization
    def __init__(self,example_data:Gait_data,num_layer:int=100,dropout:float=0.25,learning_rate:float=0.001) -> None:
        num_classes=example_data.num_classes
        input_shape=example_data.input.shape[1:]
        
        self.model=Sequential([
            layers.LSTM(int(num_layer),return_sequences=True,input_shape=input_shape),
            layers.LSTM(int(num_layer/2),return_sequences=True),
            layers.LSTM(int(num_layer/4),return_sequences=False),
            layers.Dropout(dropout),
            layers.Dense(num_classes,activation='softmax')
            ])
        self.compile(learning_rate=learning_rate)
    # endregion Initialization
        
    # region Training
    def compile(self,learning_rate:float=0.001) -> None:
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy'])
    
    def fit(self,train:Gait_data,val:Gait_data,
            epochs:int=30,batch_size:int=128,shuffle:bool=True,
            use_early_stopping:bool=True,patience:int=5,
            use_checkpoint:bool=True, use_weight:bool=False) -> None:
        
        in_train,out_train=train.input,train.output
        in_val,out_val=val.input,val.output
        sample_weight=train.sample_weight
        
        callbacks=[]
        
        if use_early_stopping:
            early_stop=EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
            
        tem_path=None
        if use_checkpoint:
            fd,tem_path=tempfile.mkstemp(suffix=".keras")
            os.close(fd)
            checkpoint=ModelCheckpoint(
                filepath=tem_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=0
            )
            callbacks.append(checkpoint)
        
        if use_weight:
            self.record=self.model.fit(
                in_train,out_train,
                sample_weight=sample_weight,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(in_val,out_val),
                shuffle=shuffle,
                callbacks=callbacks
                )
        else:
            self.record=self.model.fit(
                in_train,out_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(in_val,out_val),
                shuffle=shuffle,
                callbacks=callbacks
                )
        
        if use_checkpoint and tem_path and os.path.exists(tem_path):
            self.model=load_model(tem_path)
            os.remove(tem_path)
    # endregion Training
    
    # region Testing
    def predict(self,data:Gait_data) -> tuple[np.ndarray,np.ndarray]:
        in_test=data.input
        prob=self.model.predict(in_test)
        cls=prob.argmax(axis=1)
        return cls,prob
    
    def post_process(self,cls:np.ndarray) -> np.ndarray:
        new=cls.copy()
        for i in range(1, len(cls)):
            new[i] = self.post_smoothing(cls[i-1], cls[i])
        return new
    
    def post_smoothing(self,past_cls:int,present_cls:int) -> int:
        causality=Gait_model.Class_causality[past_cls, present_cls]
        if causality==1:
            return present_cls
        else:
            return past_cls
    
    def test(self,data:Gait_data,show:bool=False,plot:bool=False,plot_period:int=10,use_postprocess:bool=False, postprocess_kwargs:dict|None=None) -> TestResult:
        cls_pred,prob_pred=self.predict(data)
        cls_data=data.output.numpy()
        
        if use_postprocess:
            from .gaitpostprocessing import OnlinePostProcessor
            
            kw= postprocess_kwargs or {}
            pp=OnlinePostProcessor(causality=Gait_model.Class_causality, **kw)
            cls_pred_pp=np.zeros_like(cls_pred)
            for t in range(len(cls_pred)):
                cls_pred_pp[t]=pp.step(prob_pred[t])
            cls_pred=cls_pred_pp
        
        result=Gait_model.TestResult.create(
            class_predicted=cls_pred,
            class_original=cls_data,
            probability_predicted=prob_pred
        )
        
        if show:
            result.show_accuracy()
        
        if plot:
            result.plot(period=plot_period)
            
        return result
    # endregion Testing

    # region Visualization
    def summary(self,return_string:bool=False) -> None | str:
        if return_string:
            buffer = StringIO()
            self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
            return buffer.getvalue()
        else:
            self.model.summary()
            return None
    # endregion Visualization
    
    # region Management
    def save(self,save_path:str='saved_models',prefix:str='gait_model') -> None:
        timestamp=datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        
        os.makedirs(save_path,exist_ok=True)
        
        save_filename=f"{prefix}_{timestamp}.keras"
        save_full_path=os.path.join(save_path,save_filename)
        
        self.model.save(save_full_path)
        
        print(f"Model saved to {save_full_path}")
        
    def log(self,log_path:str='logs',prefix:str='gait_model') -> None:
        timestamp=datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        
        os.makedirs(log_path,exist_ok=True)
        
        log_filename=f"{prefix}_{timestamp}.log"
        log_full_path=os.path.join(log_path,log_filename)
        
        with open(log_full_path, 'w') as f:
            f.write(str(self.record.history))
        
        print(f"Log saved to {log_full_path}")
        
    def markdown(self,markdown_path:str='markdown',prefix:str='gait_model') -> None:
        timestamp=datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        
        os.makedirs(markdown_path,exist_ok=True)
        
        markdown_filename=f"{prefix}_{timestamp}.md"
        markdown_full_path=os.path.join(markdown_path,markdown_filename)
        
        with open(markdown_full_path, 'w') as f:
            f.write("# Gait Model Summary\n")
            f.write(f"Model Summary:\n{self.summary(return_string=True)}\n")
            f.write(f"Training Parameters:\n")
            f.write(f"Training History:\n{self.record.history}\n")
        
        print(f"Markdown saved to {markdown_full_path}")
    
    @classmethod
    def load(cls,path:str,example_data:Gait_data) -> 'Gait_model':
        obj=cls(example_data=example_data)
        obj.model=load_model(path)
        return obj
    # endregion Management
    
def measure_accuracy(class_predicted:np.ndarray,class_original:np.ndarray) -> tuple[int, int, float]:
        result= class_predicted == class_original
        
        correct=np.sum(result)
        length=len(result)
        accuracy=np.mean(result)*100
        
        return correct, length, accuracy
    
def sampled_plot(pred:np.ndarray,true:np.ndarray,period:int=10) -> None:
        x=np.arange(len(pred))
        x_sampled=x[::period]
        pred_sampled=pred[::period]
        true_sampled=true[::period]
        
        plt.figure(figsize=(12,4))
        plt.plot(x_sampled,true_sampled,'o',label='True',linestyle='None')
        plt.plot(x_sampled,pred_sampled,'x',label='Pred',linestyle='None')
        plt.legend()
        plt.show()