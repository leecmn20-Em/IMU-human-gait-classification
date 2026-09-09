import tensorflow as tf
import pandas as pd
from typing import Any
import numpy as np

class Gait_data:
    # region Attributes
    name:str
    num_classes:int
    sequence_length:int
    raw:pd.DataFrame|None
    time_raw:pd.Series
    time:tf.Tensor
    angle_leg_raw:pd.Series
    angle_leg:tf.Tensor
    angle_back_raw:pd.Series
    angle_back:tf.Tensor
    data_raw:pd.DataFrame
    data:tf.Tensor
    label_raw:pd.Series
    label:tf.Tensor
    input:tf.Tensor
    output:tf.Tensor
    sample_weight:tf.Tensor
    # endregion Attributes
    
    # region Initialization
    def __init__(self,name:str,num_classes:int=6,sequence_length:int=100) -> None:
        self.name=name
        self.num_classes=num_classes
        self.sequence_length=sequence_length
        try:
            self.raw=pd.read_csv(name)
            self.data_process()
        except FileNotFoundError:
            self.raw=None
    # endregion Initialization
    
    # region Processing
    def data_process(self,use_weight:bool=True) -> None:
        self.time_raw=self.raw["time"]
        self.time=self.cast(self.time_raw)
        
        self.angle_leg_raw=self.raw["angle1"]
        self.angle_leg=self.cast(self.angle_leg_raw)
        
        self.angle_back_raw=self.raw["angle2"]
        self.angle_back=self.cast(self.angle_back_raw)
        
        self.data_raw=self.raw.iloc[:,4:16]
        self.data=self.cast(self.data_raw)
        
        self.label_raw=self.raw.iloc[:,16]
        self.label=self.cast(self.label_raw,cast_type=tf.int32)
        
        self.create_sequence()
        
    def create_sequence(self) -> None:
        sequences=[]
        labels=[]
        weights=[]
        
        y_raw=self.label_raw.to_numpy()
        n=int(self.data.shape[0])
        L=self.sequence_length
        
        change_points=np.where(y_raw[1:]!=y_raw[:-1])[0]+1
        
        tau = 20.0      # 감쇠 스케일
        alpha = 2.0     # 최대 추가 가중치
        clip_max = 5.0  # 너무 커지는거 방지

        for i in range(n - L + 1):
            sequences.append(self.data[i:i+L])
            labels.append(self.label[i+L-1])

            j = i + L - 1  # 라벨이 찍히는 시점

            if len(change_points) > 0:
                dists = np.abs(change_points - j)

                # 전이가 여러 개 겹치면 기여를 합산
                contrib = np.exp(-dists / tau).sum()

                w = 1.0 + alpha * contrib
                w = min(w, clip_max)
            else:
                w = 1.0
            weights.append(float(w))
            
        self.input=tf.stack(sequences)
        self.output=tf.stack(labels)
        self.sample_weight=tf.cast(tf.stack(weights),tf.float32)
        
    def cast(self,x:Any,cast_type:Any=tf.float32) -> tf.Tensor:
        return tf.cast(x,cast_type)
    # endregion Processing
    
    # region Visualization
    def summary(self) -> None:
        if self.raw is None:
            print("No raw data available.")
            return
        print("data name: ",self.name)
        print("raw length: ",self.data.shape[0])
        print("sequence length: ",self.sequence_length)
        print("input shape: ",self.input.shape)
        print("output_shape: ",self.output.shape)
    
    def plot_weight(self,period:list[int]|None=None) -> None:
        import matplotlib.pyplot as plt
        if period==None:
            period=[0,len(self.output)]
            
        label_sliced=self.output.numpy()[period[0]:period[1]]
        weight=self.sample_weight.numpy()[period[0]:period[1]]
        
        fig,ax1=plt.subplots(figsize=(12,4))
        
        ax1.plot(weight[period[0]:period[1]],label="weight",color='orange')
        ax1.legend(loc='upper right')
        ax2=ax1.twinx()
        ax2.plot(label_sliced,label="label",linestyle='None',marker='o',color='blue')
        ax2.legend(loc='upper left')
        changes = np.where(np.diff(label_sliced) != 0)[0] + 1 + period[0]
        for xc in changes:
            ax1.axvline(x=xc, color='gray', linestyle='--', alpha=0.3)
        
        plt.show()
    # endregion Visualization
        
def merge_gait_data(data:list[Gait_data],name:str="merged") -> Gait_data:
    merged_input=tf.concat([d.input for d in data],axis=0)
    merged_output=tf.concat([d.output for d in data],axis=0)
    merged_sample_weight=tf.concat([d.sample_weight for d in data],axis=0)
    
    sample=data[0]
    merged=Gait_data(name=name,num_classes=sample.num_classes,sequence_length=sample.sequence_length)
    
    merged.input=merged_input
    merged.output=merged_output
    merged.sample_weight=merged_sample_weight
    
    return merged