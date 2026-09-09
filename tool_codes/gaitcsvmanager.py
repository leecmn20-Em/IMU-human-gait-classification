import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect
import pickle
from typing import Optional

class GaitCSVManager:
    # region Attributes
    name: str
    data: pd.DataFrame
    length: int
    boundaries: list[int]
    labels: list[int]
    # endregion Attributes
    
    # region Initialization
    def __init__(self,name:str) -> None:
        self.name=name
        self.data=pd.read_csv(self.name+".csv")
        self.length=len(self.data)
        self.boundaries=[]
        self.labels=[-1]
    # endregion Initialization
    
    # region Modifiers
    def insert_boundary(self,boundary:Optional[int]=None,label:int=-1,show:bool=False) -> None:
        if boundary==None:
            return
        
        if boundary==0:
            self.labels[0]=label
            
        elif boundary in self.boundaries:
            idx=self.boundaries.index(boundary)
            self.labels[idx+1]=label
        
        else:
            idx=bisect.bisect_right(self.boundaries,boundary)
            self.boundaries.insert(idx,boundary)
            self.labels.insert(idx+1,label)
        
        self._validate_boundaries_vs_labels(self.boundaries,self.labels)
            
        if show:
            print(self.boundaries)
            print(self.labels)
    
    def delete_boundary(self,boundary:Optional[int]=None,show:bool=False) -> None:
        if boundary==None:
            return
        
        if not boundary in self.boundaries:
            print(f"Boundary {boundary} does not exist")
            return
        
        idx=self.boundaries.index(boundary)
        self.boundaries.pop(idx)
        self.labels.pop(idx+1)
        
        self._validate_boundaries_vs_labels(self.boundaries,self.labels)
        
        if show:
            print(self.boundaries)
            print(self.labels)
        
    def move_boundary(self,boundary:Optional[int]=None,changed_boundary:Optional[int]=None,changed_label:Optional[int]=None,show:bool=False) -> None:
        if boundary==None:
            return
        if changed_boundary==None:
            changed_boundary=boundary
        
        if not boundary in self.boundaries:
            print(f"Boundary {boundary} does not exist")
            return
        
        idx=self.boundaries.index(boundary)
        if (changed_boundary > self.boundaries[idx+1] if idx+1 < len(self.boundaries) else True) or \
           (changed_boundary < self.boundaries[idx-1] if idx-1 >= 0 else True):
            print(f"Changed boundary {changed_boundary} is out of range")
            return
        if changed_boundary in self.boundaries:
            print(f"Changed boundary {changed_boundary} already exists")
            return
        
        self.boundaries[idx]=changed_boundary
            
        if not changed_label==None:
            self.labels[idx+1]=changed_label
                
        self._validate_boundaries_vs_labels(self.boundaries,self.labels)
        
        if show:
            print(self.boundaries)
            print(self.labels)
        
    def set_boundary(self,boundaries:Optional[list[int]]=None,labels:Optional[list[int]]=None,show:bool=False) -> None:
        if boundaries==None:
            boundaries=[]
            
        if labels==None:
            labels=[-1]
        
        self._validate_boundaries_vs_labels(boundaries,labels)
        
        self.boundaries=boundaries
        self.labels=labels
    
        if show:
            print(self.boundaries)
            print(self.labels)
            
    def remove_labels(self,reset_boundary:bool=False,reset_label:bool=False) -> None:
        self.data.drop(columns="label",errors='ignore',inplace=True)
        if reset_boundary:
            self.boundaries=[]
        if reset_label:
            self.labels=[-1]
    # endregion Modifiers
    
    # region Processing       
    def label_csv_by_boundaries(self,boundaries:Optional[list[int]]=None,labels:Optional[list[int]]=None) -> None:
        if labels==None:
            labels=self.labels
        if boundaries==None:
            boundaries=self.boundaries
        self._validate_boundaries_vs_labels(boundaries,labels)
        
        self.labels=labels
        self.boundaries=boundaries
        
        label=np.zeros(self.length,dtype=int)
        
        if len(boundaries)==0:
            label[:]=-1
        else:
            label[:boundaries[0]]=labels[0]
            for i in range(len(boundaries)-1):
                label[boundaries[i]:boundaries[i+1]]=labels[i+1]
            label[boundaries[-1]:]=labels[-1] 
    
        self.data["label"]=label
    # endregion Processing

    # region Visualization
    def plot(self,period:Optional[list[int]]=None,show_rotation:bool=False,show_labels:bool=True) -> None:
        if period==None:
            period=[0,self.length]
            
        data_sliced=self.data.iloc[period[0]:period[1]]
        
        fig,ax1=plt.subplots(figsize=(4,4))
        
        ax1.plot(data_sliced["angle1"],label="angle1")
        ax1.plot(data_sliced["angle2"],label="angle2")
        if show_labels:
            ax1.legend(loc='upper right')
        if show_rotation:
            ax1.plot(data_sliced["gyroX2"],label="gyroX2")
            if show_labels:
                ax1.legend(loc='upper right')
        if "label" in self.data.columns:
            ax2=ax1.twinx()
            if show_labels:
                ax2.plot(data_sliced["label"],label="label",linestyle='None',marker='o')
            labels = data_sliced["label"].to_numpy()
            changes = np.where(np.diff(labels) != 0)[0] + 1 + period[0]
            for xc in changes:
                ax1.axvline(x=xc, color='gray', linestyle='--', alpha=0.3)
            
        #plt.title(self.name)
        plt.show()
        
    def islabeled(self) -> bool:
        return "label" in self.data.columns
    
    def show_boundaries(self,show_labels:bool=False) -> None:
        print(self.boundaries)
        if show_labels:
            print(self.labels)
    # endregion Visualization
    
    # region Management
    def save_csv(self,name:Optional[str]=None) -> None:
        if name==None:
            name=self.name
        else:
            self.name=name
        
        self.data.to_csv(name+".csv",index=False)
        
    def read_csv(self,name:str) -> None:
        self.data=pd.read_csv(name+".csv")
        self.length=len(self.data)
        
    def save(self,name:Optional[str]=None) -> None:
        if name==None:
            name=self.name
        else:
            self.name=name
            
        with open(name+".pkl",'wb') as f:
            pickle.dump(self,f)
            
    def copy(self) -> 'GaitCSVManager':
        cls=self.__class__
        new=cls(self.name)
        new.__dict__.update(self.__dict__)
        return new
    
    @classmethod
    def load(cls,name:str) -> 'GaitCSVManager':
        with open(name+".pkl",'rb') as f:
            obj=pickle.load(f)
        return obj
    
    def _validate_boundaries_vs_labels(self,boundaries:list[int],labels:list[int]) -> None:
        assert all(isinstance(b, int) for b in boundaries), "boundaries must be integers"
        assert all(b >= 0 for b in boundaries), "boundaries must be non-negative"
        assert all(isinstance(l, int) for l in labels), "labels must be integers"
        assert all(l >= -1 for l in labels), "labels must be -1 or non-negative integers"
        assert len(labels)==len(boundaries)+1, "length of boundaries does not match"
        assert all(boundaries[i] < boundaries[i + 1] for i in range(len(boundaries) - 1)), "boundaries must be unique and in ascending order"
    # endregion Management
    
def gaitCSVLabelGenerator(name:str,boundaries:list[int],labels:list[int],plot:Optional[list[int]]=None) -> None:
    name2=name+"_labeled"
    data=GaitCSVManager(name)
    data.set_boundary(boundaries=boundaries,labels=labels)
    data.label_csv_by_boundaries()
    if plot!=None:
        data.plot(plot,False,False)
    data.show_boundaries()
    data.plot()
    data.save_csv(name2)
    data.save(name2)
    
def gaitCSVshow(name:str,show_rotation:bool=False) -> None:
    name2=name+"_labeled"
    data:GaitCSVManager=GaitCSVManager.load(name2)
    data.plot(show_rotation=show_rotation)