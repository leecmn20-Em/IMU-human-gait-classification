import numpy as np

from .gait_data import Gait_data

class GaitDataManager:
    # region Attributes
    num_classes:int
    sequence_length:int
    names:list[str]
    train_names:list[str]
    val_names:list[str]
    test_names:list[str]
    data:list[Gait_data]
    train_data:list[Gait_data]
    val_data:list[Gait_data]
    test_data:list[Gait_data]
    # endregion Attributes
    
    # region Initialization
    def __init__(self,num_classes:int=6,sequence_length:int=100) -> None:
        self.num_classes=num_classes
        self.sequence_length=sequence_length
        
        self.names=[]
        self.train_names=[]
        self.val_names=[]
        self.test_names=[]
        
        self.data=[]
        self.train_data=[]
        self.val_data=[]
        self.test_data=[]
    # endregion Initialization
    
    # region Modifiers
    def add(self,name:str|list[str]) -> None:
        if isinstance(name,str):
            self.names.append(name)
        elif isinstance(name,list):
            self.names.extend(name)
    # endregion Modifiers
    
    # region Processing
    def create_Gait_data(self,raw:list[str]) -> list[Gait_data]:
        data=[Gait_data(name=f+"_labeled.csv",num_classes=self.num_classes,sequence_length=self.sequence_length) for f in raw]
        return data
    
    def split(self,train_ratio:float=0.7,val_ratio:float=0.15,test_ratio:float=0.15,shuffle:bool=True) -> None:
        names=self.names.copy()
        if shuffle:
            np.random.shuffle(names)
        
        total=len(names)
        assert total > 0, "No data available to split"
        train_end=int(total*train_ratio)
        val_end=train_end+int(total*val_ratio)
        
        if total==1:
            self.train_names=names
            self.val_names=[]
            self.test_names=names
        elif total<=5:
            self.train_names=names[:-1]
            self.val_names=[]
            self.test_names=[names[-1]]
        else:
            self.train_names=names[:train_end]
            self.val_names=names[train_end:val_end]
            self.test_names=names[val_end:]
        
    def process(self,pre_split:bool=False) -> None:
        if pre_split:
            self.split()
        
        self.data=self.create_Gait_data(self.names)
        self.train_data=self.create_Gait_data(self.train_names)
        self.val_data=self.create_Gait_data(self.val_names)
        self.test_data=self.create_Gait_data(self.test_names)
    # endregion Processing
        
    # region Visualization
    def summary(self) -> None:
        print(f"{len(self.data)}/{len(self.names)} data created in total")
        print(f"{len(self.train_data)}/{len(self.train_names)} data created for train")
        print(f"{len(self.val_data)}/{len(self.val_names)} data created for validataion")
        print(f"{len(self.test_data)}/{len(self.test_names)} data created for test")
    # endregion Visualization
      