import pandas as pd
from typing import ClassVar

class GaitCSVList:
    # region Constants
    Normal:ClassVar[str] = 'normal'
    Abnormal:ClassVar[str] = 'abnormal'
    Abnormality:ClassVar[str] = 'abnormality'
    
    Columns:ClassVar[list[str]]=[Normal, Abnormal, Abnormality]
    Valued_Columns:ClassVar[list[str]]=[Normal, Abnormal]
    # endregion Constants
    
    # region Attributes
    name:str
    lists:pd.DataFrame
    normal:list[str]
    abnormal:list[str]
    abnormality:list[str]
    # endregion Attributes
    
    # region Initialization
    def __init__(self,name:str) -> None:
        self.name=name
        
        try:
            self.lists=pd.read_csv(self.name+".csv")
        except FileNotFoundError:
            self.lists=pd.DataFrame(columns=GaitCSVList.Columns)
        
        self.normal=self.get_normal()
        self.abnormal=self.get_abnormal()
        self.abnormality=self.get_abnormality()
        assert len(self.abnormal)==len(self.abnormality), "abnormality missed"
    # endregion Initialization
    
    # region Accessors
    def get_normal(self) -> list[str]:
        return self.lists[GaitCSVList.Normal].dropna().to_list()
    
    def get_abnormal(self) -> list[str]:
        return self.lists[GaitCSVList.Abnormal].dropna().to_list()
    
    def get_abnormality(self) -> list[str]:
        return self.lists[GaitCSVList.Abnormality].dropna().to_list()
    # endregion Accessors
    
    # region Modifiers
    def add_normal(self,name:str) -> None:
        self.normal.append(name)
        self.normal.sort()
        self._rebuild_list()
        
    def add_abnormal(self,name:str,abnormality:str="unknown") -> None:
        self.abnormal.append(name)
        self.abnormality.append(abnormality)
        if self.abnormal:
            ab,abty=zip(*sorted(zip(self.abnormal,self.abnormality)))
            self.abnormal=list(ab)
            self.abnormality=list(abty)
        self._rebuild_list()
        
    def remove_normal(self,name:str) -> None:
        try:
            idx=self.normal.index(name)
            self.normal.pop(idx)
            self.normal.sort()
            self._rebuild_list()
        except ValueError:
            print(f"{name} not exists in normals")
    
    def remove_abnormal(self,name:str) -> None:
        try:
            idx=self.abnormal.index(name)
            self.abnormal.pop(idx)
            self.abnormality.pop(idx)
            if self.abnormal:
                ab,abty=zip(*sorted(zip(self.abnormal,self.abnormality)))
                self.abnormal=list(ab)
                self.abnormality=list(abty)
            self._rebuild_list()
        except ValueError:
            print(f"{name} not exists in abnormals")
    # endregion Modifiers
    
    # region Processing
    def _rebuild_list(self) -> None:
        max_len=max(len(self.normal),len(self.abnormal))
        pad=lambda lst: lst+[None]*(max_len - len(lst))
        
        data = {
            GaitCSVList.Normal: pad(self.normal),
            GaitCSVList.Abnormal: pad(self.abnormal),
            GaitCSVList.Abnormality: pad(self.abnormality),
        }
        
        self.lists = pd.DataFrame(data)
        self.lists.dropna(how='all', inplace=True)
        self.lists.reset_index(drop=True, inplace=True)
        
        if not self._check():
            print("Data has corrupted. Rolling back to saved data")
            self.__init__(self.name)
            return
        
        self.lists.to_csv(self.name+".csv",index=False)
        
    def _check(self) -> bool:
        flag=True
        errors=[]
        
        if len(self.abnormal)!=len(self.abnormality):
            flag=False
            errors.append("abnormality missed")
            
        unexpected_cols=set(self.lists.columns)-set(GaitCSVList.Columns)
        if unexpected_cols:
            flag=False
            errors.append("unexpected column exists")
            
        empty_rows=self.lists[self.lists.isnull().all(axis=1)]
        if not empty_rows.empty:
            flag=False
            errors.append("empty row exists")
            
        for col in GaitCSVList.Valued_Columns:
            col_data = self.lists[col].dropna()
            duplicated = col_data[col_data.duplicated()].unique()
            if len(duplicated) > 0:
                flag = False
                errors.append(f"{duplicated.tolist()} duplicated in {col}")
                
        sets = [set(self.lists[col].dropna()) for col in GaitCSVList.Valued_Columns]
        overlap = set.intersection(*sets)
        if len(overlap) > 0:
            flag = False
            errors.append(f"{list(overlap)} shared across columns")
            
        for err in errors:
            print(err)
        return flag
    # endregion Processing
            
    # region Visualization
    def summary(self) -> None:
        print(f"{len(self.normal)} normal data found")
        print(f"{len(self.abnormal)} abnormal data found")
    # endregion Visualization