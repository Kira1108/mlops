from dataclasses import dataclass
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from mlserver.codecs import PandasCodec
from abc import ABC, abstractmethod
from pathlib import Path
import logging

log = logging.getLogger("model-settings")

class Settings(BaseModel):
    """Settings for the MLServer Settings Service."""
    debug:bool = True
    
class ColumnData(BaseModel):
    """Column Data is a json representation of dataframe column"""
    name:str
    datatype:str = None
    parameters:Optional[Dict[str, Any]] = None
    shape:Optional[List[int]] = None
    data:List = None
    
class ColumnMeta(BaseModel):
    """Column Meta is a description of dataframe column"""
    name:str
    datatype:str = None
    parameters:Optional[Dict[str, Any]] = None
    shape:Optional[List[int]] = [-1]

class ModelParams(BaseModel):
    """Used in model-settings"""
    uri:str
    version:str = "v0.1.0"
    
class ModelSettings(BaseModel):
    """A model settings includes a model name, implementation method, 
        parameters(model file location etc.) and input formats."""
    name:str
    implementation:str
    parameters:ModelParams
    inputs:List[ColumnMeta] = None

class InputRequest(BaseModel):
    """Tell me the content type of the entire input, and give me the real request data"""
    content_type:Optional[str] = None
    parameters:Optional[dict] = None
    inputs:List[ColumnData]


class SettingsCreator(ABC):
    """Abstract class for creating model settings object"""

    @abstractmethod
    def create_model_settings(self) -> ModelSettings:
        pass

    def create_settings(self) -> Settings:
        return Settings(debug = True)

    def dump_json(self, path = "."):
        settings_path = Path(path) / "settings.json"
        model_settings_path = Path(path) / "model-settings.json"
        
        with open(settings_path, 'w') as f:
            f.write(self.create_settings().json(indent = 2))
            log.info(f"Write settings to {settings_path}")

        with open(model_settings_path, 'w') as f:
            f.write(self.create_model_settings().json(indent = 2))
            log.info(f"Write model settings to {model_settings_path}")


@dataclass
class SklearnModelSettings(SettingsCreator):

    name:str
    uri:str
    df:pd.DataFrame
    version:str = "v0.1.0"
    implementation:str = 'mlserver_sklearn.SKLearnModel'

    def create_inputs_meta(self):
        """Create metadata input for dataframe"""
        request = PandasCodec.encode_request(self.df.head(1))
        inputs = []
        for r in request.inputs:
            col_meta = {}
            if r.datatype == 'BYTES':
                col_meta['parameters'] = {'content_type':'str'}
            col_meta['name'] = r.name
            col_meta['datatype'] = r.datatype
            inputs.append(col_meta)
        return inputs

    def create_model_settings(self) -> ModelSettings:
        inputs = self.create_inputs_meta()
        model_settings = ModelSettings(
            name = self.name,
            implementation = self.implementation,
            parameters = ModelParams(uri = self.uri, version = self.version),
            inputs = inputs
        )
        return model_settings
    
def create_request(df,n = 3, dump = True):
    req = PandasCodec.encode_request(df.head(n))
    req.parameters = {'content_type':"pd"}
    j = req.json(include = {'parameters','inputs','id'}, indent = 2, exclude_none = True)
    if dump:
        with open("sample_request.json",'w') as f:
            f.write(j)
    return j
        


    



 
