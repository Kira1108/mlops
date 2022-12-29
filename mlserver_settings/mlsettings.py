from pydantic import BaseModel
from typing import Dict, Any, Optional, List

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



 
