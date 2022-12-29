#==============================================================================
from mlsettings import (
    ModelSettings, 
    ModelParams, 
    Settings, 
    InputRequest, 
    ColumnData, 
    ColumnMeta)

import os

def dump_json(obj, name):
    name += '.json'
    path = os.path.join('example_json',name)
    with open(path, 'w') as f:
        f.write(obj.json(indent = 4))
    print("Dumped json {}".format(path))

#==============================================================================
# a settings file
settings = Settings(debug = False)
dump_json(settings,'settings')
#==============================================================================
# an input request that encodes a dataframe
firstname = ColumnData(
    name="firstname", 
    shape = [2], 
    datatype = 'BYTES', 
    parameters = {'content_type':'str'} ,
    data=["John", "Jane"])

age = ColumnData(
    name="age", 
    shape = [2], 
    datatype = 'INT32', 
    data=[34, 33])

req = InputRequest(parameters = {'content_type':'pd'},inputs = [firstname, age])
dump_json(req,'df_request')
#==============================================================================
# metadata representation
ms_settings = ModelSettings(
    name = 'mymodel', 
    implementation='mlserver_sklearn.SKLearnModel',
    parameters = ModelParams(uri = 'gs://mybucket/mymodel.pkl', version = "v0.1.1")
)
dump_json(ms_settings,'simple_model_settings')
#==============================================================================
# metadata representation more complex

firstname_meta = ColumnMeta(
    name="firstname", 
    datatype = 'BYTES', 
    parameters = {'content_type':'str'})

age_meta = ColumnMeta(
    name="age", 
    datatype = 'INT32', 
    data=[34, 33])

ms_settings = ModelSettings(
    name = 'mymodel', 
    implementation='mlserver_sklearn.SKLearnModel',
    parameters = ModelParams(uri = 'gs://mybucket/mymodel.pkl', version = "v0.1.1"),
    inputs = [firstname_meta, age_meta]
)
dump_json(ms_settings,'complex_model_settings')
#==============================================================================