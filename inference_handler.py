import joblib
import os
from io import StringIO
import json
from sagemaker_containers.beta.framework import (
    encoders,
    worker,
)

# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def input_fn(input_data, content_type):
    
    print(input_data)
    
    if content_type == "application/json":
        
        data = json.loads(input_data)
        
    elif content_type == 'text/csv':

        data = input_data.split(',')
    else:
        raise RuntimeException("{} content type is not supported by this script.".format(accept))
    
    return data

def predict_fn(input_data, model):

    res = model.predict([input_data])

    return res
    
    
def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":

        json_output = {"res": prediction.tolist()}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        print(prediction)
        return worker.Response(encoders.encode(prediction, accept).strip("\n"), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))