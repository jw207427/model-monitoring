import pandas as pd
from io import StringIO
import json
import logging

def preprocess_handler(inference_record, logger):
    
#     logger.info(f"Inference record: {inference_record}")
    if inference_record.endpoint_input.encoding=='JSON':
        input_data = inference_record.endpoint_input.data
        input_data = json.loads(input_data)
        logger.info(input_data)

        output_data = inference_record.endpoint_output.data
        logger.info(output_data)
        output_data = json.loads(output_data)['res']

        data = output_data + input_data

        data_dict = {}
        col_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Categorical']
        for i, d in enumerate(data):
            if col_names[i] == 'Categorical':
                data_dict[str(i).zfill(10)] = int(d)
            else:
                data_dict[str(i).zfill(10)] = d
    
        logger.info(data_dict)
        
        return data_dict
        
    elif inference_record.endpoint_input.encoding=='CSV':
        input_data = inference_record.endpoint_input.data
        data = input_data.split(',')
        
        output_data = inference_record.endpoint_output.data
        data.insert(0, output_data)

        
        data_dict = {}
        col_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Categorical']
        for i, d in enumerate(data):
            if col_names[i] == 'Categorical':
                data_dict[str(i).zfill(10)] = int(float(d))
            else:
                data_dict[str(i).zfill(10)] = float(d)
        
        logger.info(data_dict)
        return data_dict


    
#     for i in range(len(col_names)):
#         data_dict[col_names[i]] = input_data[0][i]

#     logger.info(f"Data dict: {data_dict}")
#     return(data_dict)

#Share training dataset, share everything