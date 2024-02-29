import torch
import random
import numpy as np
from collections import defaultdict
import hickle 

# Huggingface Models
from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification, AutoTokenizer
from transformers import GPT2ForSequenceClassification

def outlier_project_bert(layer, in_outliers, out_outliers, pad=False):
    """
    Given a layer, we only select the weights corresponding to outlier dimension subspaces. 
    For now, I am allowing, us to specify differing outlier in the intermediate and output layers
    Tho in practice, they look to be the same. 

    TODO: Idk if we can do this, I woul like to do this as "pad layer".
    """
      
    # Intermediates
    layer.intermediate.dense.weight = torch.nn.Parameter(layer.intermediate.dense.weight[:,in_outliers]) 
    # We only touch this if we perform rank reduction on the weights. 
    #layer.intermediate.dense.bias = layer.intermediate.dense.bias[in_outliers] 

    # Dense
    layer.output.dense.weight = torch.nn.Parameter(layer.output.dense.weight[out_outliers,:]) #torch.nn.Parameter(torch.ones(outliers_shape,3072))
    layer.output.dense.bias = torch.nn.Parameter(layer.output.dense.bias[out_outliers]) #torch.nn.Parameter(torch.ones(outliers_shape))
    # LayerNorm
    layer.output.LayerNorm.weight = torch.nn.Parameter(layer.output.LayerNorm.weight[out_outliers])  #torch.nn.Parameter(torch.ones(outliers_shape))
    layer.output.LayerNorm.bias = torch.nn.Parameter(layer.output.LayerNorm.bias[out_outliers]) #torch.nn.Parameter(torch.ones(outliers_shape))
    layer.output.LayerNorm.normalized_shape = (out_outliers.shape[0],)
    
    #TODO fix this. 
    if pad:
        # This is how we pad the output tensor.
        # BUT I want this to be a LAYER in the network. 
        og_out_shape = layer.output.dense.out_features
        padding = (0, og_out_shape - out_outliers.shape[0])
        # Pad the tensor with ones
        padded_tensor = torch.nn.functional.pad(original_tensor, padding, mode='constant', value=1)

    return None

def get_ci_new(config, data, model, max_points, gpu_id=None):
    """Given the data and model of interest, generate a sample of size max_points,
    then calculate the covariance matrix."""       
    # Send model to gpu if available
    if gpu_id: 
        device = gpu_id 
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        
    num_points = 0 
    # need to automate this... 
    num_layers=12
    points_list = {i: [] for i in range(1, num_layers)}
    model.eval() 
    model.to(device)
    # main EVAL loop 
    for idx, batch in enumerate(data):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()} 
    
        # Set model to eval and run input batches with no_grad to disable gradient calculations   
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True) 
        
        # Collect the last state representations to a list and keep track of the number of points    
        #for layer in range(1, len(outputs.hidden_states)):
        # Token embeddings. 
        points = torch.reshape(outputs.hidden_states[-1], (-1,768))
        print("about to detatch")
        points = points.detach().cpu().numpy()  
        print("Detached")
        points_list[0].append(points)   

        num_points += points.shape[0]
        
        if num_points > max_points:
            break

    # Stack the points and calclate the sample covariance C0 
    sample = np.vstack(points_list)
    # can compute the variance directly here. 
    # var = np.var(sample, axis=1)
    
    var_dict = defaultdict(int)
    for layer, points in points_list.items():
        var_dict[layer] = np.var(points, axis=1)


    hickle.dump(config.model_name + "_" + config.task + "_var.hickle")

    #C0 = np.cov(sample.T)
    return var_dict

def get_ci(config, data, model, max_points, gpu_id=None):
    """Given the data and model of interest, generate a sample of size max_points,
    then calculate the covariance matrix. Run this as a warmup to generate a stable
    covariance matrix for IsoScore Regularization"""       
    # Send model to gpu if available
    if gpu_id: 
        device = gpu_id 
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        
    num_points = 0 
    points_list = []
    model.eval()

    model.to(device)
    print("IS MODEL CUDA", next(model.parameters()).is_cuda)
    # main EVAL loop 
    for idx, batch in enumerate(data):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()} 
       

        # Set model to eval and run input batches with no_grad to disable gradient calculations   
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True) 
            if config.layer == "all": 
                points = torch.reshape(torch.stack(outputs.hidden_states)[1:,:,:,:], (-1,768))
            else: 
                points = torch.reshape(outputs.hidden_states[config.layer], (-1,768))
        num_points += points.shape[0]
       # Collect the last state representations to a list and keep track of the number of points    
        points = points.detach().cpu().numpy()  
        points_list.append(points)   
        if num_points > max_points:
            print("NUM POINTS", num_points) 
            break
    # Stack the points and calclate the sample covariance C0 
    sample = np.vstack(points_list)
    var = np.var(sample, axis=0) 
    hickle.dump(var, config.model_name + "_" +  str(config.layer) + "_" + config.task + "_var.hickle") 
    return var