import torch
import random
import numpy as np
from collections import defaultdict
import hickle 
from typing import Optional, Tuple

# Huggingface Models
from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification, AutoTokenizer
from transformers import GPT2ForSequenceClassification

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


##########################################################################################################################
##########################################################################################################################

###################            Here are all the tools to build the  AdaptiveLayer Class.               ###################   

##########################################################################################################################
##########################################################################################################################

def outlier_project_bert(layer, in_outliers, out_outliers=False):
    """
    Given a layer, we only select the weights corresponding to outlier dimension subspaces. 
    For now, I am allowing, us to specify differing outlier in the intermediate and output layers
    Tho in practice, they look to be the same. 
    """
    
    # if the out_outliers are not provided, we set them to be identitcal to the in_outliers
    if not out_outliers:
        out_outliers=in_outliers
    
    # Intermediates
    layer.intermediate.dense.weight = torch.nn.Parameter(layer.intermediate.dense.weight[:,in_outliers]) 
    # We only touch this if we perform rank reduction on the weights. 
    #layer.intermediate.dense.bias = layer.intermediate.dense.bias[in_outliers] 

    # Dense
    layer.output.dense.weight = torch.nn.Parameter(layer.output.dense.weight[out_outliers,:]) #torch.nn.Parameter(torch.ones(outliers_shape,3072))
    #layer.output.dense.bias = torch.nn.Parameter(layer.output.dense.bias[out_outliers]) #torch.nn.Parameter(torch.ones(outliers_shape))
    # LayerNorm
    #layer.output.LayerNorm.weight = torch.nn.Parameter(layer.output.LayerNorm.weight[out_outliers])  #torch.nn.Parameter(torch.ones(outliers_shape))
    #layer.output.LayerNorm.bias = torch.nn.Parameter(layer.output.LayerNorm.bias[out_outliers]) #torch.nn.Parameter(torch.ones(outliers_shape))
    #layer.output.LayerNorm.normalized_shape = (out_outliers.shape[0],)

    return None


class DownSampleAttention(torch.nn.Module):
    
    """
    Super simple class to down sample by incides. This may be overkill to create a class for this but ¯\_(ツ)_/¯ 
    """
    
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    # INDEXING OPERATIONS ARE DIFFERENTIABLE. 
    
    def forward(self, x):
        return x[:,:,self.idx]
    
class UpSampleOutput(torch.nn.Module):
    
    """
    Here is where we can do a bunch of experimenting with padding. 
    Right now, I am zero-padding
    But I kinda feel like padding with the attention output will do well. 

    I am holding off on average padding bc it will require addtional memory to store. 
    It's not a TON of memory to do this, but still something to think about. 
    """
    
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        
    # WE can have different padding if we want. 
    def forward(self, x, attention_output):
        # zero padding... for now.
        # Write some code to specify device. Just being lazy now. 
        upsampled_output = torch.zeros(attention_output.shape).to("cuda:0")
        upsampled_output[:,:,self.idx] = x.type(torch.float32)
        
        # this is how we would do attention padding. 
        """
        Two ideas for attention-padding: 

        1. replace attention output with output processed by AdaptiveLayer
        attention_output[:,:,idx] = x
        
        OR

        2. Add the adaptive layer weights to the attention output
        attention_output[:,:,idx] += x
        
        
        return attention_output
        """
        
        return upsampled_output        

class AdaptiveBertLayer(torch.nn.Module):
    # add back config is we want. 
    def __init__(self, layer, idx):
        
        """
        This is just a prototype. Need to clean up. And- hopefully streamline code
        
        Look up config, but we are going to remove it for now. 
        """
        
        
        super().__init__()
        self.chunk_size_feed_forward = 8 #config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # note that we rely on the config from the LAYER. 
        # can change to make it more general. 
        self.attention = layer.attention
        # our function
        self.down_sample_attention = DownSampleAttention(idx)
        self.is_decoder = False #config.is_decoder
        self.add_cross_attention = False #config.add_cross_attention
        # note that we rely on the config from the LAYER. 
        self.intermediate =  layer.intermediate
        # note that we rely on the config from the LAYER. 
        # Set our output dense to be the dense Layer from the layer.
        self.output_dense = layer.output.dense
        
        # Seperate the bias from the weights.
        self.output_bias = self.output_dense.bias

        # Set the bias to None so that we can add our own bias.
        self.output_dense.bias = None

        # Get the layerNorm
        self.output_layerNorm = layer.output.LayerNorm

        # our function
        self.up_sample_output = UpSampleOutput(idx)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]   #add self attentions if we output attention weights
        
        
            
        """
        IDK what the apply_chunking function is, so will need to look closely. 
        But just editing the down-projection and up-projection here.
 
        
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        
        For now we are just manually calling feed forward chunk.
        
        """
        
        layer_output = self.feed_forward_chunk(attention_output)
        
        
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        # Down sample the attention output
        
        down_sample_attention = self.down_sample_attention(attention_output)
        
        # Comput the intermediates
        intermediate_output = self.intermediate(down_sample_attention)

        # Apply the final dense 
        layer_dense_output = self.output_dense(intermediate_output)
        
        # Upsample the output
        up_sample_output = self.up_sample_output(layer_dense_output, attention_output)

        # add the bias
        layer_output = up_sample_output + self.output_bias

        # apply layer norm
        layer_output = self.output_layerNorm(layer_output+attention_output)
        return layer_output
    

#---------------------------------------GPT2----------------------------------------------#

def outlier_project_gpt2(layer, in_outliers, out_outliers=False):
    """
    Given a layer, we only select the weights corresponding to outlier dimension subspaces. 
    For now, I am allowing, us to specify differing outlier in the intermediate and output layers
    Tho in practice, they look to be the same. 
    """
    
    # if the out_outliers are not provided, we set them to be identitcal to the in_outliers
    if not out_outliers:
        out_outliers=in_outliers
    
    # Intermediates
    print("c_fc", layer.c_fc.weight.shape)
    
    layer.c_fc.weight = torch.nn.Parameter(layer.c_fc.weight[in_outliers,:]) 
    print(layer.c_fc.weight.shape)
    #layer.c_fc.bias = torch.nn.Parameter(layer.c_fc.bias[in_outliers])
    # We only touch this if we perform rank reduction on the weights. 
    #layer.c_fc.bias = torch.nn.Parameter(layer.c_fc.bias[in_outliers]) 

    # Dense
    layer.c_proj.weight = torch.nn.Parameter(layer.c_proj.weight[:,out_outliers]) #torch.nn.Parameter(torch.ones(outliers_shape,3072))
    layer.c_proj.bias = torch.nn.Parameter(layer.c_proj.bias[out_outliers]) #torch.nn.Parameter(torch.ones(outliers_shape))
    layer.c_proj.nf = len(out_outliers)

    return None


class DownSampleAttention(torch.nn.Module):
    
    """
    Super simple class to down sample by incides. This may be overkill to create a class for this but ¯\_(ツ)_/¯ 
    """
    
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    # INDEXING OPERATIONS ARE DIFFERENTIABLE. 
    
    def forward(self, x):
        return x[:,:,self.idx]
    
class UpSampleOutput(torch.nn.Module):
    
    """
    Here is where we can do a bunch of experimenting with padding. 
    Right now, I am zero-padding
    But I kinda feel like padding with the attention output will do well. 

    I am holding off on average padding bc it will require addtional memory to store. 
    It's not a TON of memory to do this, but still something to think about. 
    """
    
    def __init__(self, idx, gpu_id):
        super().__init__()
        self.idx = idx
        self.gpu_id = gpu_id
        
    # can have different padding if we want. 
    def forward(self, x, attention_output):
        # zero padding... for now.
        # TODO Write some code to specify device. Just being lazy now.

        # should we get device???
        # otherwise, we may need to do the downsample inside of trainer...  

        upsampled_output = torch.zeros(attention_output.shape).to(self.gpu_id)
        upsampled_output[:,:,self.idx] = x.type(torch.float32)
        
        # this is how we would do attention padding. 
        """
        Two ideas for attention-padding: 

        1. replace attention output with output processed by AdaptiveLayer
        attention_output[:,:,idx] = x
        
        OR

        2. Add the adaptive layer weights to the attention output
        attention_output[:,:,idx] += x
        
        
        return attention_output
        """
        
        return upsampled_output        

class AdaptiveGPT2Layer(torch.nn.Module):
    # add back config is we want. 
    def __init__(self, layer, idx, gpu_id):
        
        """
        This is just a prototype. Need to clean up. And- hopefully streamline code
        
        Look up config, but we are going to remove it for now. 
        """

        print("ADAPTIVE LAYER GPU ID", gpu_id) 
        
        super().__init__()
        #self.gpu_id = gpu_id
        self.chunk_size_feed_forward = 8 #config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # note that we rely on the config from the LAYER. 
        # can change to make it more general. 
        
        # our function
        self.down_sample_attention = DownSampleAttention(idx)

        # note that we rely on the config from the LAYER. 
        self.c_fc = layer.c_fc
        self.act = layer.act
        
        # NOTE:
        # We may want to
        # 1) pass output of self.act to layer.c_proj.weight. 
        # 2) UPSAMPLE
        # 3) ADD BIAS. 
        # GPT-2 is structured in a way that we are passing in a zero-padded tensor (since there is no residual-connection like BERT)
        
        self.c_proj = layer.c_proj
        # our function
        self.up_sample_output = UpSampleOutput(idx, gpu_id)
        

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        outputs = self.down_sample_attention(hidden_states)
        outputs = self.c_fc(outputs)
        outputs = self.act(outputs)
        outputs = self.c_proj(outputs)
        outputs = self.up_sample_output(outputs, hidden_states)
        return outputs