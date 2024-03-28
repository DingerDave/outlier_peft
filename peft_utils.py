import torch
import random
import numpy as np
from collections import defaultdict
import hickle 
from typing import Optional, Tuple

# Huggingface Models
from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification, AutoTokenizer
from transformers import GPT2ForSequenceClassification
import torch.nn as nn

"""
class IntermediateOutputExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Optional[list] = None):
        #Wrapper Class for extracting statistics from intermediate layers of a model.
        super().__init__()
        self.model = model
        
        # WILLIAM COMMENTS 
        # TODO: ALSO, we may want to have a functionality where, we can do this projection providing only indices of outliers. 
        # ^^^ Further, we may want to do this so different layers have different down-projections. 
        # That way we don't have to run save states every time and we can run the exp for random indices. 
        
        if layers is None:
            self.layers = [module for module in self.model.named_modules()]
        else:
            modules = dict([*self.model.named_modules()])
            
            self.layers = [(name, modules[name]) for name in layers]

        self.running_raw_values = {layer[0]: None for layer in self.layers}
        self.layer_output_std = {}
        self.layer_output_means = {}

        for layer_id, module in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

        
    def save_outputs_hook(self, layer_id: str) -> Callable:

        def fn(_, __, output): 
            # runs at each hooked layer
            if self.running_raw_values[layer_id] is None:
                # set the outputs
                self.running_raw_values[layer_id] = output.detach().cpu().numpy()
            else:
                # concatenate the outputs
                 
                
                #WILLIAM COMMENT
                # Changed from dim=0 to dim=1 to stack along the "num_tokens" dimension in the tensor. 
                # We can't use any torch functions here.     
                self.running_raw_values[layer_id] = np.concatenate([self.running_raw_values[layer_id], output.detach().cpu().numpy()], axis=1)
            
        return fn

    def forward(self, input: dict) -> Dict[str, list]:
        # move inputs to the correct device
        input = {key: value.to(next(self.model.parameters()).device) for key, value in input.items()}
        _ = self.model(**input)
        return self.running_raw_values

    def save_statistics(self, file_path = None) -> None:
        
        for name, values in self.running_raw_values.items():

            if type(values) == torch.Tensor:
                # If our output is a tensor, we can just detach it and move it to the cpu
                #   Most outputs are just tensors, for instance the output of a DenseLayer
                if len(values.shape) == 3:
                    # If we haven't dealt with the squence length, we just take the classification token
                    values = values.detach().cpu().numpy()[:,0,:]
                else:
                    # WILLIAM COMMENTS
                    #^^^ To deal with sequence length, we can just reshape to -1, shape.[2] 
                    values = values.detach().cpu().numpy()
                
            else:
                # If our output is a tuple of tensors, we need to concatenate them and then detach them
                values = np.reshape(values, (-1, values.shape[-1]))
                #values = np.concatenate([outputs.last_hidden_state.detach().numpy()[:, 0, :] for outputs in values])
                
            # Compute the mean and std of the outputs across all examples    
            std = np.std(values, axis=0).astype(float)
            means = np.mean(values, axis=0).astype(float)

            # Match each mean and std to its corresponding dimension, one indexed 
            # TODO: CHANGE TO ZERO INDEXED
            ranked_std = list(sorted(zip(list(range(1, len(std)+1)), std), key=lambda x: x[1], reverse=True))
            ranked_means = list(sorted(zip(list(range(1, len(means)+1)), means), key=lambda x: x[1], reverse=True)) 

            # We store all these stats in a class dictionary for access later
            self.layer_output_std.update({name: ranked_std})
            self.layer_output_means.update({name: ranked_means})
        
        if file_path is not None:
            # In case we want to save and view these later
            with open(file_path, "w") as f:
                json_able = {name[0] : list(self.layer_output_std[name[0]]) for name in list(self.layers)}
                f.write(json.dumps(json_able))

    def down_sample_layer(self, layer_name, n=8) -> None:
        
        # Get the layers with the highest variance
        top_layers_idx = [layer[0] for layer in list(self.layer_output_std[layer_name])[:n]]
        layer = dict([*self.model.named_modules()])[layer_name]
        mean_vals = {int(row[0]): row[1] for row in self.layer_output_means[layer_name]}
        
        # for each dimension not in the top n, set the weights to 0 and the bias to the mean value of the output
        weight_type = layer.weight.data.dtype
        kept_weights = {idx: layer.weight.data[idx-1] for idx in top_layers_idx}
        kept_biases = {idx: layer.bias.data[idx-1] for idx in top_layers_idx}
    
        # Recombination point
        # For now working with dense layers we can just set the weights to 0 and the biases to the mean value of the output
        # TODO: When doing this we still have to compute the whole thing which is unnecessary. Develop some sort of adaptive dense layer
        #           that's a good name actually, AdaptiveDenseLayer
        
        # WILLIAM COMMENTS:
        # Right now, we are making the matices sparse,  
        # FOR device we will have to specify GPU_ID not the specific device
        idx_means = [torch.tensor([0]*layer.weight.data.shape[1], dtype=weight_type, device="cuda:0") if idx not in top_layers_idx  else kept_weights[idx] for idx in range(1, len(mean_vals)+1)]
        idx_biases = [torch.tensor(mean_vals[idx], dtype=weight_type, device="cuda:0") if idx not in top_layers_idx else kept_biases[idx] for idx in range(1, len(mean_vals)+1)]
        
        # Set the weights and biases
        # WILLIAM COMMENT:
        # Need to send to device as well.
                
        layer.weight.data = torch.stack(idx_means) #.to("cuda:0")
        layer.bias.data = torch.tensor(idx_biases) #.to("cuda:0")
        
    def down_sample_model(self, layers: list = None, n: int = 8) -> None:

        # For each layer, down sample the weights and biases

        # If we haven't computed the statistics yet, we do so
        if self.layer_output_std == {}:
            self.save_statistics()

        if layers is None:
            layers = [layer[0] for layer in self.layers]

        for layer in layers:
            self.down_sample_layer(layer, n=n)

"""

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


def freeze_params(model, trainable_params, freeze_attention=False):
        # FREEZE ALL params
        for _, param in model.named_parameters():
                param.requires_grad = False 
        # UNFREEZE Adaptive Layer FFNs & Model classifier + pooler 
        for layer in trainable_params:
            for name, param in layer.named_parameters(): 
                if freeze_attention:
                    if "attention" in name or "attn":
                        continue 
                param.requires_grad = True
        return None


##########################################################################################################################
##########################################################################################################################

###################            Here are all the tools to build the  AdaptiveLayer Class.               ###################   

##########################################################################################################################
##########################################################################################################################

def outlier_project_bert(layer, outliers):
    # Intermediates
    layer.intermediate.dense.weight = torch.nn.Parameter(layer.intermediate.dense.weight[:,outliers]) 
    # Dense
    layer.output.dense.weight = torch.nn.Parameter(layer.output.dense.weight[outliers,:])     
    return None

class DownSampleAttention(torch.nn.Module):
    """
    Super simple class to down sample by incides.
    """
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x[:,:,self.idx]
    
class UpSampleOutput(torch.nn.Module):
    """
    Pad the output of the downsampled FF with zero to maintain original model hidden dimension shape. 
    """
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        
    # WE can have different padding if we want. 
    def forward(self, x, attention_output):
        upsampled_output = torch.zeros(attention_output.shape).to(self.gpu_id)
        upsampled_output[:,:,self.idx] = x.type(torch.float32)
        return upsampled_output  

def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

def down_sample_kqv(layer, idx):
    layer.attention.self.num_attention_heads = 2
    layer.attention.self.attention_head_size = 4
    layer.attention.self.all_head_size = 8
    layer.attention.self.query.weight = torch.nn.Parameter(layer.attention.self.query.weight[idx,:])
    layer.attention.self.query.bias = torch.nn.Parameter(layer.attention.self.query.bias[idx])
    layer.attention.self.key.weight = torch.nn.Parameter(layer.attention.self.key.weight[idx,:])
    layer.attention.self.key.bias = torch.nn.Parameter(layer.attention.self.key.bias[idx])
    layer.attention.self.value.weight = torch.nn.Parameter(layer.attention.self.value.weight[idx,:])
    layer.attention.self.value.bias = torch.nn.Parameter(layer.attention.self.value.bias[idx])
    layer.attention.output.dense.weight = torch.nn.Parameter(layer.attention.output.dense.weight[:, idx])
    return

class AdaptiveBertLayer(torch.nn.Module):
    # add back config is we want. 
    def __init__(self, layer, idx):
        
        """
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
        down_sample_kqv(layer, idx)
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
        
        # Compute the intermediates
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

def outlier_project_gpt2(layer, outliers):
    """
    Downsample MLPs to the subspace of outliers. 
    """
    # Intermediates
    layer.c_fc.weight = torch.nn.Parameter(layer.c_fc.weight[outliers,:]) 
    print(layer.c_fc.weight.shape)
    # Dense
    layer.c_proj.weight = torch.nn.Parameter(layer.c_proj.weight[:,outliers]) #torch.nn.Parameter(torch.ones(outliers_shape,3072))
    layer.c_proj.bias = torch.nn.Parameter(layer.c_proj.bias[outliers]) #torch.nn.Parameter(torch.ones(outliers_shape))
    layer.c_proj.nf = len(outliers)

    return None     

class AdaptiveGPT2Layer(torch.nn.Module):
    # add back config is we want. 
    def __init__(self, layer, idx, gpu_id):
        
        """
        Look up config, but we are going to remove it for now. 
        """

        print("ADAPTIVE LAYER GPU ID", gpu_id) 
        
        super().__init__()
        # Outlier PEFT function
        self.down_sample_attention = DownSampleAttention(idx) 
        self.c_fc = layer.c_fc
        self.act = layer.act
        
        # NOTE:
        # We may want to
        # 1) pass output of self.act to layer.c_proj.weight. 
        # 2) UPSAMPLE
        # 3) ADD BIAS. 
        # GPT-2 is structured in a way that we are passing in a zero-padded tensor (since there is no residual-connection like BERT)
        
        self.c_proj = layer.c_proj
        #Outlier PEFT function
        self.up_sample_output = UpSampleOutput(idx, gpu_id)
        

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        outputs = self.down_sample_attention(hidden_states)
        outputs = self.c_fc(outputs)
        outputs = self.act(outputs)
        outputs = self.c_proj(outputs)
        outputs = self.up_sample_output(outputs, hidden_states)
        return outputs
    

