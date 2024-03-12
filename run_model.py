#PyTorch & Numpy
import torch 
from torch.utils.data import DataLoader 
import numpy as np
# Pytorch stuff for DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp
# Misc  
import argparse 
#import wandb
import tqdm
import typing
from datasets import Dataset
# Custom Functions 
from analysis import * 
from training_utils import *
import hickle
from typing import Dict, Iterable, Callable, Optional
import torch.nn as nn
import json
import yaml 

def ddp_setup(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        config: dict,  
        model: torch.nn.Module, 
        train_data: DataLoader, 
        eval_data: DataLoader, 
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        raw_data: typing.Optional[Dataset] = None, 
        squad_val: typing.Optional[Dataset] = None 
        ) -> None:  
        self.config = config  
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.eval_data = eval_data 
        self.optimizer = optimizer
        self.best_performance = 0
        self.steps = 0 
        #self.model = DDP(model, device_ids=[gpu_id]) 
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler()
        self.raw_data = raw_data
        self.squad_val = squad_val
     
    def _run_batch(self, batch):
       
        """ 
        INPUT: source: mini-batch data, targets: mini-batch labels, C0: shrinkage covariance matrix (only for istar regularization)  
        This function computes a forward, then backward pass for a mini-batch of data. Choice of regularizer is specified in the config file.  
        """   
        # THIS IS WHERE I NEED TO CALL MY MONITOR TRAINING         
        with torch.cuda.amp.autocast(dtype=torch.float16): 
        #with torch.cuda.amp.autocast(dtype=torch.float32):
            outputs = self.model(**batch, output_hidden_states=False)    
            loss = outputs.loss 
        #NOTE lets train all of our models first except for 1 random seed then for the last random seed run training analysis 
        # Backprop (scaler for fp16 training) 
        self.scaler.scale(loss).backward()  
        self.scaler.step(self.optimizer) 
        self.scaler.update() 
        self.model.zero_grad()
        self.optimizer.zero_grad() 

        self.steps += 1

    def _run_epoch(self, epoch):

        """ 
        INPUT: the current epoch for a given gpu-id
        Sends mini-batches to device and calls _run_batch to complete a single epoch.
        Note: At the start of each epoch, we create a new shrinkage matrix to reflect changes in the models representations. 
        """ 
        b_sz = len(next(iter(self.train_data))['input_ids']) 
        #self.train_data.sampler.set_epoch(epoch)
        # Send everything to device and call run_batch 
        print("Num_Iters", len(self.train_data))
        for _, batch in tqdm.tqdm(enumerate(self.train_data)):
            batch = {key: value.to(self.gpu_id) for key, value in batch.items()}    
            self._run_batch(batch)
   
    def _save_model(self):
        """ 
        Saves model checkpoint to PATH 
        """ 
        #PATH = "../peft_models/" + self.config.model_name + "_" + str(self.config.seed) + "_" + self.config.task + ".pth"
        PATH = "../peft_models/" + self.config.model_name + "_" + str(self.config.seed) + "_" + self.config.task + ".pth"
        torch.save(self.model.state_dict(), PATH)
        print("MODEL SAVED")
      
    def train(self): 
        """ 
        Train the model for num_epochs and save the model for the last epoch.  
        """ 
        #wandb.watch 
        for epoch in range(self.config.num_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0: 
                # WE PRINT THIS JUST TO MAKE SURE TRAINING IS HAPPENING. 
                acc = classification_eval(self.config, self.eval_data, self.model)   
                print("ACC", acc) 
                #wandb.log({"Accuracy": acc}) 
                        
                if epoch+1 == self.config.num_epochs: 
                    # SAVE MODEL AND RUN ANALYSIS 
                    print("DONE") 
                    self._save_model()   

class IntermediateOutputExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Optional[list] = None):
        """ Wrapper Class for extracting statistics from intermediate layers of a model."""
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

            
def main(rank: int, config: dict, world_size: int):
    # Wandb init
    print(config) 
    # Monitor everything with wandb. NOTE: only logging metrics for GPU0. So, look at the results files and NOT these. This is just for monitoring experiments.   
    results = {}   
    #wandb.init(project=config.model_name + "_" + config.task + "_seeds", name=str(config.seed))  
    # Training with DDP
    #ddp_setup(rank,world_size)   
    # Sow seeds  
    sow_seeds(int(config.seed))
    print("SEED", config.seed) 

    # DAVE TEST CODE BELOW
    model, train_data, eval_data = load_classification_objs(config)
    train_loader = prepare_dataloader(config, train_data)
    eval_loader = prepare_dataloader(config, eval_data, is_eval=True) 

    # @ David: I was planning on keeping the specific outliers for given layers in an "adaptive config" 
    # mainly bc this is going to get very large if we have more models and more strategies for getting outlliers.  
    # Load adaptive config -> What's the best way to do this? 

    adaptive_config = {
        
        "bert-sst2": {
            1: [557, 136, 15, 304, 562],
            2: [557, 283, 136, 505, 562], 
            3: [557, 283, 136, 505, 143], 
            4: [557, 283, 136, 505, 455],
            5: [557, 283, 136, 505, 455],
            6: [557, 283, 136, 505, 455], 
            7: [557, 283, 136, 505, 455], 
            8: [557, 304, 136, 283, 138],
            9: [557, 304, 138, 136, 98],
            10: [557, 304, 138, 136, 609], 
            11: [557, 609, 252, 455, 697]
        },


        "gpt2-sst2":{
        1: [138, 378, 447, 64, 393],
        2: [138, 447, 378, 64, 393], 
        3: [447, 138, 378, 64, 393], 
        4: [447, 138, 378, 64, 39],
        5: [447, 138, 378, 373, 64],
        6: [447, 138, 378, 373, 64], 
        7: [447, 138, 378, 481, 373], 
        8: [447, 138, 378, 481, 373],
        9: [447, 138, 481, 373, 314],
        10: [447, 138, 481, 378, 314], 
        11: [447, 138, 481, 373, 496],  
    } 
    }

    #STEP 1: "pre-process" model weights by down-projecting specified layers. 
    # will clean this up layer, but can specify layer indices in the config, then run this function. 
    # outliers is a tensor of dimension indices.
    if config.model_name == "bert":  
        adaptive_config = adaptive_config[config.model_name + "-" + config.task]
        for i in config.layers:
            outliers = torch.tensor(adaptive_config[i][:config.num_outliers])
            outlier_project_bert(model.bert.encoder.layer[i], outliers)

        # STEP 2: Make the layers that were projected above AdaptiveLayers. 
            model.bert.encoder.layer[i] = AdaptiveBertLayer(model.bert.encoder.layer[i], outliers)
        
        # How was I freezing params before? I accidentally overwrote it lmao. 
        # I did it better before. This is a little janky.
        # STEP 3: Freeze all non-AdaptiveLayer FF parameter 
        trainable_params = [model.bert.encoder.layer[i] for i in config.layers]
        # ADD POOLER AND CLASSIFICATION HEAD TO TRAINABLE PARAMS
        trainable_params.append(model.bert.pooler)
        trainable_params.append(model.classifier)  
        # FREEZE ALL params
        for _, param in model.named_parameters():
                param.requires_grad = False 
        # UNFREEZE Adaptive Layer FFNs & Model classifier + pooler 
        for layer in trainable_params:
            for name, param in layer.named_parameters():
                
                """
                if "attention" in name:
                    continue 
                """
                
                param.requires_grad = True

    elif config.model_name == "gpt2":
        # This downsamples IN PLACE    
        adaptive_config = adaptive_config[config.model_name + "-" + config.task]
        for i in config.layers:
            outliers = torch.tensor(adaptive_config[i][:config.num_outliers]) 
            outlier_project_gpt2(model.transformer.h[i].mlp, outliers)
        
        # STEP 2: Make the layers that were projected above AdaptiveLayers. 
            model.transformer.h[i].mlp = AdaptiveGPT2Layer(model.transformer.h[i].mlp, outliers)
     
        trainable_params = [model.transformer.h[i] for i in config.layers]
        # ADD POOLER AND CLASSIFICATION HEAD TO TRAINABLE PARAMS
        trainable_params.append(model.transformer.ln_f)
        trainable_params.append(model.score)

        # FREEZE ALL params
        for _, param in model.named_parameters():
                param.requires_grad = False 
        # UNFREEZE Adaptive Layer FFNs & Model classifier + pooler 
        for layer in trainable_params:
            for name, param in layer.named_parameters():
                param.requires_grad = True
        
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("TRUE ", name)
            else:
                print("False", name) 

    # Specifying the correct optimizer params
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
       
    # STEP 4: Train like normal  
    trainer = Trainer(config, model, train_loader, eval_loader, optimizer, rank)   

    trainer.train()
    print("Training done") 
    destroy_process_group()

    
if __name__  == "__main__":
    # Argparser to create config 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int) 
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--model_name", default="gpt2", type=str)
    parser.add_argument("--seed", default=1, type=int)  
    # --training also takes the argument "Mini" which will train the model on a very small subset for debugging purposes.
    parser.add_argument("--training", default="True", type=str) 
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    
    # AdaptiveLayer PARAMS
    # Can specify specific layers or integers. "all" downsamples all. "none" downsamples none
    parser.add_argument("--layers",  nargs="+", type=int)
    parser.add_argument("--num_outliers", default=2, type=int)
    config = parser.parse_args() 
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(config, world_size), nprocs=world_size) 
    #main(0, config, world_size)