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
import wandb
import typing
from datasets import Dataset
# Custom Functions 
from analysis import * 
from training_utils import *
import hickle
from typing import Dict, Iterable, Callable, Optional
import torch.nn as nn
import json

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
        self.model = DDP(model, device_ids=[gpu_id]) 
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
        self.train_data.sampler.set_epoch(epoch)
        # Send everything to device and call run_batch 
        for _, batch in enumerate(self.train_data):
            batch = {key: value.to(self.gpu_id) for key, value in batch.items()}    
            self._run_batch(batch)
   
    def _save_model(self):
        """ 
        Saves model checkpoint to PATH 
        """ 
        PATH = "../peft_models/" + self.config.model_name + "_" + str(self.config.seed) + "_" + self.config.task + ".pth"
        torch.save(self.model.module.state_dict(), PATH)
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
    ddp_setup(rank,world_size)   
    # Sow seeds  
    sow_seeds(int(config.seed))
    print("SEED", config.seed) 

    # DAVE TEST CODE BELOW
    model, train_data, eval_data, optimizer = load_classification_objs(config)
     
    # from pprint import pprint
    train_loader = prepare_dataloader(config, train_data)
    eval_loader = prepare_dataloader(config, eval_data, is_eval=True) 
    
    # DAVIDS CODE: Run intermediate output extractor. 
    #inter_outputs = IntermediateOutputExtractor(model, layers = ["bert.encoder.layer.11.intermediate.dense", "bert.encoder.layer.10.intermediate.dense", "bert.encoder.layer.3.intermediate.dense"])
    #model.to("cuda:0") 
    #inter_outputs(next(iter(train_loader)))
    #inter_outputs.down_sample_model(n=8) 
    #trainer = Trainer(config, inter_outputs.model, train_loader, eval_loader, optimizer, rank)    
    # Create dataloaders

    #WILLIAMS CODE:

    #STEP 1: "pre-process" model weights by down-projecting specified layers. 
    # will clean this up layer, but can specify layer indices in the config, then run this function. 
    # outliers is a tensor of dimension indices. 
    outliers = torch.tensor([0,1,2])
    # This downsamples IN PLACE 
    outlier_project_bert(model.bert.encoder.layer[11], outliers)

    # STEP 2: Make the layers that were projected above AdaptiveLayers. 
    model.bert.encoder.layer[11] = AdaptiveBertLayer(model.bert.encoder.layer[11], outliers)


    
    # STEP 3: Train like normal 

    trainer = Trainer(config, model, train_loader, eval_loader, optimizer, rank)   

    trainer.train()
    print("Training done") 
    
    destroy_process_group()

    
if __name__  == "__main__":
    # Argparser to create config 
    print("Running")
    quit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int) 
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--model_name", default="bert", type=str)
    parser.add_argument("--seed", default=1, type=int)
    # --training also takes the argument "Mini" which will train the model on a very small subset for debugging purposes.
    parser.add_argument("--training", default="True", type=str) 
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    config = parser.parse_args() 
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(config, world_size), nprocs=world_size) 
