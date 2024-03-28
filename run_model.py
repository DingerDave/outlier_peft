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

def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def count_all_params(model):
    return sum(param.numel() for param in model.parameters())

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
   
def main(gpu_id: int, config: dict, world_size: int):
    # Wandb init
     
    print(gpu_id)
    
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
    
    print("PARAM COUNT", count_all_params(model))
    
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
    if config.model_name == "bert":  
        adaptive_config = adaptive_config[config.model_name + "-" + config.task]
        for i in config.layers:
            outliers = torch.tensor(adaptive_config[i][:config.num_outliers])
            outlier_project_bert(model.bert.encoder.layer[i], outliers)

            # STEP 2: Make the layers that were projected above AdaptiveLayers. 
            model.bert.encoder.layer[i] = AdaptiveBertLayer(model.bert.encoder.layer[i], outliers)

    
        #print("BERT {} SHAPE AFTER DOWNSAMPLING: {}".format("QUERY",str(model.bert.encoder.layer[i].attention.self.query.weight.shape)))
        #print("BERT {} SHAPE AFTER DOWNSAMPLING: {}".format("KEY",str(model.bert.encoder.layer[i].attention.self.key.weight.shape))) 
        #print("BERT {} SHAPE AFTER DOWNSAMPLING: {}".format("VALUE",str(model.bert.encoder.layer[i].attention.self.value.weight.shape)))
        #print("BERT {} SHAPE AFTER DOWNSAMPLING: {}".format("OUTPUT",str(model.bert.encoder.layer[i].attention.output.dense.weight.shape)))
        

        # STEP 3: Freeze all non-AdaptiveLayer FF parameter 
        trainable_params = [model.bert.encoder.layer[i] for i in config.layers]
        # ADD POOLER AND CLASSIFICATION HEAD TO TRAINABLE PARAMS
        trainable_params.append(model.bert.pooler)
        trainable_params.append(model.classifier)
        freeze_params(model, trainable_params)  

    elif config.model_name == "gpt2":
        # This downsamples IN PLACE    
        adaptive_config = adaptive_config[config.model_name + "-" + config.task]
        for i in config.layers:
            outliers = torch.tensor(adaptive_config[i][:config.num_outliers]) 
            outlier_project_gpt2(model.transformer.h[i].mlp, outliers)
        
        # STEP 2: Make the layers that were projected above AdaptiveLayers.
            model.transformer.h[i].mlp = AdaptiveGPT2Layer(model.transformer.h[i].mlp, outliers, gpu_id)
     
        trainable_params = [model.transformer.h[i] for i in config.layers]
        # ADD POOLER AND CLASSIFICATION HEAD TO TRAINABLE PARAMS
        trainable_params.append(model.transformer.ln_f)
        trainable_params.append(model.score)
        freeze_params(model, trainable_params) 
        
    # Specifying the correct optimizer params
    print(count_trainable_params(model))
    print("PARAM COUNT AFTER DOWNSAMLPE", count_all_params(model))


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
       
    # STEP 4: Train like normal  
    trainer = Trainer(config, model, train_loader, eval_loader, optimizer, gpu_id)   

    trainer.train()
    print("Training done") 
    destroy_process_group()

    
if __name__  == "__main__":
    # Argparser to create config 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int) 
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--model_name", default="bert", type=str)
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