#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:01:04 2019

@author: assiene
"""

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms

from pytorch_pretrained_bert import BertConfig

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.utils import padded_3d
from parlai.core.logs import TensorboardLogger

from .MacNetwork import MacNetwork

from tensorboardX import SummaryWriter

losses = []

class MacNetAgent(TorchAgent):
    
    @staticmethod    
    def add_cmdline_args(argparser):
        TorchAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group("MacNet Arguments")
        
        agent.add_argument("-dim", "--dimension", type=int, default=512, help="Dimension for all layers")
        agent.add_argument("-nrh", "--num-reasoning-hops", type=int, default=12, help="Number of reasoning hops")
            
        MacNetAgent.dictionary_class().add_cmdline_args(argparser)
        
        return agent
    
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        
        if opt['tensorboard_log'] is True:
            self.writer = TensorboardLogger(opt)
            
        self.criterion = nn.CrossEntropyLoss()
        torch.autograd.set_detect_anomaly(True)
        self.vocab_size = 645575  
        self.n_labels = 100
        self.learning_rate = 1e-4
        self.batch_size = opt["batchsize"]
        self.max_seq_length = 512
        self.on_text = True
        self.batch_iter = 0
        
        self.model = MacNetwork(vocab_size=self.vocab_size, n_labels=self.n_labels, batch_size=self.batch_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.writer = SummaryWriter()
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self.model.to(self.device)
        
            
    def vectorize(self, *args, **kwargs):
        """Override options in vectorize from parent."""
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        #kwargs['split_lines'] = True
        return super().vectorize(*args, **kwargs)
        
    
    def train_step(self, batch):
        
        print("hello")
        top_n = 3
        
        if self.on_text == False:
            image_preprocessing = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            images = torch.stack([image_preprocessing(img) for img in batch.image])
            contexts = images
            
        else:
            contexts = batch
               
        questions = batch.text_vec
        
        self.optimizer.zero_grad()
        
        answers_dist = self.model(contexts, questions)
        label_indices = torch.zeros(len(batch.observations), dtype=torch.long).to(self.device)
        for i in range(label_indices.shape[0]):
            label_indices[i] = batch.observations[i]["label_candidates"].index(batch.observations[i]["labels"][0])
        
        loss = self.criterion(answers_dist, label_indices)
        loss.backward(retain_graph=True)
        
        self.writer.add_scalar("data/loss", loss, self.batch_iter)
        
        for name, param in self.model.named_parameters():
            if "bert_model" not in name:
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.batch_iter)
            #self.writer.add_histogram(name + "_grad", param.grad.clone().cpu().data.numpy(), self.batch_iter)
        for mac_cell in self.model.mac_cells:
            for name_in, param_in in mac_cell.named_parameters():
                self.writer.add_histogram(name_in, param_in.clone().cpu().data.numpy(), self.batch_iter)
                #self.writer.add_histogram(name_in + "_grad", param_in.grad.clone().cpu().data.numpy(), self.batch_iter)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        
        pred = answers_dist.argmax(dim=1)
        answers = [batch.observations[i]["label_candidates"][pred[i]] for i in range(label_indices.shape[0])]
        
        for (i, observation) in enumerate(batch.observations):
            sorted_pred = answers_dist.argsort(dim=1, descending=True)
            to_log = "True answer {} : {}".format(i + 1, observation["labels"][0])
            self.writer.add_text("preds", to_log, self.batch_iter, 0)
            print(to_log)
            num_candidates = len(observation["label_candidates"])
            to_log = "Out of {} : ".format(num_candidates)
            self.writer.add_text("preds", to_log, self.batch_iter, 0)
            print(to_log)
            for j in range(min(top_n, num_candidates)):
                to_log = "Top {} answer : {}".format(j + 1, observation["label_candidates"][sorted_pred[i][j]])
                self.writer.add_text("preds", to_log, self.batch_iter, 0)
                print(to_log)
        
        self.batch_iter+= 1
        
        return Output(answers)
        
        
    
    def eval_step(self, batch):
        
        if self.on_text == False:
            image_preprocessing = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            images = torch.stack([image_preprocessing(img) for img in batch.image])
            contexts = images
            
        else:
            contexts = batch
               
        questions = batch.text_vec
        
        answers_dist = self.model(contexts, questions)
        
        pred = answers_dist.argmax(dim=1)
        answers = [batch.observations[i]["label_candidates"][pred[i]] for i in range(pred.shape[0])]
        
        return Output(answers)
        
    
    


from parlai.scripts.train_model import TrainLoop, setup_args

if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    opt["tensorboard_log"] = True
    opt["model_file"] = "m1"
    opt["tensorboard_tag"] = "task,batchsize"
    opt["tensorboard_metrics"] = "all"
    opt["metrics"] = "all"
    opt["model"] = "mac_net"
    opt["no_cuda"] = True
    opt['history_size'] = 1
    opt['truncate'] = -1
    opt["rank_candidates"] = False
    TrainLoop(opt).train()

