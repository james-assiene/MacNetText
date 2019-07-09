#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:33:55 2019

@author: assiene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputUnit(nn.Module):
    
    def __init__(self, n_labels, d=512, match_candidates=True, hidden_size_candidate=768):
        
        super(OutputUnit, self).__init__()
        
        self.d = d
        
        self.linear1 = nn.Linear(2 * self.d, self.d)
        
        if match_candidates:
            self.linear2 = nn.Linear(self.d, hidden_size_candidate)
        else:
            self.linear2 = nn.Linear(self.d, n_labels)
        
    
    def forward(self, mp, q, label_candidates_encoded=None):
        
        #mp : batch x d
        #q : batch x 2d
        
        out = self.linear1(torch.cat([mp, q], dim=1))
        out = F.elu(out)
        out = self.linear2(out)
            
        if label_candidates_encoded is not None:
            scores = []
            max_num_candidates = 0
            mask = -1000
            for batch_index in range(len(label_candidates_encoded)):
                current_scores = label_candidates_encoded[batch_index] @ out[batch_index] # num_candidates
                scores.append(current_scores)  #list(batch_size) x list(num_candidatesà
                max_num_candidates = current_scores.shape[0] if current_scores.shape[0] > max_num_candidates else max_num_candidates
                    
                
            out = torch.zeros((len(label_candidates_encoded)), max_num_candidates) + mask
            for batch_index in range(out.shape[0]):
                num_candidates = scores[batch_index].shape[0]
                out[batch_index, :num_candidates] = scores[batch_index]
        
        return out