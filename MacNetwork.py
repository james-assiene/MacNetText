#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:05:09 2019

@author: assiene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .InputUnit import InputUnit
from .OutputUnit import OutputUnit
from .MacCell import MacCell

class MacNetwork(nn.Module):
    
    def __init__(self, device, vocab_size=3, n_labels=1, use_lstm=True, batch_size=2, d=512, p=12, on_text=True, max_seq_len=512):
        
        super(MacNetwork, self).__init__()
        self.d = d # Dimension of control and memory states
        self.vocab_size = vocab_size
        self.n_labels = n_labels
        self.p = p
        self.on_text = on_text
        self.max_seq_len = max_seq_len
        self.device = device
        self.use_lstm = use_lstm
        
        self.input_unit = InputUnit(self.device, self.vocab_size, self.on_text, self.max_seq_len, d=self.d)
        self.mac_cells = [MacCell(self.device, self.use_lstm, self.d).to(self.device) for i in range(self.p)]
        self.output_unit = OutputUnit(self.n_labels, d=self.d, use_lstm=self.use_lstm)

        print(self.input_unit)
        print(self.output_unit)
        print(self.mac_cells[0])
        
    def init_hidden(self, batch_size):
        
        self.batch_size = batch_size
        
        self.m0 = torch.rand((self.batch_size, self.d)).to(self.device)
        self.c0 = torch.rand((self.batch_size, self.d)).to(self.device)
        self.C_past = torch.zeros((self.batch_size, self.p + 1, self.d)).to(self.device)
        self.C_past[:,0,:] = self.c0
        self.M_past = torch.zeros((self.batch_size, self.p + 1, self.d)).to(self.device)
        self.M_past[:,0,:] = self.m0
        
    
    def forward(self, context, question):
        self.init_hidden(context.shape[0])
        K, q, cws, label_candidates_encoded = self.input_unit(context, question)
        ci, mi = self.c0, self.m0
        
        for i in range(self.p):
            mac_cell = self.mac_cells[i]
            mac_cell.K = K
            mac_cell.q = q
            mac_cell.cws = cws
            mac_cell.C_past = self.C_past[:,:i+1,:].clone()
            mac_cell.M_past = self.M_past[:,:i+1,:].clone()
            ci, mi = mac_cell(ci, mi)
            ci = ci.detach()
            mi = mi.detach()
            self.C_past[:,i+1,:] = ci
            self.M_past[:,i+1,:] = mi
        
        
        ans = self.output_unit(mi, q, label_candidates_encoded)
        return ans