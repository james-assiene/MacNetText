#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:05:33 2019

@author: assiene
"""

import torch
import nlp

dataset = nlp.load_dataset("qangaroo", "wikihop")
train_set = dataset["train"]
val_set = dataset["validation"]
dataset = dataset.map(lambda example : {"support" : " ".join(example["supports.support"])})