#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:33:55 2019

@author: assiene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

class InputUnit(nn.Module):
    
    def __init__(self, device, vocab_size, on_text=True, max_seq_len=512, batch_size=2, use_bert_encoder_for_question=True, d=512):
        
        super(InputUnit, self).__init__()
        self.d = d # Dimension of control and memory states
        self.S = None # Number of words in the question
        self.p = None # Number of reasoning steps
        self.vocab_size = vocab_size
        self.on_text = on_text
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.max_question_length = 0
        self.num_text_chunks = 35
        self.device = device
        
        self.use_bert_encoder_for_question = use_bert_encoder_for_question
        
        self.question_encoder = nn.LSTM(input_size=self.d, hidden_size=self.d, bidirectional=True)
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d)
        
        if self.use_bert_encoder_for_question == False:
            self.cws_projection = nn.Linear(self.d * 2, self.d)
            
        else:
            self.cws_projection = nn.Linear(768, self.d)
        
        if self.on_text == False:
            self.build_img_encoder()
            
        else:
            self.build_text_encoder()
        
        
    
    def forward(self, context, question):
        
        if self.on_text:
            queries = []
            max_text_length = 0
            for observation in context.observations:
                query = observation["query"]
                query_tokens_ids = self.string_to_token_ids(query)
                self.max_question_length = len(query_tokens_ids) if len(query_tokens_ids) > self.max_question_length else self.max_question_length
                queries.append(query_tokens_ids)
                max_text_length = len(observation["text"]) if len(observation["text"]) > max_text_length else max_text_length
                
            questions = torch.zeros((len(context.observations), self.max_question_length), dtype=torch.long).to(self.device)
            self.num_text_chunks = max_text_length // self.max_seq_len + 2
            
            for (i, query) in enumerate(queries):
                questions[i,:len(query)] = torch.tensor(query, dtype=torch.long).to(self.device).detach()
                
        
        if self.use_bert_encoder_for_question == False:
            question = self.embedding_layer(question)
            cws, (q, _) = self.question_encoder(question.transpose(0,1))
            q = q.transpose(0,1) # batch x 2 x d
            q = q.reshape(q.shape[0], -1) # batch x 2d
            cws = cws.transpose(0,1)
            
        else:
            with torch.no_grad():
                cws, _ = self.bert_model(questions) #batch_size x max_question_length x hidden_size=768
#                cws = torch.rand((self.batch_size, self.max_question_length, 768))
        
        cws = self.cws_projection(cws) # batch x S x d
        q = cws.mean(dim=1) # batch_size x d
        label_candidates_encoded = None
        
        K = context
        if self.on_text == False:
            K = self.resnet101(K)
            
        else:
            K, label_candidates_encoded = self.encode_text(K)
        
        K = self.context_encoder(K)
        
        if self.on_text == False:
            K = K.transpose(1,2).transpose(2,3) # batch x h x w x d
        
        return K, q, cws, label_candidates_encoded
    
    def build_img_encoder(self):
        self.resnet101 = models.resnet101(pretrained=True)
        modules = list(self.resnet101.children())[:-3]
        self.resnet101 = nn.Sequential(*modules)
        for p in self.resnet101.parameters():
            p.requires_grad = False
            
        self.context_encoder = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=self.d, kernel_size=3),
                nn.ELU(),
                nn.Conv2d(in_channels=self.d, out_channels=self.d, kernel_size=3),
                nn.ELU())
        
    def build_text_encoder(self):
        self.bert_tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False, max_len=self.max_seq_len)
        self.bert_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', pretrained_model_name_or_path="bert-base-cased")
        self.context_encoder = nn.Linear(in_features=768, out_features=self.d) #768 : hidden_size of transformer
        
        
    def encode_text(self, batch):
        
#        padded_context = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long)
        self.bert_model.eval()
        label_candidates_encoded = []
        contexts = []
        
        for (i, observation) in enumerate(batch.observations):
            encoded_text = self.encode_large_text(" ".join(observation["supports"])) # num_text_chunks x sequence_length=512 x hidden_size=768
            contexts.append(encoded_text)
            
            label_candidates_encoded.append([])
            for label_candidate in observation["label_candidates"]:
                label_candidate_tokens_ids = torch.tensor([self.string_to_token_ids(label_candidate)]).to(self.device)
                #segment_label_candidates_tensor = torch.zeros((1, len(label_candidate_tokens_ids)), dtype=torch.long)
                with torch.no_grad():
                    encoded_layers, _ = self.bert_model(label_candidate_tokens_ids) #1 x sequence_length x hidden_size=768
                    label_candidates_encoded[i].append(encoded_layers.mean(dim=1).squeeze(0)) # list(batch_size) x num_candidates x hidden_size
#                    label_candidates_encoded[i].append(torch.rand((768)))
            
            label_candidates_encoded[i] = torch.stack(label_candidates_encoded[i]).to(self.device)
#                indexed_tokens_supports.append(current_indexed_tokens_support)
#                segments_ids_supports.append([0] * len(current_indexed_tokens_support))
        
        # Convert inputs to PyTorch tensors
        #segments_supports_tensors = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long)
        contexts = torch.stack(contexts).to(self.device) # batch_size x num_text_chunks x sequence_length=512 x hidden_size=768
#        contexts = torch.rand((self.batch_size, self.num_text_chunks, self.max_seq_len, 768)).to(self.device)
        
        return contexts, label_candidates_encoded
    
    def string_to_token_ids(self, string):
        
        string = "[CLS] " + string + " [SEP]"
        tokenized_string = self.bert_tokenizer.tokenize(string)
        token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokenized_string)
        
        return token_ids
    
    def encode_large_text(self, text):
        
        subtext_length = self.max_seq_len - 2 # 2 for [CLS] and [SEP]
        subtexts = [text[i:i+subtext_length] for i in range(0, len(text), subtext_length)]
        encoded_text = torch.zeros((self.num_text_chunks, self.max_seq_len, 768)).to(self.device)
        for (i, subtext) in enumerate(subtexts):
            subtext_tokens_ids = torch.tensor([self.string_to_token_ids(subtext)]).to(self.device)
            
            with torch.no_grad():
                encoded_layers, _ = self.bert_model(subtext_tokens_ids) #1 x sequence_length=512 x hidden_size=768
                current_seq_len = encoded_layers.shape[1] #reason why we can't compute num_text_chunks dynamically : seq_len of encoded_layer will vary between the chunks. Since the resulting tensors won't have the same size, we can't stack them
                encoded_text[i,:current_seq_len,:] = encoded_layers[0]
                
        return encoded_text # num_text_chunks x sequence_length=512 x hidden_size=768
        