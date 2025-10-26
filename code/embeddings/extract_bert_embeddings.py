#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import argparse
import os
import time as tm
from transformers import AutoModel, AutoTokenizer
import json

# Use GPU if possible
device = "cuda:0" if torch.cuda.is_available() else "cpu"

@torch.inference_mode()
def get_model_layer_representations(args, text_array, word_ind_to_extract):
    seq_len = args.sequence_length
    model_name = args.model
    with open('text_model_config.json', 'r') as f:
        model_config = json.load(f)[model_name]
        model_hf_path = model_config['huggingface_hub']
    print(model_config, model_hf_path, seq_len)
    n_total_layers = model_config['num_layers']
    
    model = AutoModel.from_pretrained(model_hf_path, cache_dir="./").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, cache_dir="./")
    model.eval()

    # get the token embeddings
    token_embeddings = []
    for word in text_array:
        current_token_embedding = get_model_token_embeddings([word], tokenizer, model)
        token_embeddings.append(np.mean(current_token_embedding.detach().numpy(), 1))
    
    # where to store layer-wise embeddings of particular length
    Words_layers_representations = {}
    for layer in range(n_total_layers):
        Words_layers_representations[layer] = []
    Words_layers_representations[-1] = token_embeddings

    # Before we've seen enough words to make up the seq_len
    # Extract index 0 after supplying tokens 0 to 0, extract 1 after 0 to 1, 2 after 0 to 2, ... , 19 after 0 to 19
    start_time = tm.time()
    for truncated_seq_len in range(1, 1+seq_len):
        word_seq = text_array[:truncated_seq_len]
        from_start_word_ind_to_extract = -1 + truncated_seq_len
        Words_layers_representations = add_avrg_token_embedding_for_specific_word(word_seq, tokenizer, model, 
                                                            from_start_word_ind_to_extract, Words_layers_representations, model_config)
        if truncated_seq_len % 100 == 0:
            print('Completed {} out of {}: {}'.format(truncated_seq_len, len(text_array), tm.time()-start_time))
            start_time = tm.time()

    word_seq = text_array[:seq_len]
    if word_ind_to_extract < 0: # the index is specified from the end of the array, so invert the index
        from_start_word_ind_to_extract = seq_len + word_ind_to_extract
    else:
        from_start_word_ind_to_extract = word_ind_to_extract
        
    # Then, use sequences of length seq_len, still adding the embedding of the last word in a sequence
    for end_curr_seq in range(seq_len, len(text_array)):
        word_seq = text_array[end_curr_seq-seq_len+1:end_curr_seq+1]
        Words_layers_representations = add_avrg_token_embedding_for_specific_word(word_seq, tokenizer, model,
                                                            from_start_word_ind_to_extract, Words_layers_representations, model_config)

        if end_curr_seq % 100 == 0:
            print('Completed {} out of {}: {}'.format(end_curr_seq, len(text_array), tm.time()-start_time))
            start_time = tm.time()

    print('Done extracting sequences of length {}'.format(seq_len))
    return Words_layers_representations

# extracts layer representations for all words in words_in_array
@torch.inference_mode()
def predict_model_embeddings(words_in_array, tokenizer, model, model_config):    
    n_seq_tokens = 0
    seq_tokens = []
    
    word_ind_to_token_ind = {}             # dict that maps index of word in words_in_array to index of tokens in seq_tokens
    
    for i,word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []      # initialize token indices array for current word
        word_tokens = tokenizer.tokenize(word)
            
        for token in word_tokens:
            seq_tokens.append(token)
            word_ind_to_token_ind[i].append(n_seq_tokens)
            n_seq_tokens = n_seq_tokens + 1
    
    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    if model_config['model_type'] == 'encoder' or model_config['model_type'] == 'decoder':
        outputs = model(tokens_tensor, output_hidden_states=True)
        hidden_states = outputs['hidden_states'][1:]    # This is a tuple: (layer1, layer2, ..., layer6)
        all_layers_hidden_states = hidden_states
    elif model_config['model_type'] == 'encoder-decoder':
        outputs = model(tokens_tensor, decoder_input_ids=tokens_tensor, output_hidden_states=True)
        encoder_hidden_states = outputs['encoder_hidden_states'][1:]    # This is a tuple: (layer1, layer2, ..., layer6)
        decoder_hidden_states = outputs['decoder_hidden_states'][1:]
        all_layers_hidden_states = encoder_hidden_states + decoder_hidden_states
    
    return all_layers_hidden_states, word_ind_to_token_ind, None
  
# add the embeddings for a specific word in the sequence
@torch.inference_mode()
def add_word_model_embedding(model_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):
    if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.cpu().detach().numpy()
        model_dict[specific_layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0))
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.cpu().detach().numpy()
            model_dict[layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0)) # avrg over all tokens for specified word
    return model_dict

# predicts representations for specific word in input word sequence, and adds to existing layer-wise dictionary
@torch.inference_mode()
def add_avrg_token_embedding_for_specific_word(word_seq,tokenizer,model,from_start_word_ind_to_extract,model_dict, model_config):
    
    word_seq = list(word_seq)
    all_sequence_embeddings, word_ind_to_token_ind, _ = predict_model_embeddings(word_seq, tokenizer, model, model_config)
    token_inds_to_avrg = word_ind_to_token_ind[from_start_word_ind_to_extract]
    model_dict = add_word_model_embedding(model_dict, all_sequence_embeddings,token_inds_to_avrg)
    
    return model_dict

# get the model token embeddings
@torch.inference_mode()
def get_model_token_embeddings(words_in_array, tokenizer, model):    
    n_seq_tokens = 0
    seq_tokens = []
    
    word_ind_to_token_ind = {}             # dict that maps index of word in words_in_array to index of tokens in seq_tokens
    
    for i,word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []      # initialize token indices array for current word
        word_tokens = tokenizer.tokenize(word)
            
        for token in word_tokens:
            seq_tokens.append(token)
            word_ind_to_token_ind[i].append(n_seq_tokens)
            n_seq_tokens = n_seq_tokens + 1
    
    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    
    input_embedding_module = model.base_model.get_input_embeddings()
    token_embeddings = input_embedding_module(tokens_tensor.to(torch.long)).cpu()
    
    return token_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "BERT feature extraction for subset")
    parser.add_argument("--input_dir", help="Choose input directory", type = str, required=True)
    parser.add_argument("--model", help="Choose model", type = str, required = True)
    parser.add_argument("--sequence_length", help="Choose context", type = int)
    parser.add_argument("--output_file", help="Choose output filename", type = str)
    args = parser.parse_args()

    print(args.input_dir)

    word_ind_to_extract = -1
    
    # Get list of story files (excluding TextGrid directory)
    story_files = []
    for item in sorted(os.listdir(args.input_dir)):
        if os.path.isfile(os.path.join(args.input_dir, item)) and item != 'TextGrid':
            story_files.append(item)
    
    stories_files = {}
    for eachstory in story_files:
        print(f"Processing {eachstory}")
        read_eachstory = open(os.path.join(args.input_dir, eachstory),'r')
        words = read_eachstory.read().strip().split('\n')
        read_eachstory.close()

        embeddings = get_model_layer_representations(args, np.array(words), word_ind_to_extract)
        stories_files[eachstory] = embeddings

    np.save(args.output_file+'_'+args.model+'_'+str(args.sequence_length), stories_files)
    print(f"Saved embeddings to {args.output_file}_{args.model}_{args.sequence_length}.npy")
