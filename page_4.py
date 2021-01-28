# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:51:47 2021

@author: ULVI PC
"""

import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms

#1: Define a transform to pre-process the testing images.
transform_test = transforms.Compose([ 
                    transforms.Resize(256),                          # smaller edge of image resized to 256
                    transforms.RandomCrop(224),                      # get 224x224 crop from random location
                    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                    transforms.ToTensor(),                           # convert the PIL Image to a tensor
                    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                         (0.229, 0.224, 0.225))])

#-#-#-# Do NOT modify the code below this line. #-#-#-#

# Create the data loader.

data_loader = get_loader(transform=transform_test,    
                         mode='test')

import numpy as np
import matplotlib.pyplot as plt


# Obtain sample image before and after pre-processing.
orig_image, image = next(iter(data_loader))

# Visualize sample image, before pre-processing.
#plt.imshow(np.squeeze(orig_image))
#plt.title('example image')
#plt.show()

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os
import torch
from model import EncoderCNN, DecoderRNN

# TODO #2: Specify the saved models to load.
encoder_file = "encoder-1.pkl"
decoder_file = "decoder-1.pkl"

# TODO #3: Select appropriate values for the Python variables below.
embed_size = 250
hidden_size = 200

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)

# Move image Pytorch Tensor to GPU if CUDA is available.
image = image.to(device)

# Obtain the embedded image features.
features = encoder(image).unsqueeze(1)

# Pass the embedded image features through the model to get a predicted caption.
output = decoder.sample(features)
#print('example output:', output)

assert (type(output)==list), "Output needs to be a Python list" 
assert all([type(x)==int for x in output]), "Output should be a list of integers." 
assert all([x in data_loader.dataset.vocab.idx2word for x in output]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

import string


def clean_sentence(output):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding sentence (as a single Python string).
    """
    sentence=""
    for x in output[1:-1]:
         word = data_loader.dataset.vocab.idx2word[x]
         if word == "<end>":
             sentence = sentence
         else:
             sentence += word+ " "
    return sentence

sentence = clean_sentence(output)
#print('example sentence:', sentence)

assert type(sentence)==str, 'Sentence needs to be a Python string!'

def get_prediction():
    orig_image, image = next(iter(data_loader))
        
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)    
    sentence = clean_sentence(output)
    plt.imshow(np.squeeze(orig_image))
    plt.title(sentence)
    plt.show()
    print(sentence)
    
  
get_prediction()