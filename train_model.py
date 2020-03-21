import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import Network
from network_param import NetworkParam
import openpyxl
import os
import numpy as np
import pathlib
import spacy
from itertools import chain 
import argparse


google_cloud = True

base_data_path =  os.path.join(pathlib.Path().absolute(), 'Data')
base_model_path =  os.path.join(pathlib.Path().absolute(), 'Models')
sentence_word_limit = 5

if google_cloud:
    base_data_path = os.path.join(r'/home/amitudedhia/BidOptimization/Code', 'Data')
    base_model_path = os.path.join(r'/home/amitudedhia/BidOptimization/Code', 'Models')
print("Data file path =", base_data_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def get_campaign_vector(campaign):
    vector = [0 , 0, 0 , 0]
    map = {
        "Hair Oil || SP || Generic || 21st feb'20" : 0,
        "Pain relief Oil || SP || Generic || 22nd Nov'19" : 1,
        "Pain relief Oil || SP || Generic-New || 18th Jan'19" : 2,
        "Sunscreen || SP || Generic || 24th Nov'19" : 3,
    }
    if campaign in map:
        vector[map[campaign]] = 1
        return vector
    else:
        return None

def get_match_type_vector(match_type):
    vector = [0 , 0]
    map = {
        "EXACT" : 0,
        "PHRASE" : 1
    }
    if match_type in map:
        vector[map[match_type]] = 1
        return vector
    else:
        return None

bid_strategy_map = {
    'Dynamic bids - up and down': 1,
    'Dynamic bids - down only': 2
}

nlp = spacy.load("en_trf_bertbaseuncased_lg")

def get_sentence_vector(sentence):
    vectors = nlp(sentence).tensor.tolist()
    if len(vectors) < sentence_word_limit:
        for _i in range(sentence_word_limit - len(vectors)):
            vectors.append([0]*768)
    elif len(vectors) > sentence_word_limit:
        vectors = vectors[:sentence_word_limit]

    flatten_vector = list(chain.from_iterable(vectors))
    return flatten_vector

def get_training_data_item(row):
    training_data_item_input = []
    training_data_item_input.extend(get_campaign_vector(row['CampaignName']))
    training_data_item_input.extend(get_match_type_vector(row['MatchType']))
    #training_data_item_input.append(bid_strategy_map[row['Bidding strategy']])
    training_data_item_input.extend(get_sentence_vector(row['Targeting']))
    training_data_item_output = []
    #training_data_item_output.append(row['TotalCampaignImpressions'])

    standardize_data = False
    normalize_data = True
    if standardize_data:
        #output data needs to be standardized
        impressions = (row['Impressions'] - 352.1933216) / 1353.593304
        clicks = (row['Clicks'] - 1.157293497) / 3.809080996
        cpc = row['CPC']
        if not cpc:
            cpc = 0
        cpc = (cpc - 24.38682728) / 18.15524255
        orderValue = (row['OrderValue'] - 46.69323374) / 213.8099544
    
    if normalize_data:
        #output data needs to be normalized
        impressions = row['Impressions'] / 2000
        clicks = row['Clicks'] / 50
        cpc = row['CPC']
        if not cpc:
            cpc = 0
        cpc = cpc / 100
        orderValue = row['OrderValue'] / 1000

    training_data_item_output.append(impressions)
    training_data_item_output.append(clicks)
    
    training_data_item_output.append(cpc)
    training_data_item_output.append(orderValue)
    
    return training_data_item_input, training_data_item_output

def get_training_data():
    training_data_input = []
    training_data_output = []

    wb = openpyxl.load_workbook(os.path.join(base_data_path, 'Train_Data.xlsx')) 
    ws = wb.active
    header = ['Date', 'CampaignName', 'Bidding strategy', 'TargetingType', 'Targeting', 
        'MatchType', 'TotalCampaignImpressions', 'Impressions', 'Clicks', 
        'CPC', 'OrderValue']

    row_dict = {}
    for row in ws.iter_rows(min_row=2):
        if not row[0] or not row[0].value:
            break

        index = 0
        for index in range(len(header)):
            row_dict[header[index]] = row[index].value

        if row_dict['TargetingType'] == 'Manual targeting':
            # filter out some data
            if row_dict['CampaignName'] == "Night Cream || SP || Generic || 24th Nov'19":
                continue
            if row_dict['MatchType'] == "EXACT":
                pass
            input, output = get_training_data_item(row_dict)
            training_data_input.append(input)
            training_data_output.append(output)

    return np.array(training_data_input), np.array(training_data_output)

def train_model(input, output, num_batches, activation_function, network_param, lr, weight_decay):
    nn_instance = Network(device, network_param.num_input, network_param.num_output, 
                            network_param.num_hidden_layers, network_param.hidden_layer_info,
                            activation_function).to(device)
    nn_instance.train()
    print(nn_instance)
    optimizer = optim.Adam(nn_instance.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = optim.SGD(nn_instance.parameters(), lr=lr, weight_decay=weight_decay)

    total_data_points = len(input)
    batch_size = total_data_points // num_batches
    current_batch = 0
    loss_platue_count = 0
    for epoch in range(30001):
        # get the next batch
        total_loss = 0
        epoch_completed = False
        start = -1
        end = -1
        prev_loss = 9223372036854775807 #2^63 - 1
        while not epoch_completed:

            start = end + 1
            end = start + batch_size - 1
            if end >= total_data_points - 1:
                end = total_data_points - 1
                epoch_completed = True

            current_input = torch.Tensor(input[start: end + 1]).to(device)
            current_output = torch.Tensor(output[start: end + 1]).to(device)

            if 4 == 5:
                if epoch % 1000 == 0:
                    print()
                    print("weights at epoch", epoch)
                    for name, param in nn_instance.named_parameters():
                        if param.requires_grad:
                            print (name, param.data)
                            

            # compute predictions
            preds = nn_instance(current_input)

            if 4 == 5:
                if epoch % 1000 == 0:
                    print()
                    print("preds at epoch", epoch)
                    print(preds)
                
            # compute loss
            loss = F.mse_loss(preds, current_output)
            total_loss += loss.item()

            # compute gradients using backprop
            optimizer.zero_grad() #so that the new grad do not accumulate over prev ones
            loss.backward()

            # update weights - using optimizer of choice
            optimizer.step()

           # print(nn_instance)
            pass

        if abs(prev_loss - total_loss) < 0.001:
            loss_platue_count += 1
        else:
            loss_platue_count = 0 

        if loss_platue_count >= 5:       
            print(f"epoch : {epoch}, current_loss: {total_loss}, Plataue - loss is not reducing")
            break

        if total_loss < 0.001:
            print(f"epoch : {epoch}, current_loss: {total_loss}, Loss reduced to Zero")
            break

        if epoch % 100 == 0:
            print(f"epoch : {epoch}, current_loss: {total_loss}")
            

    #torch.save(nn_instance, os.path.join(base_model_path, network_param.get_model_filename))
    filename = str(network_param.num_hidden_layers) + "_" + str(activation_function) + str(weight_decay) + ".pt"
    torch.save(nn_instance, os.path.join(base_model_path, filename))

if __name__ == '__main__':
    # Main program starts here
    parser = argparse.ArgumentParser(description = 'Bid Optimization Training Module')
    parser.add_argument("numLayers")
    parser.add_argument("numBatches")
    parser.add_argument("activationFunction")
    parser.add_argument("lr")
    parser.add_argument("weightDecay")

    args = parser.parse_args()
    num_layers = int(args.numLayers)
    num_batches = int(args.numBatches)
    activation_function = args.activationFunction
    lr = float(args.lr)
    weight_decay = float(args.weightDecay)

    input, output = get_training_data()
    input_length = input.shape[1]
    output_length = output.shape[1]

    if not google_cloud:
        num_layers = 1
        num_batches = 1
        lr = 0.000001
        activation_function = 'sigmoid'
        weight_decay = 1

    param = NetworkParam(input_length, output_length, num_layers)
    train_model(input, output, num_batches, activation_function, param, lr, weight_decay)