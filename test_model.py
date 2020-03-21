from network import Network
from network_param import NetworkParam
from train_model import base_data_path, base_model_path, device 
from train_model import get_training_data, get_training_data_item
import openpyxl
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

def get_test_data():
    test_data_input = []
    test_data_output = []

    wb = openpyxl.load_workbook(os.path.join(base_data_path, 'Test_Data.xlsx')) 
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
            test_data_input.append(input)
            test_data_output.append(output)

    return np.array(test_data_input), np.array(test_data_output)

def _save_as_csv(output, pred, path):
    input_header = ['Date', 'CampaignName', 'Bidding strategy', 'TargetingType', 'Targeting', 
        'MatchType', 'TotalCampaignImpressions'] 
    output_header = ['Impressions', 'Clicks',  'CPC', 'OrderValue']
    pred_header = ['P-Impressions', 'P-Clicks',  'P-CPC', 'P-OrderValue']

    num_rows = output.shape[0]
    f = open(path, 'w')
    f.write(', '.join([str(x) for x in output_header]))
    f.write(", ")
    f.write(', '.join([str(x) for x in pred_header]))
    f.write('\n')

    for i in range(num_rows):
        f1 = float(output[i][0])*2000 
        f2 = float(output[i][1])*50
        f3 = float(output[i][2])*100 
        f4 = float(output[i][3])*1000 
        f5 = float(pred[i][0])*2000 
        f6 = float(pred[i][1])*50
        f7 = float(pred[i][2])*100 
        f8 = float(pred[i][3])*1000
        text = "{:10.2f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}\n".format(
                f1, f2, f3, f4, f5, f6, f7, f8
            )

        f.write(text)
    f.close()


def test_model(model_path):
    nn_instance = torch.load(model_path, map_location=device)
    nn_instance.eval()
    print(nn_instance)

    input, output = get_training_data()
    input_length = input.shape[1]
    output_length = output.shape[1]

    current_input = torch.Tensor(input).to(device)
    current_output = torch.Tensor(output).to(device)

    # compute predictions
    preds = nn_instance(current_input)
    #save
    _save_as_csv(current_output.detach().cpu().numpy(), preds.detach().cpu().numpy(), 'train-result.csv')

    # compute loss
    loss = F.mse_loss(preds, current_output)
    total_training_loss = loss.item()
    print("total_training_loss", total_training_loss)

    input, output = get_test_data()
    input_length = input.shape[1]
    output_length = output.shape[1]

    current_input = torch.Tensor(input).to(device)
    current_output = torch.Tensor(output).to(device)

    # compute predictions
    preds = nn_instance(current_input)
    #print(current_input)
    print(current_output)
    print(preds)
    #save
    _save_as_csv(current_output.detach().cpu().numpy(), preds.detach().cpu().numpy(), 'test-result.csv')

    # compute loss
    loss = F.mse_loss(preds, current_output)
    total_test_loss = loss.item()
    print("total_test_loss", total_test_loss)
    pass

if __name__ == '__main__':
    # Main program starts here
    parser = argparse.ArgumentParser(description = 'Test Bid Optimization Training Module')
    parser.add_argument("filename")

    args = parser.parse_args()
    filename = str(args.filename)

    #filename = '5_relu1e-07.pt'
    path = os.path.join(base_model_path, filename)
    test_model(path)