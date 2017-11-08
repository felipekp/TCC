'''
Script to create several different configuration files..
Could be just a function called run(parameters**) where all parameters would be defined in the caller.
'''
import json

data = {
    "data": {
        "filename": "out/pca.csv",
        "target_col": 25,
        "lead_time": 2,
        "train_test_split": 0.8
    },
    "model": {
        "epochs": 0,
        "look_back": 8,
        "batch_size": 512,
        "input_nodes": 50,
        "loss_function": "mse",
        "optimiser_function": "nadam",
        "testtraining_loss_graph": True,
    }
}

data['model']['epochs'] = 10
for epoch in range(0,500,10):

    with open('configuration_files/test.json', 'w+') as fp: json.dump(data, fp, indent=4)