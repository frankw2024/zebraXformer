from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import os as os
from scipy import stats
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import logging

# Configure logging
# Move the current log file to a new name
# Move the current log file to a new name
if os.path.exists('ZBXformer1.log'):
    if os.path.exists('ZBXformer2.log'):
        os.remove('ZBXformer2.log')  # Remove the existing ZBXformer2.log
    os.rename('ZBXformer1.log', 'ZBXformer2.log')
if os.path.exists('ZBXformer.log'):
    if os.path.exists('ZBXformer1.log'):
        os.remove('ZBXformer1.log')  # Remove the existing ZBXformer2.log
    os.rename('ZBXformer.log', 'ZBXformer1.log')
logging.basicConfig(filename='ZBXformer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# trajectorytools needs to be installed. To install,
# pip install trajectorytools or follow the instructions at
# https://gitlab.com/polavieja_lab/trajectorytools
import trajectorytools as tt
#from dataprep import fish_data, plot1
from util import MaskedAutoencoderModel, TrajectoryClassifier, plot1, pre_dataset1
from util import eval_model, TrajectoryDataset, append_datasets
from util import list_files_in_directory, list_dirs_in_directory, build_dataset
# Change current working directory to './'
os.chdir('./')

# Verify the change
print("Current working directory:", os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")
    
if torch.cuda.is_available():
    print(f"GPU device number: {device}")
else:
    print("No GPU available.")


feature_dim = 1  # 输入特征维度
# Initialize the MAE model architecture
mae_model1 = MaskedAutoencoderModel(feature_dim=feature_dim, d_model=72, nhead=6, num_layers=2)
mae_model1.load_state_dict(torch.load('mae_model.pth', map_location=device))

pretrained_encoder = mae_model1.encoder 
# Initialize the classifier model architecture
classifier_model1 = TrajectoryClassifier(encoder=pretrained_encoder, d_model=72, num_classes=3)
classifier_model1.load_state_dict(torch.load('classifier_model.pth', map_location=device))
classifier_model = classifier_model1
# Ensure the new classifier model is on the correct device
classifier_model1.to(device)

case_names = ["healthy\\c52", "healthy\\c54", "healthy\\c89","severe\\e86", "severe\\e82", "severe\\e85", "moderate\\e71", "moderate\\e75", "moderate\\e74"]
trajectories = build_dataset(case_names)

labels = [0]*30 + [1]*30 + [2]*30

dataset_cls = TrajectoryDataset(trajectories, labels)

# test all data
all_test_loader = DataLoader(dataset_cls, batch_size=6, shuffle=False)
eval_model(classifier_model, all_test_loader)

common_path = ".\\data\\"

def predict_1class(class_name, predict_1 = False):

    # Specify the directory
    mydir = common_path + class_name+"\\"
    # Get the list of files
    file_list = list_dirs_in_directory(mydir)
    # Print the list of files
    print(f"{class_name} " + str(len(file_list)) + " groups: " + ", ".join(map(str, file_list)))
    logging.info(f"{class_name} " + str(len(file_list)) + " groups: " + ", ".join(map(str, file_list)))

    fish_metrics1n = []
    n = len(file_list)
    i = 0
    for file in file_list:
        filename = mydir + file + '\\trajectories\\without_gaps.npy'
        tr = tt.Trajectories.from_idtrackerai(
            filename, interpolate_nans=True, smooth_params={"sigma": 1}
        )
        fish_metrics1, fish_data, vmax, min_x, max_x, min_y, max_y = pre_dataset1(tr)
        fish_metrics1n = append_datasets(tr, fish_metrics1n, fish_metrics1)
        if predict_1 == True:
            idx = 0 if class_name == "healthy" else 1 if class_name == "severe" else 2
            nn1 = len(fish_metrics1)
            labels = [idx]*nn1
            dataset_cls1 = TrajectoryDataset(fish_metrics1, labels)
            test_loader = DataLoader(dataset_cls1, batch_size=1, shuffle=False)
            eval_model(classifier_model, test_loader)
            print(f"Predicting {class_name} fish {i} for "+ file)
            logging.info(f"Predicting {class_name}  fish {i} for "+ file)
        i += 1
    nn = len(fish_metrics1n)
    return nn, fish_metrics1n

details = False #True
hnn, hfish_metrics1n = predict_1class("healthy", details)
labels = [0]*hnn
hdataset_cls = TrajectoryDataset(hfish_metrics1n, labels)
test_loader = DataLoader(hdataset_cls, batch_size=1, shuffle=False)
eval_model(classifier_model, test_loader)
print(f"Predicting healthy fish, totaling {hnn} fish")
logging.info(f"Predicting healthy fish, totaling {hnn} fish")

snn, sfish_metrics1n = predict_1class("severe", details)
labels = [1]*snn
sdataset_cls = TrajectoryDataset(sfish_metrics1n, labels)
test_loader = DataLoader(sdataset_cls, batch_size=1, shuffle=False)
eval_model(classifier_model, test_loader)
print(f"Predicting severely diabetic fish, totaling {snn} fish")
logging.info(f"Predicting severely diabetic fish, totaling {snn} fish")

mnn, mfish_metrics1n = predict_1class("moderate", details)
labels = [2]*mnn
mdataset_cls = TrajectoryDataset(mfish_metrics1n, labels)
test_loader = DataLoader(mdataset_cls, batch_size=1, shuffle=False)
eval_model(classifier_model, test_loader)
print(f"Predicting moderately diabetic fish, totaling {mnn} fish")
logging.info(f"Predicting moderately diabetic fish, totaling {mnn} fish")   

trajectories_path2 = common_path +  "severe\\e85\\trajectories\\without_gaps.npy"
tr = tt.Trajectories.from_idtrackerai(
    trajectories_path2, interpolate_nans=True, smooth_params={"sigma": 1}
)   
fish_metrics1all = append_datasets(tr, hfish_metrics1n, sfish_metrics1n)
fish_metrics2all = append_datasets(tr, fish_metrics1all, mfish_metrics1n)
nnn = hnn + snn + mnn
labels = [0]*hnn + [1]*snn + [2]*mnn
dataset_cls_all  = TrajectoryDataset(fish_metrics2all, labels)
test_loader_all = DataLoader(dataset_cls_all, batch_size=1, shuffle=False)
eval_model(classifier_model, test_loader_all)

# Ensure the model is on the correct device
classifier_model1.to(device)
def predict_group(trajectories_path2, plotD = False, savefile="6metrics" ):
    tr = tt.Trajectories.from_idtrackerai(
        trajectories_path2, interpolate_nans=True, smooth_params={"sigma": 1}
    )
    fish_metrics2, fish_data, vmax, min_x, max_x, min_y, max_y = pre_dataset1(tr)
    if plotD == True:
        plot1(0, fish_data, savefile)


    for j in range(tr.number_of_individuals):
        new_features = fish_metrics2[j]

        # Convert to PyTorch tensor
        X_new = torch.tensor(new_features, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        classifier_model1.eval()
        with torch.no_grad():
            logits = classifier_model1(X_new)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()

        #print(f"Prediction: {'Diabetic' if pred_label == 1 else 'Normal'}, Confidence: {confidence:.2f}")
        print(f"Fish {j} Prediction: {'Severe Diabetic' if pred_label == 1 else 'Moderate Diabetic' if pred_label == 2 else 'Healthy'}")
        logging.info(f"Fish {j} Prediction: {'Severe Diabetic' if pred_label == 1 else 'Moderate Diabetic' if pred_label == 2 else 'Healthy'}")

pltgr = False

# Load new fish trajectory
trajectories_path2 = common_path +  "severe\\e85\\trajectories\\without_gaps.npy"
print(" Predicting severe diabetic fish group "+"e85")
predict_group(trajectories_path2, pltgr, "e85")

trajectories_path2 =  common_path + "severe\\e82\\trajectories\\without_gaps.npy"
print(" Predicting severe diabetic fish group "+"e82")
predict_group(trajectories_path2, pltgr, "e82")
'''
trajectories_path2 =  common_path + "severe\\e87\\trajectories\\without_gaps.npy"
print(" Predicting severe diabetic fish group "+"e87")
predict_group(trajectories_path2, pltgr, "e87")
'''
print(" Predicting moderate diabetic fish group "+"e74")
trajectories_path2 =  common_path + "moderate\\e74\\trajectories\\without_gaps.npy"
predict_group(trajectories_path2, pltgr, "e74")
print(" Predicting moderate diabetic fish group "+"e71")
trajectories_path2 =  common_path + "moderate\\e71\\trajectories\\without_gaps.npy"
predict_group(trajectories_path2, pltgr, "e71")
#print(" Predicting moderate diabetic fish group "+"e73")
#predict_group(trajectories_path2, True, "e73")

print(" Predicting moderate diabetic fish group "+"e75")
trajectories_path2 =  common_path + "moderate\\e75\\trajectories\\without_gaps.npy"
predict_group(trajectories_path2, pltgr, "e75")

# Load new fish trajectory
print(" Predicting healthy fish group "+"c54")
trajectories_path2 =  common_path + "healthy\\c54\\trajectories\\without_gaps.npy"
predict_group(trajectories_path2, pltgr, "c54")

print(" Predicting healthy fish group "+"c52")
trajectories_path2 =  common_path + "healthy\\c52\\trajectories\\without_gaps.npy"
predict_group(trajectories_path2, pltgr, "c52")

print(" Predicting healthy fish group "+"c89")
trajectories_path2 =  common_path + "healthy\\c89\\trajectories\\without_gaps.npy"
predict_group(trajectories_path2, pltgr, "c89")
'''
print(" Predicting healthy fish group "+"c56")
trajectories_path2 =  common_path + "healthy\\c56\\trajectories\\without_gaps.npy"
predict_group(trajectories_path2)
'''
print("End of inference")