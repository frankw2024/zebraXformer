from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import os as os
from scipy import stats
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import os
import logging

# Configure logging
logging.basicConfig(filename='ZBXformer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# trajectorytools needs to be installed. To install,
# pip install trajectorytools or follow the instructions at
# https://gitlab.com/polavieja_lab/trajectorytools
import trajectorytools as tt
from sklearn.metrics import confusion_matrix, classification_report
#from dataprep import fish_data, plot1

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


common_path = ".\\data\\"
end_path = "\\trajectories\\without_gaps.npy"

def list_files_in_directory(directory):
    try:
        # List all files in the given directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return files
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        return []

def list_dirs_in_directory(directory):
    try:
        # List all directories in the given directory
        dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        return dirs
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        return []
    
# Loop through each fish to calculate and store their data
sp_nbins = np.linspace(0, 15, 100)  # ADJUST THE RANGE BINS TO YOUR DATA
#acl_nbins = np.linspace(0, 150, 100)
acl_nbins = np.linspace(0, 5, 100)
#dist_nbins = np.linspace(0, np.nanmax(tr.distance_to_origin), 100)
hard_max_dist = 3000
#curv_nbins = np.arange(-3, 3, 0.02)
curv_nbins = np.arange(-0.5, 0.5, 0.02)
dist_nbins = np.linspace(0, hard_max_dist, 100)
hard_max_bdist = 800 #hard_max_dist 
burstdist_nbins = np.linspace(0, hard_max_bdist, 100)
len_metrics = 15+150 + hard_max_dist +3+150

def pre_dataset1(tr):

    # Initialize a dictionary to store data for each fish
    fish_data = {}
    fish_metrics = {}

    frame_rate = tr.params["frame_rate"]

    for j in range(tr.number_of_individuals):
        fish_id = j  # or any unique identifier for the fish
        kbe_array = np.nan_to_num(tr.speed[:, j]) * np.nan_to_num(tr.acceleration[:, j])
        #burst_distances = tr.speed[:, j]
        #burstDist = np.where(tr.speed[:, j] < 2, 0, tr.speed[:, j])
        burstDist = np.cumsum(np.where(tr.speed[:, j] < 2, 0, tr.speed[:, j])) / frame_rate
        fish_data[fish_id] = {
            'ori_fish_id': j,
            'speed': np.histogram(tr.speed[:, j], bins=sp_nbins, density=True)[0],
            'distance': np.histogram(tr.distance_to_origin[:, j], bins=dist_nbins, density=True)[0],
            'burstDistance': np.histogram(burstDist, bins=burstdist_nbins, density=True)[0],
            'acceleration': np.histogram(tr.acceleration[:, j], bins=acl_nbins, density=True)[0],
            'curvature': np.histogram(tr.curvature[:, j], bins=curv_nbins, density=True)[0],
            'kineticEnergy': np.histogram(kbe_array, bins=acl_nbins, density=True)[0],
            #'total_distance': np.sum(tr.distance_to_origin[:, j]),
            #'burst_distance': calculate_burst_distance(tr.speed[:, j]),  # Define this function as needed
            #'total_kinetic_energy': calculate_kinetic_energy(tr.speed[:, j]),  # Define this function as needed
            'label': 'healthy'  # Define or calculate the label as needed
        }
        # Concatenate the data members into a single array
        fish_metrics[fish_id] = np.concatenate([
            fish_data[fish_id]['speed'],
            fish_data[fish_id]['acceleration'],
            fish_data[fish_id]['burstDistance'],
            fish_data[fish_id]['distance'],
            fish_data[fish_id]['curvature'],
            fish_data[fish_id]['kineticEnergy']
        ])[:, np.newaxis]  # Add a new

    hist2d: list[np.ndarray] = []

    for focal in range(tr.number_of_individuals):
        hist2d.append(
            np.histogram2d(
                tr.s[:, focal, 0][~np.isnan(tr.s[:, focal, 0])],
                tr.s[:, focal, 1][~np.isnan(tr.s[:, focal, 1])],
                25,
            )[0]
        )
    vmax = np.asarray(hist2d).max()
    min_x, max_x = np.nanmin(tr.s[..., 0]), np.nanmax(tr.s[..., 0])
    min_y, max_y = np.nanmin(tr.s[..., 1]), np.nanmax(tr.s[..., 1])

    return fish_metrics, fish_data, vmax, min_x, max_x, min_y, max_y
# combine two fish metrics with the same number of individuals
def combine_datasets(tr, fm1, fm2):
    fish_metrics = {}
    trni = tr.number_of_individuals
    for j in range(tr.number_of_individuals):
        fish_id = j  # or any unique identifier for the fish
        # Concatenate the data members into a single array
        fish_metrics[fish_id] = fm1[fish_id]
        
    for j in range(tr.number_of_individuals):
        fish_id = j  # or any unique identifier for the fish
        # Concatenate the data members into a single array
        
        fish_metrics[fish_id+trni] = fm2[fish_id]

    return fish_metrics
# append fm2 to fm1, both of which can have any length
def append_datasets(tr, fm1, fm2):
    fish_metrics = {}
    trni = 0
    cnt = len(fm1)
    for j in range(cnt):
        fish_id = j  # or any unique identifier for the fish
        # Concatenate the data members into a single array
        fish_metrics[fish_id] = fm1[fish_id]
        trni += 1

    cnt2 = len(fm2)    
    for j in range(cnt2):
        fish_id = j  # or any unique identifier for the fish
        # Concatenate the data members into a single array
        fish_metrics[fish_id+trni] = fm2[fish_id]

    return fish_metrics

def plot1(i, fish_data, savefile="6metrics"):
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Fish {i + 1} Data Visualizations")

    # Access the data for the specific fish
    fish = fish_data[i]

    # Plot speed histogram
    axs[0, 0].bar(sp_nbins[:-1], fish['speed'], width=np.diff(sp_nbins), align='edge')
    axs[0, 0].set(title="Speed ", xlabel="Speed (BL/s)", ylabel="Density")

    # Plot acceleration histogram
    axs[0, 1].bar(acl_nbins[:-1], fish['acceleration'], width=np.diff(acl_nbins), align='edge')
    axs[0, 1].set(title="Acceleration ", xlabel="Acceleration (BL/s²)", ylabel="Density")

    # Plot distance to origin histogram
    axs[0, 2].bar(dist_nbins[:-1], fish['distance'], width=np.diff(dist_nbins), align='edge')
    axs[0, 2].set(title="Distance to Origin ", xlabel="Distance (BL)", ylabel="Density")
    

    # Plot curvature histogram
    axs[1, 0].bar(curv_nbins[:-1], fish['curvature'], width=np.diff(curv_nbins), align='edge')
    axs[1, 0].set(title="Curvature", xlabel="Curvature (1/BL)", ylabel="Density")

    axs[1, 2].bar(sp_nbins[:-1], fish['kineticEnergy'], width=np.diff(sp_nbins), align='edge')
    axs[1, 2].set(title="Kinetic Energy ", xlabel="Kinetic Energy (Jourel)", ylabel="Density")

    # plot the burst distance histogram
    axs[1, 1].bar(burstdist_nbins[:-1], fish['burstDistance'], width=np.diff(burstdist_nbins), align='edge')
    axs[1, 1].set(title="Burst Distance ", xlabel="Burst Distance (BL)", ylabel="Density")
    


    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # Save the plot to a file
    plt.savefig(savefile+".png")  # Save as a PNG file. You can change the extension to .pdf, .svg, etc.



# Custom dataset: Used for self-supervised pretraining (no labels)
class TrajectoryDatasetSSL(Dataset):
    def __init__(self, trajectories):
        """
        trajectories: A list containing multiple trajectory sequences,
                     each element is a numpy array (T_i, F)
        """
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Return the tensor corresponding to the trajectory sequence (seq_len, feature_dim)
        traj = self.trajectories[idx]
        # Convert to PyTorch tensor
        return torch.tensor(traj, dtype=torch.float32)

# Positional encoding module (Sin-Cos format)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a positional encoding matrix (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        # Compute Sinusoid encoding according to the formula
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # Odd dimensions
        pe = pe.unsqueeze(0)  # Add batch dimension -> (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x dimensions: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Add positional encoding to input (without changing gradients because pe is registered as a buffer)
        x = x + self.pe[:, :seq_len]
        return x

# Define the trajectory Transformer encoder
class TrajectoryEncoder(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        """
        feature_dim: Input feature dimension (e.g., 5, including x, y, speed, accel, turn_angle)
        d_model: Internal embedding dimension of the Transformer model
        nhead: Number of attention heads
        num_layers: Number of stacked TransformerEncoderLayers
        """
        super(TrajectoryEncoder, self).__init__()
        self.d_model = d_model
        # Map input features to d_model dimension
        self.input_proj = nn.Linear(feature_dim, d_model)
        # Define positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        # Define Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):         
        """
        Input: x tensor, shape (batch_size, seq_len, feature_dim)
        Output: Encoded representation, shape (batch_size, seq_len, d_model)
        """
        # Project to model dimension and scale
        x_emb = self.input_proj(x) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        # Add positional encoding
        x_emb = self.pos_encoder(x_emb)
        # Transformer expects input shape as (seq_len, batch, d_model), so transpose
        x_emb = x_emb.transpose(0, 1)
        # Forward propagation through the encoder
        encoded = self.encoder(x_emb)  # (seq_len, batch, d_model)
        # Transpose back to (batch, seq_len, d_model)
        encoded = encoded.transpose(0, 1)
        return encoded

# Define Masked Autoencoder model (uses TrajectoryEncoder + linear decoder head)
class MaskedAutoencoderModel(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2):
        super(MaskedAutoencoderModel, self).__init__()
        self.encoder = TrajectoryEncoder(feature_dim, d_model, nhead, num_layers)
        # Decoder head: Maps d_model dimension output from the encoder back to original feature dimension
        self.decoder_head = nn.Linear(d_model, feature_dim)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        encoded = self.encoder(x)            # (batch, seq_len, d_model)
        reconstructed = self.decoder_head(encoded)  # (batch, seq_len, feature_dim)
        return reconstructed

# 自定义数据集：用于有监督的分类训练
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, labels):
        """
        trajectories: 轨迹序列列表，每个元素是 numpy array (T, F)
        labels: 与轨迹对应的标签列表（0或1）
        """
        self.trajectories = trajectories
        self.labels = labels
    def __len__(self):
        return len(self.trajectories)
    def __getitem__(self, idx):
        X = torch.tensor(self.trajectories[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return X, y

# 定义分类模型：包含预训练好的编码器和一个分类头
class TrajectoryClassifier(nn.Module):
    def __init__(self, encoder, d_model, num_classes=2):
        super(TrajectoryClassifier, self).__init__()
        self.encoder = encoder  # 传入预训练的编码器 (TrajectoryEncoder)
        self.classifier_head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        输出: (batch, num_classes) 的 logits
        """
        # 获取编码器输出 (batch, seq_len, d_model)
        with torch.no_grad():
            # 我们在初始化时可以选择冻结编码器参数：
            # 这里forward用 no_grad 是为了演示冻结编码器，如果需要微调，可去掉该上下文管理器
            encoded_seq = self.encoder(x)
        # 简单地对时间序列维度做平均，以获取整个序列的表示
        seq_repr = encoded_seq.mean(dim=1)  # (batch, d_model)
        logits = self.classifier_head(seq_repr)  # (batch, num_classes)
        return logits

def build_dataset(case_names):
    #change this to the path to you  data, e.g., common_path = ".\\ground_truth_std\\"
    # the data directory should be like this:
    # common_path
    # ├── healthy
    # │   ├── c52
    # │   ├── c54
    # │   └── c89
    # └── severe    
    
    
    fish_metrics1n = []
    for case_name in case_names:
        trajectories_path1 =  common_path + case_name + end_path
        tr1 = tt.Trajectories.from_idtrackerai(
            trajectories_path1, interpolate_nans=True, smooth_params={"sigma": 1}
        )
        fish_metrics1, fish_data, vmax, min_x, max_x, min_y, max_y = pre_dataset1(tr1)
        fish_metrics1n = append_datasets(tr1, fish_metrics1n, fish_metrics1)
    return fish_metrics1n

def eval_model(classifier_model, test_loader):
    classifier_model.to(device)
    # 模型评估（在测试集上）
    classifier_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)  # Move input batch to GPU
            y_batch = y_batch.to(device)  # Move input batch to GPU
            logits = classifier_model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())


    # Specify the expected labels
    expected_labels = [0, 1, 2]  # Assuming 0 is "Normal" and 1 is "Diabetic", and 2 is "Moderate"


    # 计算混淆矩阵和分类报告
    # Check the unique classes in all_labels
    #unique_classes = np.unique(all_labels)
    # Calculate confusion matrix and classification report
    unique_classes = np.unique(all_labels + all_preds)  # Check unique classes in both labels and predictions


    # Adjust target_names based on the unique classes
    #'''
    if len(unique_classes) == 1:
        tgt_names = ["Healthy"] if unique_classes[0] == 0 else ["Severe Diabetic"]
    elif len(unique_classes) == 2:
        tgt_names = ["Healthy", "Severe Diabetic"] if unique_classes[0] == 0 else ["Moderate Diabetic", "Severe Diabetic"]
    else:
        tgt_names = ["Healthy", "Severe Diabetic", "Moderate Diabetic"]
    #'''
    #tgt_names = ["healthy", "heavy"]

    cm = confusion_matrix(all_labels, all_preds, labels=expected_labels)
    report = classification_report(all_labels, all_preds, target_names=tgt_names, zero_division=0)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    print("End of evaluation")
    
    # Log the results instead of printing
    logging.info("Confusion Matrix:\n%s", cm)
    logging.info("Classification Report:\n%s", report)
    logging.info("End of evaluation")

print("End of util")