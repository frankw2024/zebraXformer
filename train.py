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

from util import MaskedAutoencoderModel, TrajectoryClassifier, plot1, pre_dataset1
from util import eval_model, TrajectoryDataset, append_datasets, build_dataset
from util import TrajectoryDatasetSSL, PositionalEncoding, TrajectoryEncoder
from util import MaskedAutoencoderModel, TrajectoryClassifier, list_dirs_in_directory

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

#common_path = ".\\ground_truth_std\\"
common_path = ".\\data\\"
end_path = "\\trajectories\\without_gaps.npy"

case_names = ["healthy\\c52", "healthy\\c54", "healthy\\c89","severe\\e86", "severe\\e82", "severe\\e85", "moderate\\e71", "moderate\\e75", "moderate\\e74"]
fish_metrics = build_dataset(case_names)
hnn, snn, mnn = 30, 30, 30
total_nn = 90

def build_1class_dataset(class_name):

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
        filename = mydir + file + end_path
        tr = tt.Trajectories.from_idtrackerai(
            filename, interpolate_nans=True, smooth_params={"sigma": 1}
        )
        fish_metrics1, fish_data, vmax, min_x, max_x, min_y, max_y = pre_dataset1(tr)
        fish_metrics1n = append_datasets(tr, fish_metrics1n, fish_metrics1)
        i += 1
    nn = len(fish_metrics1n)
    return nn, fish_metrics1n

def build_dataset_all():

    details = False #True
    hnn, hfish_metrics1n = build_1class_dataset("healthy")
    print(f"Built dataset for healthy fish, totaling {hnn} fish")
    logging.info(f"Built dataset for healthy fish, totaling {hnn} fish")

    snn, sfish_metrics1n = build_1class_dataset("severe")
    print(f"Built dataset for severely diabetic fish, totaling {snn} fish")
    logging.info(f"Built dataset for severely diabetic fish, totaling {snn} fish")

    mnn, mfish_metrics1n = build_1class_dataset("moderate")
    print(f"Built dataset for moderately diabetic fish, totaling {mnn} fish")
    logging.info(f"Built dataset for moderately diabetic fish, totaling {mnn} fish")   

    trajectories_path2 = common_path +  "severe\\e85\\"+end_path
    tr = tt.Trajectories.from_idtrackerai(
        trajectories_path2, interpolate_nans=True, smooth_params={"sigma": 1}
    )   
    fish_metrics1all = append_datasets(tr, hfish_metrics1n, sfish_metrics1n)
    fish_metrics2all = append_datasets(tr, fish_metrics1all, mfish_metrics1n)
    total_nn = hnn + snn + mnn

    return hnn, snn, mnn, total_nn, fish_metrics2all

#hnn, snn, mnn, total_nn, fish_metrics = build_dataset_all()

# ---------------- 自监督预训练过程 ----------------
# 初始化MAE模型
feature_dim = 1  # 输入特征维度
#feature_dim = 2  # 输入特征维度
# 准备示例轨迹数据列表（使用随机数据模拟多条轨迹）
# 在实际应用中，这里应该使用真实的轨迹数据数组
np.random.seed(0)
#trajectories = [np.random.rand(100, feature_dim) for _ in range(20)]  # 20 条随机轨迹，每条100步，5维特征
trajectories = fish_metrics
dataset_ssl = TrajectoryDatasetSSL(trajectories)
dataloader_ssl = DataLoader(dataset_ssl, batch_size=4, shuffle=True)
mae_model = MaskedAutoencoderModel(feature_dim=feature_dim, d_model=72, nhead=6, num_layers=2)
mae_model.to(device)
mae_model.train()

# 定义优化器和损失函数
optimizer = torch.optim.Adam(mae_model.parameters(), lr=1e-4)
#optimizer = torch.optim.Adam(mae_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

#num_epochs = 5
num_epochs = 1000 #500
mask_ratio = 0.3  # 遮蔽30%的时间步
maemodel_saved = False
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_data in dataloader_ssl:
        # batch_data: (batch, seq_len, feature_dim)
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        # 在张量上随机选择部分时间步进行遮蔽
        batch_size, seq_len, feat_dim = batch_data.shape
        # 生成mask矩阵：True表示遮蔽的位置
        mask = torch.rand(batch_size, seq_len) < mask_ratio  # (batch, seq_len)
        # 制作被遮蔽的输入：复制原始数据
        input_masked = batch_data.clone()
        # 将遮蔽位置的特征设置为0（或其他标记值；这里简单使用0）
        input_masked[mask] = 0.0
        # 前向传播重建
        output_recon = mae_model(input_masked)  # (batch, seq_len, feature_dim)
        # 计算损失：只在被遮蔽的位置计算MSE误差
        # 我们需要将 output_recon 和 batch_data 展平成二维，再应用掩码
        # 方式一：直接按元素计算所有位置的MSE，再用掩码筛选
        loss = criterion(output_recon[mask], batch_data[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
    avg_loss = total_loss / len(dataset_ssl)
    if num_epochs < 50 or epoch % 40 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Pretrain MSE Loss = {avg_loss:.4f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Pretrain MSE Loss = {avg_loss:.4f}")
    if avg_loss < 0.01:
    #if avg_loss < 0.005:
    #if avg_loss < 0.014:
        print(f"Epoch {epoch+1}/{num_epochs}, Pretrain MSE Loss = {avg_loss:.4f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Pretrain MSE Loss = {avg_loss:.4f}")
        # Save the MAE model
        torch.save(mae_model.state_dict(), 'mae_model.pth')
        maemodel_saved = True
        break

if maemodel_saved == False:
    torch.save(mae_model.state_dict(), 'mae_model.pth')

from sklearn.metrics import confusion_matrix, classification_report



# 创建示例数据集（这里仍用随机数据模拟，假设前10条正常(0)，后10条糖尿病(1)）
#ll = tr.s.shape[0] //2
#labels = [0]*ll + [1]*ll
#labels = [0]*30 + [1]*30 + [2]*30
labels = [0]*hnn + [1]*snn + [2]*mnn

dataset_cls = TrajectoryDataset(trajectories, labels)
# 划分训练集和测试集
train_size = int(0.8 * len(dataset_cls))
test_size = len(dataset_cls) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset_cls, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

# 初始化分类模型，并加载预训练的编码器权重
pretrained_encoder = mae_model.encoder  # 从之前训练的MAE模型获取编码器
#classifier_model = TrajectoryClassifier(encoder=pretrained_encoder, d_model=72, num_classes=2)
classifier_model = TrajectoryClassifier(encoder=pretrained_encoder, d_model=72, num_classes=3)

# （可选）如果希望微调编码器，可将 encoder 的参数 requires_grad 设置为 True
for param in classifier_model.encoder.parameters():
    param.requires_grad = True  # 如果需要微调预训练编码器，则设为 True

# 定义优化器和损失函数（交叉熵用于二分类）
#learning_rate = 1e-1
learning_rate = 1e-3 #5e-3
#learning_rate = 1e-4
optimizer_cls = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate)
criterion_cls = nn.CrossEntropyLoss()

# 监督训练循环
classifier_model.to(device)  # Move model to GPU
classifier_model.train()
num_epochs_cls = 1000 #500 #5
#num_epochs_cls = 15
classifier_model_saved = False
for epoch in range(num_epochs_cls):
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)  # Move input batch to GPU
        y_batch = y_batch.to(device)  # Move input batch to GPU
        optimizer_cls.zero_grad()
        logits = classifier_model(X_batch)        # (batch, 2)
        loss = criterion_cls(logits, y_batch)     # 计算交叉熵损失
        loss.backward()
        optimizer_cls.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_dataset)
    # 简单计算训练集上的准确率
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)  # Move input batch to GPU
        y_batch = y_batch.to(device)  # Move input batch to GPU
        preds = classifier_model(X_batch).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    train_acc = correct / total
    if num_epochs_cls < 50 or epoch % 40 == 0:
        print(f"Epoch {epoch+1}/{num_epochs_cls}, Train Loss = {avg_loss:.4f}, Train Acc = {train_acc:.2f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs_cls}, Train Loss = {avg_loss:.4f}, Train Acc = {train_acc:.2f}")
    #if train_acc > 0.87:
    if train_acc > 0.90:
        print(f"Epoch {epoch+1}/{num_epochs_cls}, Train Loss = {avg_loss:.4f}, Train Acc = {train_acc:.2f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs_cls}, Train Loss = {avg_loss:.4f}, Train Acc = {train_acc:.2f}")
        # Save the classifier model
        torch.save(classifier_model.state_dict(), 'classifier_model.pth')
        break

if classifier_model_saved == False:
    torch.save(classifier_model.state_dict(), 'classifier_model.pth')

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
if len(unique_classes) == 1:
    tgt_names = ["Healthy"] if unique_classes[0] == 0 else ["Severe Diabetic"]
elif len(unique_classes) == 2:
    tgt_names = ["Healthy", "Severe Diabetic"] if unique_classes[0] == 0 else ["Severe Diabetic", "Moderate Diabetic"]
else:
    tgt_names = ["Healthy", "Severe Diabetic", "Moderate Diabetic"]

cm = confusion_matrix(all_labels, all_preds, labels=expected_labels)
report = classification_report(all_labels, all_preds, target_names=tgt_names)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

dataset_cls = TrajectoryDataset(trajectories, labels)
# test all data
all_test_loader = DataLoader(dataset_cls, batch_size=6, shuffle=False)
eval_model(classifier_model, all_test_loader)


# Initialize the MAE model architecture
mae_model1 = MaskedAutoencoderModel(feature_dim=feature_dim, d_model=72, nhead=6, num_layers=2)
mae_model1.load_state_dict(torch.load('mae_model.pth'))

pretrained_encoder = mae_model1.encoder 
# Initialize the classifier model architecture
classifier_model1 = TrajectoryClassifier(encoder=pretrained_encoder, d_model=72, num_classes=3)
classifier_model1.load_state_dict(torch.load('classifier_model.pth'))
classifier_model = classifier_model1
# Ensure the new classifier model is on the correct device
classifier_model.to(device)

#trajectories = proc_data()
#labels = [0]*30 + [1]*30 + [2]*30
#dataset_cls = TrajectoryDataset(trajectories, labels)

# test all data
all_test_loader = DataLoader(dataset_cls, batch_size=6, shuffle=False)
eval_model(classifier_model, all_test_loader)

print("End of training")

# Load new fish trajectory
trajectories_path2 =  common_path + "severe\\e85\\trajectories\\without_gaps.npy"

tr = tt.Trajectories.from_idtrackerai(
    trajectories_path2, interpolate_nans=True, smooth_params={"sigma": 1}
)
fish_metrics2, fish_data, vmax, min_x, max_x, min_y, max_y = pre_dataset1(tr)
#plot1(0, fish_data)

for j in range(tr.number_of_individuals):
    new_features = fish_metrics2[j]

    # Convert to PyTorch tensor
    X_new = torch.tensor(new_features, dtype=torch.float32).unsqueeze(0)
    X_new = X_new.to(device)

    # Predict
    classifier_model.eval()
    with torch.no_grad():
        logits = classifier_model(X_new)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label].item()

    #print(f"Prediction: {'Diabetic' if pred_label == 1 else 'Normal'}, Confidence: {confidence:.2f}")
    print(f"Fish {j} Prediction: {'Severe Diabetic' if pred_label == 1 else 'Moderate Diabetic' if pred_label == 2 else 'Healthy'}")

 # Initialize the MAE model architecture
mae_model1 = MaskedAutoencoderModel(feature_dim=feature_dim, d_model=72, nhead=6, num_layers=2)
mae_model1.load_state_dict(torch.load('mae_model.pth'))

pretrained_encoder = mae_model1.encoder 
# Initialize the classifier model architecture
classifier_model1 = TrajectoryClassifier(encoder=pretrained_encoder, d_model=72, num_classes=3)
classifier_model1.load_state_dict(torch.load('classifier_model.pth'))

for j in range(tr.number_of_individuals):
    new_features = fish_metrics2[j]

    # Convert to PyTorch tensor
    X_new = torch.tensor(new_features, dtype=torch.float32).unsqueeze(0)

    # Predict
    classifier_model1.eval()
    with torch.no_grad():
        logits = classifier_model1(X_new)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label].item()

    #print(f"Prediction: {'Diabetic' if pred_label == 1 else 'Normal'}, Confidence: {confidence:.2f}")
    print(f"Fish {j} Prediction: {'Severe Diabetic' if pred_label == 1 else 'Moderate Diabetic' if pred_label == 2 else 'Healthy'}")


print("End of inference")