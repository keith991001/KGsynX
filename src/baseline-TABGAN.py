import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    import xgboost as xgb



def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    print(f"数据列名: {list(df.columns)}")

    df = df.dropna()
    print(f"处理缺失值后数据形状: {df.shape}")
    

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    if y.dtype == 'object':
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y)
    else:
        y_encoder = None
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, label_encoders, y_encoder


class Generator(nn.Module):

    def __init__(self, noise_dim, data_dim, hidden_dim=256):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, data_dim),
            nn.Tanh()  
        )
    
    def forward(self, noise):
        return self.model(noise)

class Discriminator(nn.Module):
    def __init__(self, data_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.data_dim = data_dim
        
        self.model = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data):
        return self.model(data)

class TabGAN:
    def __init__(self, data_dim, noise_dim=100, device='cpu'):
        self.device = device
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        
        self.generator = Generator(noise_dim, data_dim).to(device)
        self.discriminator = Discriminator(data_dim).to(device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        
    def train(self, dataloader, epochs=100):        
        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            
            for batch_idx, (real_data, _) in enumerate(dataloader):
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)

                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                self.d_optimizer.zero_grad()

                real_output = self.discriminator(real_data)
                d_real_loss = self.criterion(real_output, real_labels)

                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(noise)

                fake_output = self.discriminator(fake_data.detach())
                d_fake_loss = self.criterion(fake_output, fake_labels)

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                self.g_optimizer.zero_grad()

                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, real_labels)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
            
            if (epoch + 1) % 20 == 0:
                avg_d_loss = np.mean(d_losses)
                avg_g_loss = np.mean(g_losses)
                print(f"Epoch [{epoch+1}/{epochs}], D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
    
    def generate(self, num_samples):
        print(f"生成 {num_samples} 个合成样本...")
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim).to(self.device)
            synthetic_data = self.generator(noise)
            
        return synthetic_data.cpu().numpy()

def evaluate_downstream_task(X_real, y_real, X_synthetic, y_synthetic):

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )
    
    print(f"真实数据分割: 训练集 {X_real_train.shape[0]} 样本, 测试集 {X_real_test.shape[0]} 样本")
    print(f"合成数据: {X_synthetic.shape[0]} 样本")
    
    xgb_real = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_real.fit(X_real_train, y_real_train)
    
    y_pred_real = xgb_real.predict(X_real_test)
    y_prob_real = xgb_real.predict_proba(X_real_test)
    
    acc_real = accuracy_score(y_real_test, y_pred_real)
    precision_real = precision_score(y_real_test, y_pred_real, average='weighted', zero_division=0)
    recall_real = recall_score(y_real_test, y_pred_real, average='weighted', zero_division=0)
    f1_real = f1_score(y_real_test, y_pred_real, average='weighted', zero_division=0)
    
    n_classes = len(np.unique(y_real))
    if n_classes == 2:
        auc_real = roc_auc_score(y_real_test, y_prob_real[:, 1])
    else:
        try:
            auc_real = roc_auc_score(y_real_test, y_prob_real, multi_class='ovr', average='weighted')
        except:
            auc_real = 0.0
    
    xgb_synthetic = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_synthetic.fit(X_synthetic, y_synthetic)
    
    y_pred_synthetic = xgb_synthetic.predict(X_real_test)
    y_prob_synthetic = xgb_synthetic.predict_proba(X_real_test)
    
    acc_synthetic = accuracy_score(y_real_test, y_pred_synthetic)
    precision_synthetic = precision_score(y_real_test, y_pred_synthetic, average='weighted', zero_division=0)
    recall_synthetic = recall_score(y_real_test, y_pred_synthetic, average='weighted', zero_division=0)
    f1_synthetic = f1_score(y_real_test, y_pred_synthetic, average='weighted', zero_division=0)
    
    if n_classes == 2:
        auc_synthetic = roc_auc_score(y_real_test, y_prob_synthetic[:, 1])
    else:
        try:
            auc_synthetic = roc_auc_score(y_real_test, y_prob_synthetic, multi_class='ovr', average='weighted')
        except:
            auc_synthetic = 0.0

    print(f"\n{'='*60}")
    print(f"{'指标':<15} {'真实数据':<15} {'合成数据':<15} {'性能保持率':<15}")
    print(f"{'='*60}")
    print(f"{'Accuracy':<15} {acc_real:<15.4f} {acc_synthetic:<15.4f} {acc_synthetic/acc_real:<15.4f}")
    print(f"{'Precision':<15} {precision_real:<15.4f} {precision_synthetic:<15.4f} {precision_synthetic/precision_real:<15.4f}")
    print(f"{'Recall':<15} {recall_real:<15.4f} {recall_synthetic:<15.4f} {recall_synthetic/recall_real:<15.4f}")
    print(f"{'F1-Score':<15} {f1_real:<15.4f} {f1_synthetic:<15.4f} {f1_synthetic/f1_real:<15.4f}")
    print(f"{'AUC':<15} {auc_real:<15.4f} {auc_synthetic:<15.4f} {auc_synthetic/auc_real if auc_real > 0 else 0:<15.4f}")
    print(f"{'='*60}")
    
    print(classification_report(y_real_test, y_pred_real))
    
    print(classification_report(y_real_test, y_pred_synthetic))
    
    # 返回所有指标
    metrics = {
        'real': {
            'accuracy': acc_real,
            'precision': precision_real,
            'recall': recall_real,
            'f1': f1_real,
            'auc': auc_real
        },
        'synthetic': {
            'accuracy': acc_synthetic,
            'precision': precision_synthetic,
            'recall': recall_synthetic,
            'f1': f1_synthetic,
            'auc': auc_synthetic
        }
    }
    
    return metrics

def main():
    # 数据路径
    data_path = "xxx"

    X, y, scaler, label_encoders, y_encoder = load_and_preprocess_data(data_path)
    
    print(f"处理后数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    X_scaled = 2 * X - 1  # 从[0,1]映射到[-1,1]

    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    data_dim = X.shape[1]
    noise_dim = max(100, data_dim)  # 噪声维度至少为100
    tabgan = TabGAN(data_dim, noise_dim, device)

    tabgan.train(dataloader, epochs=100)

    num_synthetic_samples = len(X)  
    X_synthetic_raw = tabgan.generate(num_synthetic_samples)
    
    X_synthetic_normalized = (X_synthetic_raw + 1) / 2  
    X_synthetic = scaler.inverse_transform(X_synthetic_normalized)
    
    X_synthetic_rescaled = scaler.fit_transform(X_synthetic)
    
    unique_labels, label_counts = np.unique(y, return_counts=True)
    label_probs = label_counts / len(y)
    y_synthetic = np.random.choice(unique_labels, size=num_synthetic_samples, p=label_probs)
    
    print(f"\n合成数据生成完成!")
    print(f"合成数据形状: X={X_synthetic_rescaled.shape}, y={y_synthetic.shape}")
    print(f"合成数据类别分布: {np.bincount(y_synthetic)}")
    
    metrics = evaluate_downstream_task(X, y, X_synthetic_rescaled, y_synthetic)
    
    synthetic_df = pd.DataFrame(X_synthetic)
    synthetic_df['target'] = y_synthetic
    synthetic_df.to_csv('/content/synthetic_data_tabgan.csv', index=False)
    print("\n合成数据已保存到: /content/synthetic_data_tabgan.csv")
    
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'Real_Data': [
            metrics['real']['accuracy'],
            metrics['real']['precision'], 
            metrics['real']['recall'],
            metrics['real']['f1'],
            metrics['real']['auc']
        ],
        'Synthetic_Data': [
            metrics['synthetic']['accuracy'],
            metrics['synthetic']['precision'],
            metrics['synthetic']['recall'], 
            metrics['synthetic']['f1'],
            metrics['synthetic']['auc']
        ]
    })
    results_df['Performance_Ratio'] = results_df['Synthetic_Data'] / results_df['Real_Data']
    results_df.to_csv('/content/tabgan_evaluation_results.csv', index=False)
    
    return X_synthetic, y_synthetic, metrics

if __name__ == "__main__":
    X_synthetic, y_synthetic, metrics = main()
