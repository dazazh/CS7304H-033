import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from preprocess import preprocess
import numpy as np
from torch.autograd import Function

data = pd.read_csv("/data2/yuhao/class/CS7304H-033/Datasets/train.csv", header=None)
test_data = pd.read_csv("/data2/yuhao/class/CS7304H-033/Datasets/test.csv", header=None)

X_train_t, y_train_t, X_val_t, y_val_t, X_test_t = preprocess(data, test_data)

# 构建DataLoader
batch_size = 64
source_dataset = TensorDataset(X_train_t, y_train_t)
source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
target_dataset = TensorDataset(X_test_t, torch.zeros(X_test_t.shape[0]).long()) # 目标域无标签
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

class EarlyStopping:
    def __init__(self, patience, save_path_domain, save_path_cls):
        self.patience = patience  # 连续满足条件的次数
        self.counter = 0  # 当前满足条件的计数
        self.best_val_acc = 0
        self.save_path_domain = save_path_domain  # 模型保存路径
        self.save_path_cls = save_path_cls
        self.should_stop = False  # 是否应该提前停止

    def check_early_stop(self, model_domain,model_cls, loss):
        if loss < 0.00005:  # 验证集准确率达到100%
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered. Saving model.")
                torch.save(model_domain.state_dict(), self.save_path_domain)
                torch.save(model_cls.state_dict(), self.save_path_cls)
                self.should_stop = True
        else:
            self.counter = 0  # 重置计数

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        """
        前向传播时不做任何改变，只传递输入。
        """
        ctx.lambda_val = lambda_val  # 将 lambda_val 存入上下文
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播时，将梯度乘以负的 lambda_val，从而实现梯度反转。
        """
        lambda_val = ctx.lambda_val
        grad_input = grad_output.neg() * lambda_val  # 反转梯度并加权
        return grad_input, None  # 返回梯度和 None，因为 lambda_val 是标量

# 定义DANN网络结构
class Classifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)  # 两类: 源域(0), 目标域(1)
        )
    def forward(self, x, lambda_val=None):
        if lambda_val is not None:
            x = GradientReversalLayer.apply(x, lambda_val)
        return self.fc(x)

# 若您在预处理中降维了(例如PCA到D=50)，则input_dim应为50而不是512
input_dim = X_train_t.shape[1]
num_classes = len(np.unique(y_train_t))  # 类别数
hidden_dim = 256
classifier = Classifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes).cuda()
domain_disc = DomainDiscriminator(input_dim=input_dim, hidden_dim=hidden_dim).cuda()
early_stopping = EarlyStopping(patience=20, save_path_domain="/data2/yuhao/class/CS7304H-033/DANN/model/domain.ckpt",save_path_cls="/data2/yuhao/class/CS7304H-033/DANN/model/cls.ckpt")

criterion_label = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

optimizer = optim.Adam(list(classifier.parameters())+list(domain_disc.parameters()), lr=1e-5)

num_epochs = 300
len_dataloader = min(len(source_loader), len(target_loader))

# 提示：DANN通常需要Gradient Reversal Layer，这里用简单方法模拟
def grad_reverse(x, alpha=1.0):
    # 一个简易的梯度反转操作，可使用 autograd Function 实现更优雅的操作
    return x * 1.0

for epoch in range(num_epochs):
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    
    classifier.train()
    domain_disc.train()

    for i in range(len_dataloader):
        X_s_batch, y_s_batch = next(source_iter)
        X_t_batch, _ = next(target_iter)
        
        X_s_batch = X_s_batch.cuda()
        y_s_batch = y_s_batch.cuda()
        X_t_batch = X_t_batch.cuda()
        
        # 动态调整 lambda_val
        p = float(i + epoch * len_dataloader) / (num_epochs * len_dataloader)
        lambda_val = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        
        # 源域标签预测
        y_pred = classifier(X_s_batch)
        loss_label = criterion_label(y_pred, y_s_batch)
        
        # 域分类
        X_concat = torch.cat([X_s_batch, X_t_batch], dim=0)
        domain_labels = torch.cat([
            torch.zeros(X_s_batch.size(0)),
            torch.ones(X_t_batch.size(0))
        ], dim=0).long().cuda()
        
        domain_output = domain_disc(X_concat, lambda_val=lambda_val)
        loss_domain = criterion_domain(domain_output, domain_labels)
        
        # 损失计算
        loss = loss_label + lambda_val * loss_domain
        # print("domain_loss:",loss_domain)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证集准确率
    classifier.eval()
    with torch.no_grad():
        X_val_cuda = X_val_t.cuda()
        y_val_cuda = y_val_t.cuda()
        y_val_pred = classifier(X_val_cuda)
        val_pred_labels = torch.argmax(y_val_pred, dim=1)
        val_acc = (val_pred_labels == y_val_cuda).float().mean().item()
    print(f"Epoch: {epoch+1}/{num_epochs}, Val Acc: {val_acc:.4f}, Loss: {loss:.6f}")
    early_stopping.check_early_stop(domain_disc, classifier, loss)

    if early_stopping.should_stop:
        print("Training stopped early.")
        break

# 最终在测试集上预测（无标签）
classifier.eval()
with torch.no_grad():
    X_test_cuda = X_test_t.cuda()
    y_test_pred = classifier(X_test_cuda)
    test_pred_labels = torch.argmax(y_test_pred, dim=1).cpu().numpy()

result_df = pd.DataFrame({
    'Id': np.arange(0, len(y_test_pred)),  # Id是测试集的行号
    'Label': test_pred_labels  # 将预测结果转换为numpy数组
})

# 此时test_pred_labels为DANN对平滑后特征的预测结果
# 如果有测试集标签，可以计算accuracy，否则仅输出预测结果用于提交
# 保存为CSV文件
result_df.to_csv('dann_predictions.csv', index=False)

print("测试集预测结果已保存为 'dann_predictions.csv'")