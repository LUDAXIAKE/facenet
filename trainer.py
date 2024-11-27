import os
import random  # 添加导入random模块
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.models import Inception_V3_Weights
from torch.amp import GradScaler, autocast  # 更新导入GradScaler和autocast用于混合精度训练

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义数据集类
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        for label, person in enumerate(os.listdir(self.root_dir)):
            person_dir = os.path.join(self.root_dir, person)
            if os.path.isdir(person_dir):
                self.label_map[label] = person
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        positive_path = random.choice([p for p, l in zip(self.image_paths, self.labels) if l == anchor_label and p != anchor_path])
        negative_path = random.choice([p for p, l in zip(self.image_paths, self.labels) if l != anchor_label])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载数据集
dataset = TripletFaceDataset(root_dir='./dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)  # 减小批量大小以减少显存占用

# 定义FaceNet模型
class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        x = self.model(x)
        x = x.logits  # 提取logits部分
        x = self.l2_norm(x, p=2, dim=1)
        return x

# 初始化模型、损失函数和优化器
model = FaceNet(embedding_size=128).to(device)  # 将模型移动到GPU
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()  # 初始化GradScaler用于混合精度训练

# 保存模型和优化器状态
def save_checkpoint(model, optimizer, scaler, epoch, batch_count, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch,
        'batch_count': batch_count,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }
    torch.save(state, filename)
    print(f'Checkpoint saved at epoch {epoch}, batch {batch_count}')

# 加载模型和优化器状态
def load_checkpoint(model, optimizer, scaler, filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        batch_count = checkpoint['batch_count']
        print(f'Checkpoint loaded from epoch {start_epoch}, batch {batch_count}')
        return start_epoch, batch_count
    else:
        print('No checkpoint found')
        return 0, 0

# 训练模型
def train_model(model, dataloader, criterion, optimizer, num_epochs=25, save_interval=200, accumulation_steps=4):
    print('Training started...')
    start_epoch, batch_count = load_checkpoint(model, optimizer, scaler)  # 加载检查点
    model.train()
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        running_loss = 0.0
        optimizer.zero_grad()
        batch_count = 0
        for i, (anchors, positives, negatives) in enumerate(dataloader):
            print(f'Batch {batch_count + 1}')
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)  # 将数据移动到GPU
            with autocast(device_type='cuda'):  # 使用autocast进行混合精度训练，并指定设备类型为'cuda'
                anchor_embeddings = model(anchors)
                positive_embeddings = model(positives)
                negative_embeddings = model(negatives)
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            scaler.scale(loss).backward()  # 使用GradScaler进行反向传播
            if (i + 1) % accumulation_steps == 0:  # 梯度累积
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item() * anchors.size(0)
            torch.cuda.empty_cache()  # 清理缓存以释放显存
            batch_count += 1  # 增加批次数量

            # 每处理save_interval*20批数据保存一次模型
            if batch_count % (save_interval*20) == 0:
                torch.save(model.state_dict(), f'facenet_epoch_{epoch + 1}_{batch_count + 1}.pth')

            # 每处理20批数据保存一次参数和优化器状态
            if batch_count % 20 == 0:
                save_checkpoint(model, optimizer, scaler, epoch, batch_count)
                print(f'Checkpoint saved at epoch {epoch + 1}, batch {batch_count}')

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

        # 每训练save_interval次保存一次模型
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'facenet_epoch_{epoch + 1}.pth')
            print(f'Model saved at epoch {epoch + 1}')

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')  # 设置多进程启动方式为'spawn'
    # 开始训练
    train_model(model, dataloader, criterion, optimizer, num_epochs=25, save_interval=10)
