import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from torchvision.models import Inception_V3_Weights
from scipy.spatial.distance import cosine

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
        if isinstance(x, tuple):  # 检查输出是否为元组
            x = x.logits  # 提取logits部分
        x = torch.nn.functional.normalize(x, p=2, dim=1)  # 使用torch.nn.functional.normalize进行L2归一化
        return x

# 初始化模型并加载权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceNet(embedding_size=128).to(device)
model.load_state_dict(torch.load('checkpoint.pth.tar')['state_dict'])
model.eval()  # 设置模型为评估模式

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 获取图像的嵌入向量
def get_embedding(model, image):
    image = transform(image).unsqueeze(0).to(device)  # 添加批次维度并移动到GPU
    with torch.no_grad():
        outputs = model(image)
        if isinstance(outputs, tuple):  # 检查输出是否为元组
            outputs = outputs.logits  # 提取logits部分
    return outputs.cpu().numpy().flatten()  # 展平为1维向量

# 加载人脸数据库
def load_face_database(database_path):
    face_database = {}
    for person_name in os.listdir(database_path):
        person_dir = os.path.join(database_path, person_name)
        if os.path.isdir(person_dir):
            embeddings = []
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = Image.open(image_path).convert('RGB')
                embedding = get_embedding(model, image)
                embeddings.append(embedding)
            face_database[person_name] = embeddings
    return face_database

# 比较两个嵌入向量
def is_same_person(embedding1, embedding2, threshold=0.5):
    distance = cosine(embedding1, embedding2)
    return distance < threshold

# 找到最相似的人脸
def find_best_match(face_database, embedding, threshold=0.5):
    best_match = None
    best_distance = float('inf')
    for person_name, embeddings in face_database.items():
        for db_embedding in embeddings:
            distance = cosine(embedding, db_embedding)
            if distance < best_distance:
                best_distance = distance
                best_match = person_name
    if best_distance < threshold:
        return best_match
    else:
        return "Stranger"

# 加载人脸数据库
database_path = 'faces'
face_database = load_face_database(database_path)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 使用OpenCV的Haar级联分类器进行人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为灰度图像进行人脸检测
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 提取人脸区域并转换为PIL图像
        face_image = frame[y:y+h, x:x+w]
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

        # 计算当前帧的嵌入向量
        current_embedding = get_embedding(model, pil_image)

        # 找到最相似的人脸
        label = find_best_match(face_database, current_embedding)

        # 在帧上绘制矩形框和标签
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示帧
    cv2.imshow('Face Recognition', frame)

    # 按下'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
