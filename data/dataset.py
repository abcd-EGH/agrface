import os
import pandas as pd
from torch.utils.data import Dataset
from utils.face_utils import get_embedding
from config.config import DATA_ROOT

# 라벨 매핑
AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
GENDER_CLASSES = ["Male", "Female"]
RACE_CLASSES = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]

class FairFaceDataset(Dataset):
    """
    CSV 파일에서 이미지 경로와 라벨 정보를 읽어오며,
    ArcFace 임베딩을 추출하는 데이터셋 클래스입니다.
    """
    def __init__(self, csv_path, images_root=DATA_ROOT, transform=None, max_samples=None):
        self.data = pd.read_csv(csv_path)
        if max_samples is not None:
            self.data = self.data.iloc[:max_samples]
        self.images_root = images_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_rel_path = row["file"]  # 예: train/1.jpg 또는 val/1.jpg
        image_path = os.path.join(self.images_root, image_rel_path)
        
        # ArcFace 임베딩 추출
        embedding = get_embedding(image_path)
        
        age_label = AGE_CLASSES.index(row["age"])
        gender_label = GENDER_CLASSES.index(row["gender"])
        race_label = RACE_CLASSES.index(row["race"])
        
        sample = {
            "embedding": embedding, 
            "age": age_label,
            "gender": gender_label,
            "race": race_label
        }
        return sample
