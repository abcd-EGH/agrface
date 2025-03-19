import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.multitask import MultiTaskHead
from models.base_multitask import BaseMultiTaskHead
from data.dataset import FairFaceDataset
from utils.face_utils import train_model, AGE_CLASSES, GENDER_CLASSES, RACE_CLASSES
from config.config import DATA_ROOT, TRAIN_CSV, VAL_CSV, EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_SAMPLES, DROPOUT_P, INPUT_DIM, SHARED_DIM, HIDDEN_DIM

def main():
    parser = argparse.ArgumentParser(description="FairFace 멀티태스크 모델 학습")
    parser.add_argument("--train_csv", type=str, default=TRAIN_CSV, help="학습 라벨 csv 경로")
    parser.add_argument("--val_csv", type=str, default=VAL_CSV, help="검증 라벨 csv 경로")
    parser.add_argument("--train_img_root", type=str, default=DATA_ROOT, help="학습 이미지 폴더 루트")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="배치 사이즈")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="학습률")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES, help="사용할 이미지 최대 개수 (None이면 전체 사용)")
    parser.add_argument("--dropout_p", type=float, default=DROPOUT_P, help="Dropout 확률")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiTaskHead(
        input_dim=INPUT_DIM,
        shared_dim=SHARED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_age=len(AGE_CLASSES),
        num_gender=len(GENDER_CLASSES),
        num_race=len(RACE_CLASSES),
        dropout_p=args.dropout_p
    )
    
    train_dataset = FairFaceDataset(csv_path=args.train_csv, images_root=args.train_img_root, max_samples=args.max_samples)
    val_dataset = FairFaceDataset(csv_path=args.val_csv, images_root=args.train_img_root, max_samples=args.max_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    train_model(model, train_loader, val_loader, device, epochs=args.epochs, learning_rate=args.learning_rate)

if __name__ == "__main__":
    main()
    # python train.py --batch_size 256 --learning_rate 3e-4 --max_samples 1000