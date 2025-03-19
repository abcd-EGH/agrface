import os
import cv2
import insightface
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from config.config import DETECTED_FACES_DIR, RESULTS_DIR, MODEL_PATH

# 라벨 매핑
AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
GENDER_CLASSES = ["Male", "Female"]
RACE_CLASSES = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]

# MTCNN 초기화 (GPU 사용 가능 시)
device_for_mtcnn = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn_detector = MTCNN(keep_all=False, device=device_for_mtcnn)

# ArcFace 모델 초기화 (CPU 사용, GPU 사용 시 ctx_id=0)
arcface_model = insightface.model_zoo.get_model('weights/arcfaceresnet100-8.onnx')
arcface_model.prepare(ctx_id=0)

def get_embedding(image_path):
    """
    ArcFace를 사용하여 이미지에서 512차원의 임베딩을 추출합니다.
    """
    img = cv2.imread(image_path)
    faces = arcface_model.get_feat(img)
    if len(faces) > 0:
        embedding = faces[0]
        return torch.tensor(embedding, dtype=torch.float32)
    else:
        print(f"얼굴 검출 실패: {image_path}")
        return torch.zeros(512, dtype=torch.float32)

def crop_and_save_face(image_path, facial_area, save_folder=DETECTED_FACES_DIR):
    """
    얼굴 영역을 crop하여 저장하는 함수입니다.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    img = Image.open(image_path).convert("RGB")
    x = facial_area['x']
    y = facial_area['y']
    w = facial_area['w']
    h = facial_area['h']
    crop_box = (x, y, x+w, y+h)
    cropped_img = img.crop(crop_box)
    
    base_name = os.path.basename(image_path)
    save_path = os.path.join(save_folder, f"detected_{base_name}")
    cropped_img.save(save_path)
    return save_path

def train_model(model, train_loader, val_loader, device, epochs=10, learning_rate=1e-3):
    """
    모델 학습 후, 각 epoch의 손실을 CSV 파일과 그래프로 저장합니다.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            embeddings = batch["embedding"].to(device)
            age_labels = batch["age"].to(device)
            gender_labels = batch["gender"].to(device)
            race_labels = batch["race"].to(device)
            
            optimizer.zero_grad()
            logits_age, logits_gender, logits_race = model(embeddings)
            loss = (criterion(logits_age, age_labels) +
                    criterion(logits_gender, gender_labels) +
                    criterion(logits_race, race_labels))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch["embedding"].to(device)
                age_labels = batch["age"].to(device)
                gender_labels = batch["gender"].to(device)
                race_labels = batch["race"].to(device)
                logits_age, logits_gender, logits_race = model(embeddings)
                loss = (criterion(logits_age, age_labels) +
                        criterion(logits_gender, gender_labels) +
                        criterion(logits_race, race_labels))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    # 모델 저장
    if not os.path.exists("weights"):
        os.makedirs("weights")
    torch.save(model.state_dict(), MODEL_PATH)
    print("학습 완료 및 모델 저장됨.")
    
    # 결과 저장 (CSV 및 그래프)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    results_df = pd.DataFrame({
        "epoch": list(range(1, epochs+1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    csv_path = os.path.join(RESULTS_DIR, "train_val_loss.csv")
    results_df.to_csv(csv_path, index=False)
    
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(RESULTS_DIR, "train_val_loss_plot.png"))
    plt.close()
    print(f"Train/Val Loss 결과를 '{csv_path}' 파일과 그래프로 저장했습니다.")

def inference(model, image_path, device):
    """
    추론 함수로, 얼굴 검출(MTCNN)과 ArcFace 임베딩 추출 후 멀티태스크 예측을 수행합니다.
    """
    model.to(device)
    model.eval()
    
    img_pil = Image.open(image_path).convert('RGB')
    bboxes, _ = mtcnn_detector.detect(img_pil)
    if bboxes is not None and len(bboxes) > 0:
        x1, y1, x2, y2 = bboxes[0]
        facial_area = {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}
        face_crop = img_pil.crop((x1, y1, x2, y2))
        face_crop_cv2 = cv2.cvtColor(np.array(face_crop), cv2.COLOR_RGB2BGR)
        faces_arc = arcface_model.get_feat(face_crop_cv2)
        if len(faces_arc) > 0:
            embedding = faces_arc[0]
        else:
            print("ArcFace 임베딩 추출 실패 (crop된 얼굴).")
            embedding = np.zeros(512)
    else:
        print("MTCNN을 통한 얼굴 검출 실패:", image_path)
        return None
    
    face_name_align = crop_and_save_face(image_path, facial_area)
    
    emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_age, logits_gender, logits_race = model(emb_tensor)
        prob_age = torch.softmax(logits_age, dim=1).cpu().numpy()[0]
        prob_gender = torch.softmax(logits_gender, dim=1).cpu().numpy()[0]
        prob_race = torch.softmax(logits_race, dim=1).cpu().numpy()[0]
        pred_age_idx = int(torch.argmax(logits_age, dim=1).cpu().item())
        pred_gender_idx = int(torch.argmax(logits_gender, dim=1).cpu().item())
        pred_race_idx = int(torch.argmax(logits_race, dim=1).cpu().item())
    
    pred_age = AGE_CLASSES[pred_age_idx]
    pred_gender = GENDER_CLASSES[pred_gender_idx]
    pred_race = RACE_CLASSES[pred_race_idx]
    
    output = {
        "face_name_align": face_name_align,
        "race": pred_race,
        "gender": pred_gender,
        "age": pred_age,
        "race_scores_fair": prob_race.tolist(),
        "gender_scores_fair": prob_gender.tolist(),
        "age_scores_fair": prob_age.tolist()
    }
    
    return output
