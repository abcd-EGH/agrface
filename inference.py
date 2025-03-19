import os
import argparse
import torch
from models.multitask import MultiTaskHead
from utils.face_utils import inference, AGE_CLASSES, GENDER_CLASSES, RACE_CLASSES
from config.config import TEST_IMAGE, MODEL_PATH, INPUT_DIM, SHARED_DIM, HIDDEN_DIM

def main():
    parser = argparse.ArgumentParser(description="FairFace 멀티태스크 모델 추론")
    parser.add_argument("--test_image", type=str, default=TEST_IMAGE, help="추론에 사용할 테스트 이미지 경로")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="저장된 모델 파일 경로")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiTaskHead(
        input_dim=INPUT_DIM,
        shared_dim=SHARED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_age=len(AGE_CLASSES),
        num_gender=len(GENDER_CLASSES),
        num_race=len(RACE_CLASSES),
    )
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("저장된 모델 로드 완료.")
    else:
        print("모델 파일이 존재하지 않습니다. 학습 후 추론해 주시기 바랍니다.")
        exit(1)
    
    result = inference(model, args.test_image, device)
    if result is not None:
        print("추론 결과:")
        print(result)

if __name__ == "__main__":
    main()
    # python inference.py --model_path weights/multitask_head.pth --test_image test_img/winter4.jpg
