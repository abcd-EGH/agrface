import os

# 데이터 및 라벨 파일 경로
DATA_ROOT = "fairface/fairface_image_224"
TRAIN_CSV = "fairface/fairface_label_train.csv"
VAL_CSV = "fairface/fairface_label_val.csv"
TEST_IMAGE = "test_img/winter4.jpg"

# 모델 관련 경로
WEIGHTS_DIR = "weights"
MODEL_FILENAME = "arcface_multitask_head.pth"
MODEL_PATH = os.path.join(WEIGHTS_DIR, MODEL_FILENAME)

# 결과 및 검출된 얼굴 이미지 저장 폴더
RESULTS_DIR = "results"
DETECTED_FACES_DIR = "detected_faces"

# 학습 하이퍼파라미터
LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 64
MAX_SAMPLES = None
