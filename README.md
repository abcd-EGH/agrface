Windows에서 오류 발생 시 cmd (또는 Powershell)을 관리자 권한으로 실행 후 아래 명령어를 입력 (`PYTHONUTF8` 환경 변수 설정)
```bash
setx PYTHONUTF8 1
```

```bash
project/
 ├── config/
 │    └── config.py         # 공통 설정 (경로, 하이퍼파라미터 등)
 ├── data/
 │    ├── __init__.py
 │    └── dataset.py        # 데이터셋 클래스 정의
 ├── models/
 │    ├── __init__.py
 │    └── multitask.py      # 멀티태스크 모델(MultiTaskHead) 정의
 ├── utils/
 │    ├── __init__.py
 │    └── face_utils.py     # 얼굴 임베딩, 얼굴 영역 crop, 학습/추론 함수 등
 ├── train.py                # 학습 실행 스크립트
 └── inference.py            # 추론 실행 스크립트
```

train
```bash
python train.py
```

You can find the train/val loss in the `results` dir (will be made once you train the model), and a detected face in `detected_faces` dir (will be made once you perform inference on your test image).

inference
```bash
python inference.py --test_image test_img/winter4.jpg
```

To-do
- [ ] test_img 라이브러리에 있는 모든 test image에 대해 추론 수행
- [ ] 추론 결과를 csv 파일로 results 폴더에 저장
