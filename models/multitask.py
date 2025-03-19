import torch.nn as nn

class MultiTaskHead(nn.Module):
    """
    512차원 임베딩을 입력받아 나이, 성별, 인종을 예측하는 멀티태스크 헤드 모델입니다.
    각 태스크는 hidden layer (ReLU 포함)를 거쳐 예측 logits를 출력합니다.
    """
    def __init__(self, input_dim=512, hidden_dim=64, num_age=9, num_gender=2, num_race=7):
        super(MultiTaskHead, self).__init__()
        self.fc_age = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_age)
        )
        self.fc_gender = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_gender)
        )
        self.fc_race = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_race)
        )
        
    def forward(self, x):
        logits_age = self.fc_age(x)
        logits_gender = self.fc_gender(x)
        logits_race = self.fc_race(x)
        return logits_age, logits_gender, logits_race
