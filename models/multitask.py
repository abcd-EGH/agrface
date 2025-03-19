import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    간단한 Residual Block:
      - fc1 -> BN1 -> ReLU -> fc2 -> BN2
      - Skip Connection(identity)를 덧셈
      - 최종 ReLU
    """
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        
        # 차원이 다를 경우 skip용 레이어 추가
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        
        # in_dim != out_dim인 경우 skip 레이어를 통해 identity 차원을 맞춤
        if self.skip is not None:
            identity = self.skip(identity)
        
        out += identity
        out = self.relu(out)
        return out


class MultiTaskHead(nn.Module):
    """
    Shared + Separate Heads, Residual, Dropout, BatchNorm을 적용한 예시 모델
    - input_dim: 입력 임베딩 차원 (예: 512)
    - shared_dim: ResidualBlock에서 변환할 임베딩 차원
    - hidden_dim: 최종 분류 Head에 사용될 히든 차원
    - num_age, num_gender, num_race: 각 태스크 클래스 개수
    """
    def __init__(self, 
                 input_dim=512, 
                 shared_dim=256, 
                 hidden_dim=128,
                 num_age=9, 
                 num_gender=2, 
                 num_race=7,
                 dropout_p=0.2):
        super(MultiTaskHead, self).__init__()
        
        # ----- Shared Part -----
        # 필요에 따라 ResidualBlock을 여러 개 쌓거나, 중간 중간 드롭아웃을 넣을 수 있음
        self.shared_blocks = nn.Sequential(
            ResidualBlock(input_dim, shared_dim),
            nn.Dropout(p=dropout_p),
            ResidualBlock(shared_dim, shared_dim)
        )
        
        # ----- Separate Heads -----
        # 1) Age Head
        self.fc_age = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_age)
        )
        
        # 2) Gender Head
        self.fc_gender = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_gender)
        )
        
        # 3) Race Head
        self.fc_race = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_race)
        )
        
    def forward(self, x):
        # Shared Representation
        shared_feat = self.shared_blocks(x)  # (batch_size, shared_dim)
        
        # Separate Heads
        logits_age = self.fc_age(shared_feat)      # (batch_size, num_age)
        logits_gender = self.fc_gender(shared_feat)  # (batch_size, num_gender)
        logits_race = self.fc_race(shared_feat)      # (batch_size, num_race)
        
        return logits_age, logits_gender, logits_race
