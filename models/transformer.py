import torch.nn as nn

class TransformerMultiTaskHead(nn.Module):
    """
    Transformer를 사용하여 512차원 embedding으로부터 age, gender, race를 예측
    """
    def __init__(self, 
                 input_dim=512,
                 embed_dim=256,     # Transformer 입력 차원
                 hidden_dim=512,    # Feed-forward 내부 차원
                 num_heads=8,       # attention head 개수
                 num_layers=2,
                 num_age=9,
                 num_gender=2,
                 num_race=7,
                 dropout=0.2):
        super(TransformerMultiTaskHead, self).__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 태스크별 헤드
        self.head_age = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, num_age)
        )

        self.head_gender = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, num_gender)
        )

        self.head_race = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, num_race)
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.input_proj(x).unsqueeze(0)  # (1, batch_size, embed_dim)으로 변환

        # Transformer Encoder (seq_len=1로 고정)
        encoded_feat = self.transformer_encoder(x)  # (1, batch_size, embed_dim)
        encoded_feat = encoded_feat.squeeze(0)  # (batch_size, embed_dim)

        logits_age = self.head_age(encoded_feat)
        logits_gender = self.head_gender(encoded_feat)
        logits_race = self.head_race(encoded_feat)

        return logits_age, logits_gender, logits_race
