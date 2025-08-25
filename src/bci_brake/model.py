import torch
from torch import nn

from torch import Tensor
import torch.nn.functional as F


class EEGBrakeLinv0(nn.Module):
    def __init__(self):
        super().__init__()

        self.r_lin0 = nn.LazyLinear(256)
        self.r_lin1 = nn.Linear(256, 128)
        self.r_lin2 = nn.Linear(128, 64)

        self.f_lin0 = nn.LazyLinear(256)
        self.f_lin1 = nn.Linear(256, 128)
        self.f_lin2 = nn.Linear(128, 64)
        self.f_lin3 = nn.LazyLinear(1)

        self.act_fn = nn.GELU()

    def forward(self, x):
        # x: (N, R, F)

        x_r = self.r_lin0(x.permute((0, 2, 1)))
        x_r = self.act_fn(x_r)

        x_r = self.r_lin1(x_r)
        x_r = self.act_fn(x_r)

        x_r = self.r_lin2(x_r)
        x_r = self.act_fn(x_r)


        x_f = self.f_lin0(x)
        x_f = self.act_fn(x_f)

        x_f = self.f_lin1(x_f)
        x_f = self.act_fn(x_f)

        x_f = self.f_lin2(x_f)
        x_f = self.act_fn(x_f)

        x = x_f @ x_r.permute((0, 2, 1))
        x = self.f_lin3(x)

        return x.squeeze(-1)


class EEGBrakeLinv1(nn.Module):
    def __init__(self, n_rows, n_features):
        super().__init__()

        self.r_lin0 = nn.Linear(n_rows, 256)
        self.r_lin1 = nn.Linear(256, 128)
        self.r_lin2 = nn.Linear(128, n_rows)

        self.f_lin0 = nn.Linear(n_features, 64)
        self.f_lin1 = nn.Linear(64, 32)
        self.f_lin2 = nn.Linear(32, 1)

        self.lin_head = nn.Linear(n_features, 1)

        self.act_fn = nn.GELU()

    def forward(self, x):
        # x: (N, R, F)

        x_r = self.r_lin0(x.permute((0, 2, 1)))
        x_r = self.act_fn(x_r)

        x_r = self.r_lin1(x_r)
        x_r = self.act_fn(x_r)

        x_r = self.r_lin2(x_r).permute((0, 2, 1))


        x_f = self.f_lin0(x)
        x_f = self.act_fn(x_f)

        x_f = self.f_lin1(x_f)
        x_f = self.act_fn(x_f)

        x_f = self.f_lin2(x_f)

        x = x_r * x_f
        x = self.act_fn(x)

        x = self.lin_head(x)

        return x.squeeze(-1)

# ---------------------------
# Causal 1D convolution layer
# (no future leakage)
# ---------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=False):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=0, dilation=dilation, bias=bias)

    def forward(self, x):
        # x: (B, C, T). Left pad only.
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

# ---------------------------
# Temporal CNN block (causal)
# ---------------------------
class CausalTemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1, dropout=0.1):
        super().__init__()
        self.dw = CausalConv1d(in_ch, in_ch, k, dilation=dilation, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        y = self.pw(self.dw(x))
        y = self.bn(y)
        y = F.elu(y)
        return self.do(y)

# ---------------------------
# Full model (causal, seq output)
# ---------------------------
class EEGBrakeSeqLSTM(nn.Module):
    """
    Inputs:
        x: (B, T, C)  EEG window (C=channels)
        lengths: optional (B,) valid lengths if padded
    Outputs:
        seq_logits: (B, T) per-step logit (higher = brake onset now)
    """
    def __init__(self,
                 n_channels: int,
                 cnn_branch_out: int = 32,
                 kernels=(9, 25, 75),   # ~36ms, 100ms, 300ms @ 250 Hz
                 dilations=(1, 2, 4),
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        # Multi-branch causal temporal front-end
        branches = []
        for k in kernels:
            branches.append(CausalTemporalBlock(n_channels, cnn_branch_out, k, dilation=1, dropout=dropout))
        for d in dilations:
            branches.append(CausalTemporalBlock(n_channels, cnn_branch_out, kernels[1], dilation=d, dropout=dropout))
        self.branches = nn.ModuleList(branches)
        cnn_out = cnn_branch_out * len(branches)

        self.post_cnn = nn.Sequential(
            nn.Conv1d(cnn_out, cnn_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(cnn_out),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Uni-directional LSTM = causal across time
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False
        )

        # Time-distributed classification head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, lengths=None):
        B, T, C = x.shape
        # CNN expects (B, C, T)
        x = x.transpose(1, 2)

        feats = torch.cat([br(x) for br in self.branches], dim=1)  # (B, cnn_out, T)
        feats = self.post_cnn(feats)                                # (B, cnn_out, T)
        feats = feats.transpose(1, 2)                               # (B, T, cnn_out)

        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=feats.device)
            sorted_len, idx = torch.sort(lengths, descending=True)
            feats = feats.index_select(0, idx)
            packed = nn.utils.rnn.pack_padded_sequence(feats, sorted_len.cpu(), batch_first=True)
            packed_out, _ = self.lstm(packed)
            H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
            _, inv = torch.sort(idx)
            H = H.index_select(0, inv)  # (B, T, H)
        else:
            H, _ = self.lstm(feats)     # (B, T, H)

        seq_logits = self.head(H).squeeze(-1)  # (B, T)
        return seq_logits

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    B, T, C = 32, 500, 64   # batch, 2s window @250Hz, 64 EEG channels
    x = torch.randn(B, T, C)

    model = EEGBrakeSeqLSTM(n_channels=C)
    logits = model(x)   # logits: (B,), attn_w: (B, T)

    # Loss (binary): braking in next H samples (horizon) = 1 vs 0
    y = torch.randint(0, 2, (B,), dtype=torch.float32)
    pos_weight = torch.tensor([4.0])  # tune based on class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loss = criterion(logits, y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# if __name__ == "__main__":
#     x = torch.rand((25, 200, 59), dtype=torch.float32)
#
#     model = BCIModel3(200, 59)
#     y = model(x)
#     print(y.shape)
#
#     print(sum(param.numel() for param in model.parameters()))

    # conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5), stride=2, padding=(1, 2))
    # conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
    # x = conv(x)
    # print(x.shape)




