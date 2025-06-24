import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
class EnhancedCNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedCNNLSTM, self).__init__()
        
        # Multi-scale CNN with residual connections
        self.conv_block1 = self._make_multiscale_block(1, 32)
        self.conv_block2 = self._make_multiscale_block(32, 64)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d((1, 2))
        self.pool2 = nn.MaxPool2d((1, 3))
        
        # LSTM configuration (maintains your original dimensions)
        self.lstm_input_size = 64 * 9  # 576
        self.lstm_hidden = 96
        
        # Bidirectional LSTM with layer normalization
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Statistical pooling unit
        self.stat_pooling = StatisticalPooling()
        
        # Enhanced classifier with residual connection
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden * 4, 256),  # *4 for bidirectional + stat pooling
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _make_multiscale_block(self, in_channels, out_channels):
        # Split channels as evenly as possible without dropping any
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - (c1 + c2)  # Remainder goes to last branch to make total exact
    
        return nn.ModuleDict({
            'conv1x1': nn.Conv2d(in_channels, c1, kernel_size=1),
            'conv3x3': nn.Conv2d(in_channels, c2, kernel_size=3, padding=1),
            'conv5x5': nn.Conv2d(in_channels, c3, kernel_size=5, padding=2),
            'bn': nn.BatchNorm2d(out_channels),
            'residual': nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        })


    
    def _apply_multiscale_block(self, x, block):
        """Apply multi-scale convolution with residual connection"""
        # Multi-scale convolutions
        conv1 = block['conv1x1'](x)
        conv3 = block['conv3x3'](x)
        conv5 = block['conv5x5'](x)
        
        # Concatenate multi-scale features
        out = torch.cat([conv1, conv3, conv5], dim=1)
        out = block['bn'](out)
        
        # Residual connection
        residual = block['residual'](x)
        out = F.relu(out + residual)
        
        return out
    
    def _init_weights(self):
        """Advanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        # Multi-scale CNN with residual connections
        x = self._apply_multiscale_block(x, self.conv_block1)  # [B, 32, H, W]
        x = self.pool1(x)  # [B, 32, H, W/2]
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self._apply_multiscale_block(x, self.conv_block2)  # [B, 64, H, W/2]
        x = self.pool2(x)  # [B, 64, 9, 3]
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Reshape for LSTM (maintaining your original approach)
        x = x.permute(0, 3, 1, 2)  # [B, 3, 64, 9]
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # [B, 3, 576]
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # [B, 3, 192]
        
        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [B, 3, 192]
        enhanced_features = lstm_out + attn_out  # Residual connection
        
        # Statistical pooling (mean + std across time dimension)
        pooled_features = self.stat_pooling(enhanced_features)  # [B, 384]
        
        # Final classification
        output = self.fc(pooled_features)
        
        return output


class StatisticalPooling(nn.Module):
    """Statistical pooling unit for better temporal feature aggregation"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x shape: [batch, time, features]
        mean = torch.mean(x, dim=1)  # [batch, features]
        std = torch.std(x, dim=1)    # [batch, features]
        return torch.cat([mean, std], dim=1)  # [batch, features*2]


# Training utilities
class AdaptiveDropout(nn.Module):
    """Adaptive dropout that changes rate during training"""
    def __init__(self, initial_p=0.5, final_p=0.1, total_epochs=100):
        super().__init__()
        self.initial_p = initial_p
        self.final_p = final_p
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def forward(self, x):
        if self.training:
            current_p = self.initial_p - (self.initial_p - self.final_p) * (self.current_epoch / self.total_epochs)
            return F.dropout(x, p=current_p, training=True)
        return x
    
    def step_epoch(self):
        self.current_epoch += 1


# Usage example with training configuration
def create_enhanced_model(num_classes=7):  # 7 for typical emotion classes
    model = EnhancedCNNLSTM(num_classes)
    return model


