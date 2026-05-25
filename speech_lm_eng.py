import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import os
import glob
import math
from pathlib import Path
import matplotlib.pyplot as plt
import re
import numpy as np

if not hasattr(np, 'complex'):
    np.complex = complex


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads, self.d_model, self.d_k = num_heads, d_model, d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        bs, sl = x.size(0), x.size(1)
        q = self.q_linear(x).view(bs, sl, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(bs, sl, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(bs, sl, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, sl, self.d_model)
        return self.out(out)

class SELMTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.2):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class SELMSpeechLM(nn.Module):
    def __init__(self, vocab_size=1024, d_model=256, num_heads=16, num_layers=8, d_ff=None, dropout=0.2):
        super().__init__()
        if d_ff is None: d_ff = 4 * d_model
        self.vocab_size, self.d_model = vocab_size, d_model
        self.audio_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([SELMTransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.linear_classifier = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding): torch.nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, input_tokens):
        x = self.dropout(self.positional_encoding(self.audio_embedding(input_tokens)))
        for block in self.transformer_blocks: x = block(x)
        return self.linear_classifier(x)



class CodecTokenizer:
    def __init__(self, layer_ids=[6]):
        self.codec = torch.hub.load("lucadellalib/discrete-wavlm-codec", "discrete_wavlm_large", layer_ids=layer_ids, pretrained=True)
        self.codec.eval().requires_grad_(False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.codec = self.codec.to(self.device)
        self.vocab_size = getattr(self.codec.quantizer, 'num_clusters', [1024])[0]
    def features_to_tokens(self, features):
        if isinstance(features, str): features = torch.load(features, weights_only=True)
        features = features.float().to(self.device)
        if len(features.shape) == 2: features = features.unsqueeze(0).unsqueeze(-1)
        elif len(features.shape) == 3:
            if features.shape[0] == 1: features = features.unsqueeze(-1)
            else: features = features[:, :, 0].unsqueeze(0).unsqueeze(-1)
        with torch.no_grad(): tokens = self.codec.feats_to_toks(features.contiguous())
        return tokens.squeeze(0).cpu()

class FeatureTokenDataset(Dataset):
    def __init__(self, alar_dirs, normal_dirs, tokenizer, sequence_length=500):
        self.sequence_length, self.tokenizer = sequence_length, tokenizer
        alar_files, normal_files = [], []
        
        for d in alar_dirs:
            files = glob.glob(str(Path(d) / "*.pt"))
            if not files: print(f"Warning: No files found in alar dir: {d}")
            alar_files.extend(files)
        for d in normal_dirs:
            files = glob.glob(str(Path(d) / "*.pt"))
            if not files: print(f"Warning: No files found in normal dir: {d}")
            normal_files.extend(files)
        
        paired_files = self._smart_pair_files(alar_files, normal_files)
        
        print(f"\n--- Found {len(paired_files)} Paired Files ---")

        real_count = sum(1 for a, n in paired_files if 'real_001' in a)
        aug_count = len(paired_files) - real_count
        print(f"Real alaryngeal pairs: {real_count}")
        print(f"Augmented alaryngeal pairs: {aug_count}")
        print(f"Ratio (Aug/Real): {aug_count/real_count if real_count > 0 else 'inf':.2f}")

        self.input_tokens, self.target_tokens, self.sample_weights = [], [], []
        
        for af, nf in tqdm(paired_files, desc="Processing Tokens"):
            try:
                at = self.tokenizer.features_to_tokens(af).flatten()
                nt = self.tokenizer.features_to_tokens(nf).flatten()
                is_real = 'real_001' in af
                weight = 5.0 if is_real else 1.0  # 5x weight for real data
                
                self._add_segments(at, nt, weight)
            except Exception as e:
                continue
        
        print(f"\n--- Dataset Statistics ---")
        print(f"Total samples: {len(self.input_tokens)}")
        print(f"Weight distribution - min: {min(self.sample_weights):.1f}, max: {max(self.sample_weights):.1f}")
        weighted_real = sum(1 for w in self.sample_weights if w > 1.0)
        print(f"Weighted (real) samples: {weighted_real} ({100*weighted_real/len(self.sample_weights):.1f}%)")

    def _smart_pair_files(self, alar_files, normal_files):
        paired, normal_lookup = [], {}
        
        def get_key(f):
            fname = os.path.basename(f)
            key = fname.replace("-converted_normal_to_alaryngeal", "").replace("_converted_normal_to_alaryngeal", "")
            key = key.replace("-original_normal_to_alaryngeal", "").replace("_original_normal_to_alaryngeal", "")
            if "segment" in key or (re.search(r'\d+_\d+', key)):
                nums = re.findall(r'\d+', key)
                return "_".join([str(int(n)) for n in nums])
            return key.replace(".pt", "")

        for f in normal_files:
            normal_lookup.setdefault(get_key(f), []).append(f)

        failed_to_pair = []
        for f in alar_files:
            key = get_key(f)
            if key in normal_lookup and normal_lookup[key]:
                paired.append((f, normal_lookup[key].pop(0)))
            else:
                failed_to_pair.append(os.path.basename(f))
        
        if failed_to_pair and len(failed_to_pair) < 10:
            print(f"\n--- {len(failed_to_pair)} files failed to pair ---")
            print(f"Failed: {failed_to_pair}")
            
        return paired

    def _add_segments(self, at, nt, weight):
        ml = min(len(at), len(nt))
        if ml <= self.sequence_length:
            self.input_tokens.append(self._pad(at))
            self.target_tokens.append(self._pad(nt))
            self.sample_weights.append(weight)
        else:
            stride = self.sequence_length // 3
            for s in range(0, ml - self.sequence_length + 1, stride):
                self.input_tokens.append(at[s:s+self.sequence_length])
                self.target_tokens.append(nt[s:s+self.sequence_length])
                self.sample_weights.append(weight)
                
    def _pad(self, t):
        if len(t) >= self.sequence_length: return t[:self.sequence_length]
        return torch.cat([t, torch.zeros(self.sequence_length - len(t), dtype=t.dtype)])
    
    def __len__(self): return len(self.input_tokens)
    def __getitem__(self, idx): 
        return self.input_tokens[idx], self.target_tokens[idx], self.sample_weights[idx]


class SubsetWeightedRandomSampler(torch.utils.data.Sampler):
    """Weighted sampler that works with Subset datasets"""
    def __init__(self, subset, base_weights, real_weight_multiplier=1.0):
        self.subset = subset
        self.base_weights = base_weights
        self.real_weight_multiplier = real_weight_multiplier
        self.weights = []
        for idx in subset.indices:
            base_w = base_weights[idx]
            if base_w > 1.0:
                self.weights.append(base_w * real_weight_multiplier)
            else:
                self.weights.append(base_w)
        
        self.num_samples = len(self.subset)
    
    def __iter__(self):
        return iter(torch.multinomial(torch.tensor(self.weights), self.num_samples, replacement=True).tolist())
    
    def __len__(self):
        return self.num_samples


def plot_loss(history, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('SELM Training History'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss_plot.png")); plt.close()

def train_selm(model, train_subset, val_subset, full_dataset, num_epochs, lr, device, save_dir, resume_path=None):
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    start_epoch, best_v = 0, float('inf')
    history = {'train_loss': [], 'val_loss': []}

    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'history' in ckpt: history = ckpt['history']
        best_v = min(history['val_loss']) if (history['val_loss'] and len(history['val_loss']) > 0) else best_v
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_v:.4f}")

   
    CURRICULUM_EPOCHS = 15
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        
        if epoch < CURRICULUM_EPOCHS:
            real_multiplier = 2.0  # 5.0 base * 2.0 = 10x total
            print(f"Curriculum Phase 1: Real weight multiplier = {real_multiplier}")
        else:
            real_multiplier = 0.6  # 5.0 base * 0.6 = 3x total
            print(f"Curriculum Phase 2: Real weight multiplier = {real_multiplier}")
        
        print(f"{'='*60}")
        
       
        train_sampler = SubsetWeightedRandomSampler(
            train_subset, 
            full_dataset.sample_weights,
            real_weight_multiplier=real_multiplier
        )
        
        t_loader = DataLoader(train_subset, batch_size=16, sampler=train_sampler, num_workers=4)
        v_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4)
        
        model.train()
        t_loss = 0
        pbar = tqdm(t_loader, desc=f"Training")
        for batch in pbar:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(x).transpose(1, 2), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            t_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        avg_t = t_loss / len(t_loader)
        history['train_loss'].append(avg_t)
        scheduler.step()
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in v_loader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                v_loss += criterion(model(x).transpose(1, 2), y).item()
        
        avg_v = v_loss / len(v_loader)
        history['val_loss'].append(avg_v)
        print(f"Train: {avg_t:.4f} | Val: {avg_v:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        plot_loss(history, save_dir)

        
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }
        torch.save(ckpt_data, os.path.join(save_dir, 'last_checkpoint.pt'))
        
        if avg_v < best_v:
            best_v = avg_v
            torch.save(ckpt_data, os.path.join(save_dir, 'selm_tokens_best.pt'))
            print(f"New Best Model Saved! Val Loss: {best_v:.4f}")


def main():
    SAVE_DIR = "/workspace/Kris/knn-vc/LM_detokenizer_package/speech_lm_models_U_Eng/with_augmentation_balanced"
    CHECKPOINT_FILE = os.path.join(SAVE_DIR, "last_checkpoint.pt")
    
    ALAR_DIRS = [
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/real_001",
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/001/converted",
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/002/converted",
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/003/converted",
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/004/converted",
    ]
    NORM_DIRS = [
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/real_001",
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/001",
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/002",
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/003",
        "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/004",
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CodecTokenizer()
    
    v_size = tokenizer.vocab_size
    if os.path.exists(CHECKPOINT_FILE):
        try:
            v_size = torch.load(CHECKPOINT_FILE, map_location='cpu')['model_state_dict']['audio_embedding.weight'].shape[0]
            print(f"Resuming with detected vocab_size: {v_size}")
        except: pass

    model = SELMSpeechLM(vocab_size=v_size, d_model=512, num_heads=16, num_layers=8, dropout=0.2).to(device)
    
    full_dataset = FeatureTokenDataset(ALAR_DIRS, NORM_DIRS, tokenizer, 500)
    
    if len(full_dataset) < 2:
        print("Error: Not enough data paired.")
        return
    v_len = int(len(full_dataset) * 0.1)
    train_indices = list(range(len(full_dataset) - v_len))
    val_indices = list(range(len(full_dataset) - v_len, len(full_dataset)))
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_selm(model, train_subset, val_subset, full_dataset, 100, 1e-4, device, SAVE_DIR, CHECKPOINT_FILE)

if __name__ == "__main__":
    main()
