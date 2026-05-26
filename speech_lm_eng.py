
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from tqdm import tqdm
import os
import glob
import math
from pathlib import Path
import re
import numpy as np
import random

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
        # Causal mask: decoder-only (each token attends only to previous tokens)
        causal_mask = torch.triu(torch.ones(sl, sl, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
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
    def __init__(self, vocab_size=1025, d_model=512, num_heads=16, num_layers=4, d_ff=None, dropout=0.2):
        super().__init__()
        if d_ff is None: d_ff = 4 * d_model
        self.vocab_size, self.d_model = vocab_size, d_model
        #vocab_size includes mask token at index (vocab_size-1)
        self.audio_embedding = nn.Embedding(vocab_size, d_model) 
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([SELMTransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # output layer predicts original tokens (0 to vocab_size-1)
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
        
        self.input_tokens, self.target_tokens, self.is_real_list = [], [], []
        
        real_count = 0
        
        for af, nf in tqdm(paired_files, desc="Processing Tokens"):
            try:
                at = self.tokenizer.features_to_tokens(af).flatten()
                nt = self.tokenizer.features_to_tokens(nf).flatten()
                
                is_real = 'real_001' in af
                if is_real: real_count += 1
                
                self._add_segments(at, nt, is_real)
            except Exception as e:
                continue
        
        print(f"\n--- Dataset Statistics ---")
        print(f"Total samples: {len(self.input_tokens)}")
        real_segments = sum(self.is_real_list)
        print(f"Real alaryngeal segments: {real_segments}")
        print(f"Augmented segments: {len(self.input_tokens) - real_segments}")
        print(f"Ratio (Aug/Real): {(len(self.input_tokens) - real_segments)/real_segments if real_segments > 0 else 'inf':.2f}")

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

    def _add_segments(self, at, nt, is_real):
        ml = min(len(at), len(nt))
        if ml <= self.sequence_length:
            self.input_tokens.append(self._pad(at))
            self.target_tokens.append(self._pad(nt))
            self.is_real_list.append(is_real)
        else:
            stride = self.sequence_length // 2  # Increased stride overlap for more data
            for s in range(0, ml - self.sequence_length + 1, stride):
                self.input_tokens.append(at[s:s+self.sequence_length])
                self.target_tokens.append(nt[s:s+self.sequence_length])
                self.is_real_list.append(is_real)
                
    def _pad(self, t):
        if len(t) >= self.sequence_length: return t[:self.sequence_length]
        return torch.cat([t, torch.zeros(self.sequence_length - len(t), dtype=t.dtype)])
    
    def __len__(self): return len(self.input_tokens)
    def __getitem__(self, idx): 
        # Return flag (is_real) as tensor for batch collation logic if needed
        return self.input_tokens[idx], self.target_tokens[idx], torch.tensor(self.is_real_list[idx], dtype=torch.bool)



# Curriculum Batch Sampler (Supports variable Real Ratio)
###we use curriculum learning in training as we don't have enough data, if you have enough data, you can discard it

class CurriculumBatchSampler(Sampler):
    
    def __init__(self, dataset_indices, is_real_list, batch_size, real_ratio=0.5):
        self.real_indices = [i for i in dataset_indices if is_real_list[i]]
        self.aug_indices = [i for i in dataset_indices if not is_real_list[i]]
        self.batch_size = batch_size
        self.target_real_count = int(batch_size * real_ratio)
        self.target_aug_count = batch_size - self.target_real_count
        
        if self.target_real_count < 1 and real_ratio > 0.001: 
            self.target_real_count = 1 
        elif real_ratio <= 0.001:
            self.target_real_count = 0 
            
        if self.target_aug_count < 1: self.target_aug_count = 1
        
        if len(self.real_indices) == 0 or len(self.aug_indices) == 0:
            raise ValueError("Dataset must contain both real and augmented samples.")
            
        self.num_batches = len(self.aug_indices) // self.target_aug_count
        
        # print(f"Curriculum Sampler initialized (Ratio {real_ratio:.2f}):")
        # print(f"  Real per batch: {self.target_real_count}")
        # print(f"  Aug per batch: {self.target_aug_count}")
        # print(f"  Batches per epoch: {self.num_batches}")
        
    def __iter__(self):
        batch = []
        multipler = 0
        if self.target_real_count > 0:
            multipler = math.ceil((self.num_batches * self.target_real_count) / len(self.real_indices))
            epoch_real_indices = (self.real_indices * multipler)
            random.shuffle(epoch_real_indices)
        else:
            epoch_real_indices = []
        
        epoch_aug_indices = list(self.aug_indices)
        random.shuffle(epoch_aug_indices)
        
        real_ptr = 0
        aug_ptr = 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            for _ in range(self.target_real_count):
                if real_ptr >= len(epoch_real_indices): real_ptr = 0
                batch_indices.append(epoch_real_indices[real_ptr])
                real_ptr += 1
            
            for _ in range(self.target_aug_count):
                if aug_ptr >= len(epoch_aug_indices): aug_ptr = 0
                batch_indices.append(epoch_aug_indices[aug_ptr])
                aug_ptr += 1
            
            yield batch_indices

    def __len__(self):
        return self.num_batches



def apply_token_masking(tokens, mask_token_id=1024, mask_prob=0.15):
   
    mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens = tokens.clone()
    masked_tokens[mask] = mask_token_id
    return masked_tokens

def plot_loss(history, save_dir):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    real_vals = [x['real'] for x in history['val_loss']]
    aug_vals = [x['aug'] for x in history['val_loss']]
    plt.plot(real_vals, label='Val Loss (Real)')
    plt.plot(aug_vals, label='Val Loss (Aug)', linestyle='--')
    
    plt.title('SELM Low-Mix Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss_plot_curriculum.png"))
    plt.close()

def train_selm(model, train_subset, val_loader, num_epochs, lr, device, save_dir, mask_token_id, 
               resume_path=None, train_is_real=None):
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    # SELM KL-divergence loss: L = -Σ p(y) · log(q(ŷ)/p(y))
    label_smoothing = 0.1
    
    def selm_kl_loss(logits, targets, ignore_index=0):
        """SELM paper loss: KL(p || q) with label smoothing."""
        vocab_size = logits.size(-1)
        # logits: [B, T, V], targets: [B, T]
        log_probs = F.log_softmax(logits, dim=-1)  # log q(ŷ)
        
        # construct smoothed target distribution p(y)
        with torch.no_grad():
            p = torch.full_like(log_probs, label_smoothing / vocab_size)
            p.scatter_(-1, targets.unsqueeze(-1), 1.0 - label_smoothing + label_smoothing / vocab_size)
            pad_mask = (targets == ignore_index).unsqueeze(-1).expand_as(p)
            p[pad_mask] = 0.0
        
        # KL per element [B, T, V], sum over vocab, average over time and batch
        kl = F.kl_div(log_probs, p, reduction='none')  # [B, T, V]
        kl = kl.sum(dim=-1)  # sum over vocab → [B, T]
        # Mask out padding positions for averaging
        non_pad = (targets != ignore_index).float()  # [B, T]
        loss = (kl * non_pad).sum() / non_pad.sum().clamp(min=1)
        return loss
    
    start_epoch, best_real_v = 0, float('inf')
    history = {'train_loss': [], 'val_loss': []}

    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'history' in ckpt: 
            history = ckpt['history']
            if history['val_loss'] and isinstance(history['val_loss'][0], dict):
                 best_real_v = min([x['real'] for x in history['val_loss']])

    print(f"Starting Curriculum Training on device: {device}")
    
    # --- Domain Alignment Analysis ---
    print("\n--- Performing Domain Alignment Check ---")
    # Sample a few batches to check token distributions
    real_tokens = []
    aug_tokens = []
    
    check_subset_indices = list(range(len(train_subset)))
    random.shuffle(check_subset_indices)
    
    for idx in check_subset_indices[:500]: # Check 500 samples
        toks, _, is_real = train_subset[idx]
        if is_real: real_tokens.extend(toks.tolist())
        else: aug_tokens.extend(toks.tolist())
    
    real_set = set(real_tokens)
    aug_set = set(aug_tokens)
    overlap = real_set.intersection(aug_set)
    
    print(f"Unique Real Tokens: {len(real_set)}")
    print(f"Unique Aug Tokens: {len(aug_set)}")
    print(f"Overlap Tokens: {len(overlap)}")
    print(f"Visualize Jaccard Similarity: {len(overlap) / len(real_set.union(aug_set)):.4f}")
    

    # ==========================================================================
    # CURRICULUM PHASE CONFIGURATION
    # ==========================================================================
    PHASE_1_EPOCHS = 15   # Epochs 1-25: Low-Mix (15% Real) - Earlier transition for English!
    PHASE_1_RATIO = 0.15
    PHASE_2_RATIO = 0.50  # Epochs 51+: High-Mix (50% Real)
    
    for epoch in range(start_epoch, num_epochs):
        
        # 2-Stage Curriculum: Low-Mix -> High-Mix
        if epoch < PHASE_1_EPOCHS:
            current_real_ratio = PHASE_1_RATIO
            phase_name = f"Phase 1 (Low-Mix {int(PHASE_1_RATIO*100)}%)"
        else:
            current_real_ratio = PHASE_2_RATIO
            phase_name = f"Phase 2 (High-Mix {int(PHASE_2_RATIO*100)}%)"
            
        sampler_indices = list(range(len(train_subset)))
        batch_sampler = CurriculumBatchSampler(
            sampler_indices, 
            train_is_real, 
            batch_size=16, 
            real_ratio=current_real_ratio
        )
        
        train_loader = DataLoader(
            train_subset, 
            batch_sampler=batch_sampler, 
            num_workers=4
        )
        
        model.train()
        t_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [{phase_name}]")
        
        for x, y, is_real_batch in pbar:
            x, y = x.to(device), y.to(device)
            
            # apply masking augmentation to input
            x_masked = apply_token_masking(x, mask_token_id=mask_token_id, mask_prob=0.15)
            
            optimizer.zero_grad()
            logits = model(x_masked)
            loss = selm_kl_loss(logits, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            t_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        avg_t = t_loss / len(train_loader)
        history['train_loss'].append(avg_t)
        scheduler.step()
        
       
        model.eval()
        v_loss_real, v_loss_aug = 0, 0
        count_real, count_aug = 0, 0
        
        with torch.no_grad():
            for x, y, is_real_batch in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                
               
                log_probs = F.log_softmax(logits, dim=-1)
                vocab_size = logits.size(-1)
                p = torch.full_like(log_probs, label_smoothing / vocab_size)
                p.scatter_(-1, y.unsqueeze(-1), 1.0 - label_smoothing + label_smoothing / vocab_size)
                pad_mask = (y == 0).unsqueeze(-1).expand_as(p)
                p[pad_mask] = 0.0
                # KL per token, then average over sequence
                kl_per_token = F.kl_div(log_probs, p, reduction='none').sum(dim=-1)  # [B, T]
                loss_per_sample = kl_per_token.mean(dim=1)  # [B]
                
                is_real_mask = is_real_batch.to(device)
                
                v_loss_real += loss_per_sample[is_real_mask].sum().item()
                count_real += is_real_mask.sum().item()
                
                v_loss_aug += loss_per_sample[~is_real_mask].sum().item()
                count_aug += (~is_real_mask).sum().item()
        
        avg_v_real = v_loss_real / max(count_real, 1)
        avg_v_aug = v_loss_aug / max(count_aug, 1)
        
        history['val_loss'].append({'real': avg_v_real, 'aug': avg_v_aug})
        
        print(f"[{phase_name}] Train: {avg_t:.4f} | Val Real: {avg_v_real:.4f} | Val Aug: {avg_v_aug:.4f}")
        plot_loss(history, save_dir)
        
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'vocab_size': model.vocab_size
        }
        torch.save(ckpt_data, os.path.join(save_dir, 'last_checkpoint_curriculum.pt'))
        
        if avg_v_real < best_real_v:
            best_real_v = avg_v_real
            torch.save(ckpt_data, os.path.join(save_dir, 'selm_curriculum_best.pt'))
            print(f"New Best REAL Model Saved! Val Loss: {best_real_v:.4f}")


def main():
    SAVE_DIR = "/workspace/Kris/knn-vc/LM_detokenizer_package/speech_lm_models_U_Eng/with_augmentation_curriculum_4L_KL"
    CHECKPOINT_FILE = os.path.join(SAVE_DIR, "last_checkpoint_curriculum.pt")
    
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
    
    vocab_size = tokenizer.vocab_size
    mask_token_id = vocab_size

    model = SELMSpeechLM(vocab_size=vocab_size + 1, d_model=512, num_heads=16, num_layers=4, dropout=0.3).to(device)
    
    print(f"Loading dataset...")
    full_dataset = FeatureTokenDataset(ALAR_DIRS, NORM_DIRS, tokenizer, 500)
    
    if len(full_dataset) < 2:
        print("Error: Not enough data paired.")
        return

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split_idx = int(dataset_size * 0.9)
    random.shuffle(indices)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    train_is_real = [full_dataset.is_real_list[i] for i in train_indices]

    if not any(train_is_real) or not any(not r for r in train_is_real):
        real_global_indices = [i for i, x in enumerate(full_dataset.is_real_list) if x]
        aug_global_indices = [i for i, x in enumerate(full_dataset.is_real_list) if not x]
        
        t_real = real_global_indices[:int(len(real_global_indices)*0.9)]
        v_real = real_global_indices[int(len(real_global_indices)*0.9):]
        
        t_aug = aug_global_indices[:int(len(aug_global_indices)*0.9)]
        v_aug = aug_global_indices[int(len(aug_global_indices)*0.9):]
        
        train_indices = t_real + t_aug
        val_indices = v_real + v_aug
        random.shuffle(train_indices)
        
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        train_is_real = [full_dataset.is_real_list[i] for i in train_indices]

    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4)

    train_selm(
        model, 
        train_subset, 
        val_loader, 
        100, 
        1e-4, 
        device, 
        SAVE_DIR, 
        mask_token_id, 
        CHECKPOINT_FILE,
        train_is_real=train_is_real
    )

if __name__ == "__main__":
    main()
