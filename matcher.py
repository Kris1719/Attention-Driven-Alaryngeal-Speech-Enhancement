import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.feature_dim = 1024 
        self.hidden_dim = 512
        self.num_heads = 8
        self.k_neighbors = 10
        self.dropout = 0.1
        
        self.batch_size = 64  
        self.learning_rate = 0.001
        self.num_epochs = 10 
        self.warmup_epochs = 5
        self.weight_decay = 1e-4
        
       
        self.reconstruction_weight = 1.0
        self.matching_weight = 1.0
        self.regularization_weight = 10.0   
        self.contrastive_weight = 50.0   
        
        # Multiple directories for alaryngeal features
        self.alaryngeal_dirs = [
            "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/real_001",
            # "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/001/converted",
            # "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/002/converted",
            # "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/003/converted",
            # "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/alaryngeal/004/converted",
        ]
        
        # Multiple directories for normal features
        self.normal_dirs = [
            "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/real_001",
            # "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/001",
            # "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/002",
            # "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/003",
            # "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/004",
        ]
        
        self.save_dir = '/workspace/Kris/matcher/checkpoints_eng_all_losses'
        self.log_dir = 'logs'
        self.feature_extension = '.pt' 

class SpeechFeatureDataset(Dataset):
    def __init__(self, a_features, n_features, config, cache_dir='knn_cache'):
        self.a_features = a_features
        self.n_features = n_features
        self.config = config
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, 'knn_cache.pt')
        
        # try to load from cache first
        if self.load_knn_cache():
            logger.info("Loaded KNN matches from cache")
        else:
            logger.info("Pre-computing KNN matches for faster training...")
            self.precompute_knn_matches()
            self.save_knn_cache()
    
    def load_knn_cache(self):
        """Load precomputed KNN matches from cache if available"""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            cache = torch.load(self.cache_file, map_location='cpu')
            
            # verify cache is compatible with current data
            if (cache['num_a_features'] == len(self.a_features) and
                cache['num_n_features'] == len(self.n_features) and
                cache['k_neighbors'] == self.config.k_neighbors):
                
                self.neighbor_distances = cache['neighbor_distances']
                self.neighbor_indices = cache['neighbor_indices']
                logger.info(f"Cache validated: {len(self.a_features)} samples, k={self.config.k_neighbors}")
                return True
            else:
                logger.info("Cache exists but is incompatible with current data, recomputing...")
                return False
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def save_knn_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        
        cache = {
            'neighbor_distances': self.neighbor_distances,
            'neighbor_indices': self.neighbor_indices,
            'num_a_features': len(self.a_features),
            'num_n_features': len(self.n_features),
            'k_neighbors': self.config.k_neighbors
        }
        
        torch.save(cache, self.cache_file)
        logger.info(f"Saved KNN cache to {self.cache_file}")
        
    def precompute_knn_matches(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        n_features_gpu = self.n_features.to(device).float()
        a_features_gpu = self.a_features.to(device).float()
        
        logger.info(f"Computing KNN matches on {device}")
        
        # pre-compute all matches in batches for memory efficiency
        batch_size = 500 if device.type == 'cuda' else 1000 
        all_distances = []
        all_indices = []
        
        with torch.no_grad(): 
            for i in tqdm(range(0, len(a_features_gpu), batch_size), desc="Computing KNN matches"):
                end_idx = min(i + batch_size, len(a_features_gpu))
                batch_features = a_features_gpu[i:end_idx]  # [batch_size, feature_dim]
                
                #cosine similarity matrix for this batch
                #normalize features for cosine similarity
                batch_norm = F.normalize(batch_features, p=2, dim=1)  # [batch_size, feature_dim]
                n_norm = F.normalize(n_features_gpu, p=2, dim=1)  # [num_n, feature_dim]
                
                similarity_matrix = torch.mm(batch_norm, n_norm.t())
                
                # convert to distance (1 - cosine_similarity)
                distance_matrix = 1.0 - similarity_matrix
                
                # get top-k nearest neighbors
                distances, indices = torch.topk(distance_matrix, k=self.config.k_neighbors, dim=1, largest=False)
                
                all_distances.append(distances.cpu())
                all_indices.append(indices.cpu())
                
                if device.type == 'cuda' and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        
       
        self.neighbor_distances = torch.cat(all_distances, dim=0).float()
        self.neighbor_indices = torch.cat(all_indices, dim=0).long()
        
        logger.info(f"Pre-computed KNN matches for {len(a_features_gpu)} samples using {device}")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    def __len__(self):
        return len(self.a_features)
    
    def __getitem__(self, idx):
        a_feat = self.a_features[idx]
        
        #use pre-computed neighbors
        indices = self.neighbor_indices[idx]
        distances = self.neighbor_distances[idx]
        
        #get neighbor features
        neighbor_features = self.n_features[indices]
        
        #random n feature for contrastive learning
        random_idx = torch.randint(0, len(self.n_features), (1,)).item()
        random_n = self.n_features[random_idx]
        
        return {
            'a': a_feat.float(),
            'neighbors': neighbor_features.float(),
            'distances': distances.float(),
            'random_n': random_n.float()
        }

class FeatureEnhancer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.enhancement_layers = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.feature_dim)
        )
        #print("the feature dimension from the last layer of feature enhancer:", config.feature_dim.shape)
        
        # residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # ensure input is float32
        x = x.float()
        enhanced = self.enhancement_layers(x)
        # learnable residual connection
        output = self.residual_weight * enhanced + (1 - self.residual_weight) * x
        return output

class AttentionMatcher(nn.Module):
    """Attention-based feature matching"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.feature_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(config.feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.feature_dim)
        )
        
    def forward(self, query_features, candidate_features):
        """
        Args:
            query_features: [batch_size, feature_dim] - a features
            candidate_features: [batch_size, k_neighbors, feature_dim] - n candidates
        """
        query_features = query_features.float()
        candidate_features = candidate_features.float()
        
        batch_size = query_features.shape[0]
        
        query_expanded = query_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
    
        attended_features, attention_weights = self.attention(
            query_expanded, candidate_features, candidate_features
        )
        
        #remove the sequence dimension
        attended_features = attended_features.squeeze(1)  # [batch_size, feature_dim]
        
        #residual connection and layer norm
        output = self.layer_norm(attended_features + query_features)
        

        ff_output = self.feed_forward(output)
        final_output = self.layer_norm(ff_output + output)
        
        return final_output, attention_weights.squeeze(1)  # [batch_size, k_neighbors]

class aSpeechConverter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.feature_enhancer = FeatureEnhancer(config)
        self.attention_matcher = AttentionMatcher(config)
        
        # quality predictor 
        self.quality_predictor = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, a_features, candidate_features):
        a_features = a_features.float()
        candidate_features = candidate_features.float()
        
        # enhance a features
        enhanced_a = self.feature_enhancer(a_features)
        
        # match with n speech features using attention
        matched_features, attention_weights = self.attention_matcher(
            enhanced_a, candidate_features
        )
        
        quality_score = self.quality_predictor(matched_features)
        
        return {
            'enhanced_a': enhanced_a,
            'matched_features': matched_features,
            'attention_weights': attention_weights,
            'quality_score': quality_score
        }

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, anchor, positive, negative):
        anchor = anchor.float()
        positive = positive.float()
        negative = negative.float()
        
        
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        
        
        pos_sim = F.cosine_similarity(anchor, positive, dim=1) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative, dim=1) / self.temperature
        
        # contrastive loss
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        return loss.mean()

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        

        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
        self.load_data()
    
        self.model = aSpeechConverter(config).to(self.device)
    
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = ContrastiveLoss()
    
        self.train_losses = []
        self.best_loss = float('inf')
        
    def load_features_from_directories(self, directories):
        """Load features from multiple directories"""
        all_feature_files = []
        all_features_list = []
        
        for directory in directories:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found, skipping: {directory}")
                continue
                
            feature_files = []
            for file in os.listdir(directory):
                if file.endswith(self.config.feature_extension):
                    feature_files.append(file)
            
            feature_files.sort()
            logger.info(f"Found {len(feature_files)} feature files in {directory}")
            
            for file in tqdm(feature_files, desc=f"Loading features from {directory}"):
                file_path = os.path.join(directory, file)
                
                if self.config.feature_extension == '.pt':
                    features = torch.load(file_path, map_location='cpu')
                
                features = features.float()
                
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                elif features.dim() == 3:
                    features = features.squeeze(0)
                    
                all_features_list.append(features)
                # store file with directory info for proper mapping
                all_feature_files.append(f"{os.path.basename(directory)}/{file}")
        
        if len(all_features_list) == 0:
            raise ValueError("No features loaded from any directory")
            
        combined_features = torch.cat(all_features_list, dim=0).float()
        logger.info(f"Loaded total {combined_features.shape[0]} feature vectors from {len(directories)} directories")
        logger.info(f"Feature dtype: {combined_features.dtype}")
        
        return combined_features, all_feature_files
    
    def load_data(self):
        logger.info("Loading WavLM features from multiple directories...")
        
        self.a_features, self.a_files = self.load_features_from_directories(
            self.config.alaryngeal_dirs
        )
        self.n_features, self.n_files = self.load_features_from_directories(
            self.config.normal_dirs
        )
        
        logger.info(f"Loaded alaryngeal features: {self.a_features.shape}")
        logger.info(f"Loaded normal features: {self.n_features.shape}")
        
        self.save_file_mappings()
        
        
        self.dataset = SpeechFeatureDataset(
            self.a_features, 
            self.n_features, 
            self.config
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  
            pin_memory=False  
        )
        
        logger.info(f"Created dataset with {len(self.dataset)} samples")
    
    def save_file_mappings(self):
        mappings = {
            'a_files': self.a_files,
            'n_files': self.n_files,
            'alaryngeal_dirs': self.config.alaryngeal_dirs,
            'normal_dirs': self.config.normal_dirs
        }
        
        mapping_path = os.path.join(self.config.log_dir, 'file_mappings.json')
        with open(mapping_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        
        logger.info(f"Saved file mappings to {mapping_path}")
        
    def compute_losses(self, batch, outputs):
        a_feat = batch['a'].to(self.device).float()
        neighbor_features = batch['neighbors'].to(self.device).float()
        random_n = batch['random_n'].to(self.device).float()
        distances = batch['distances'].to(self.device).float()
        
        enhanced_a = outputs['enhanced_a']
        matched_features = outputs['matched_features']
        attention_weights = outputs['attention_weights']
        quality_score = outputs['quality_score']
        print("quality score predicted:", quality_score)
        
       
        target_features = torch.mean(neighbor_features, dim=1)  # Average of neighbors
        reconstruction_loss = self.mse_loss(matched_features, target_features)
        
        
        reg_loss = self.mse_loss(enhanced_a, a_feat)
        
       
        contrastive_loss = self.contrastive_loss(
            matched_features, 
            target_features, 
            random_n
        )
        
        
        attention_reg = torch.mean(attention_weights * distances)
        
        with torch.no_grad():
            #compute cosine similarity between matched features and target
            matched_norm = F.normalize(matched_features, p=2, dim=1)
            target_norm = F.normalize(target_features, p=2, dim=1)
            quality_target = F.cosine_similarity(matched_norm, target_norm, dim=1).unsqueeze(1)
            # clamp to [0, 1] range for valid quality scores
            quality_target = quality_target.clamp(0, 1)
            print("quality_target", quality_target)
        
        quality_loss = self.mse_loss(quality_score, quality_target)
        
        
        # compute weighted losses
        weighted_reconstruction = self.config.reconstruction_weight * reconstruction_loss
        weighted_regularization = self.config.regularization_weight * reg_loss
        weighted_contrastive = self.config.contrastive_weight * contrastive_loss
        weighted_attention = 0.1 * attention_reg
        weighted_quality = 3.0 * quality_loss
        
        total_loss = (
            weighted_reconstruction +
            weighted_regularization +
            weighted_contrastive +
            weighted_attention  +
            weighted_quality
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': weighted_reconstruction,
            'regularization_loss': weighted_regularization,
            'contrastive_loss': weighted_contrastive,
            'attention_regularization': weighted_attention,
            'quality_loss': weighted_quality
        }
        
    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
           
            a_feat = batch['a'].to(self.device).float()
            neighbor_features = batch['neighbors'].to(self.device).float()
            
           
            outputs = self.model(a_feat, neighbor_features)
            
            
            losses = self.compute_losses(batch, outputs)
            
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_losses.append(losses['total_loss'].item())
            
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Recon': f"{losses['reconstruction_loss'].item():.4f}",
                'Reg': f"{losses['regularization_loss'].item():.4f}",
                'Con': f"{losses['contrastive_loss'].item():.4f}",
                'Q_Pred': f"{losses['quality_loss'].item():.4f}"
            })
            
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        
        return avg_loss
        
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with loss: {loss:.4f}")
            
        checkpoints = [f for f in os.listdir(self.config.save_dir) if f.startswith('checkpoint_epoch_')]
        if len(checkpoints) > 5:
            checkpoints.sort()
            oldest = os.path.join(self.config.save_dir, checkpoints[0])
            os.remove(oldest)
    
    def plot_training_progress(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.log_dir, 'training_progress.png'))
        plt.close()
        
    def train(self):
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            
            avg_loss = self.train_epoch(epoch)
        
            self.scheduler.step()
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, avg_loss)
                
            
            if (epoch + 1) % 20 == 0:
                self.plot_training_progress()
        
        
        self.save_checkpoint(self.config.num_epochs, avg_loss)
        self.plot_training_progress()
        
        logger.info("Training completed!")

def evaluate_model(model, a_features, n_features, config):
    model.eval()
    device = next(model.parameters()).device
    
   
    eval_dataset = SpeechFeatureDataset(a_features, n_features, config)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    similarities_before = []
    similarities_after = []
    quality_scores = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            a_feat = batch['a'].to(device).float()
            neighbor_features = batch['neighbors'].to(device).float()
        
            outputs = model(a_feat, neighbor_features)
            matched_features = outputs['matched_features']
            quality_score = outputs['quality_score']
            
            n_mean = torch.mean(neighbor_features, dim=1)
            
            sim_before = F.cosine_similarity(a_feat, n_mean, dim=1)
            sim_after = F.cosine_similarity(matched_features, n_mean, dim=1)
            
            similarities_before.extend(sim_before.cpu().numpy())
            similarities_after.extend(sim_after.cpu().numpy())
            quality_scores.extend(quality_score.cpu().numpy())

    avg_sim_before = np.mean(similarities_before)
    avg_sim_after = np.mean(similarities_after)
    improvement = avg_sim_after - avg_sim_before
    avg_quality = np.mean(quality_scores)
    
    logger.info(f"Average similarity before: {avg_sim_before:.4f}")
    logger.info(f"Average similarity after: {avg_sim_after:.4f}")
    logger.info(f"Improvement: {improvement:.4f}")
    logger.info(f"Average quality score: {avg_quality:.4f}")


    
    return {
        'similarity_before': avg_sim_before,
        'similarity_after': avg_sim_after,
        'improvement': improvement,
        'quality_score': avg_quality
    }

def main():
   
    config = Config()
    
    # check alaryngeal directories
    valid_a_dirs = [d for d in config.alaryngeal_dirs if os.path.exists(d)]
    if not valid_a_dirs:
        logger.error("No valid alaryngeal feature directories found")
        logger.info(f"Checked directories: {config.alaryngeal_dirs}")
        return
    
    # check normal directories
    valid_n_dirs = [d for d in config.normal_dirs if os.path.exists(d)]
    if not valid_n_dirs:
        logger.error("No valid normal feature directories found")
        logger.info(f"Checked directories: {config.normal_dirs}")
        return

    a_count = sum(
        len([f for f in os.listdir(d) if f.endswith(config.feature_extension)])
        for d in valid_a_dirs
    )
    n_count = sum(
        len([f for f in os.listdir(d) if f.endswith(config.feature_extension)])
        for d in valid_n_dirs
    )
    
    logger.info(f"Found {a_count} alaryngeal feature files across {len(valid_a_dirs)} directories")
    logger.info(f"Found {n_count} normal feature files across {len(valid_n_dirs)} directories")
    
    if a_count == 0 or n_count == 0:
        logger.error("No feature files found in one or both directory sets")
        return
    
    
    trainer = Trainer(config)

    trainer.train()
    
    
    logger.info("Evaluating final model...")
    evaluate_model(
        trainer.model, 
        trainer.a_features, 
        trainer.n_features, 
        config
    )
    
    logger.info("Generating final enhanced features...")
    trainer.model.eval()
    with torch.no_grad():
        eval_dataset = SpeechFeatureDataset(
            trainer.a_features, 
            trainer.n_features, 
            config
        )
        eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
        
        all_enhanced_features = []
        all_attention_weights = []
        
        for batch in eval_dataloader:
            a_feat = batch['a'].to(trainer.device).float()
            neighbor_features = batch['neighbors'].to(trainer.device).float()
            
            outputs = trainer.model(a_feat, neighbor_features)
            
            all_enhanced_features.append(outputs['matched_features'].cpu())
            all_attention_weights.append(outputs['attention_weights'].cpu())
        
        final_enhanced_features = torch.cat(all_enhanced_features, dim=0)
        final_attention_weights = torch.cat(all_attention_weights, dim=0)
        
    
def save_individual_enhanced_features(enhanced_features, original_files, original_features):
    """Save enhanced features as individual files matching the original structure"""
    output_dir = 'enhanced_features'
    os.makedirs(output_dir, exist_ok=True)

    current_idx = 0
    
    for i, original_file in enumerate(original_files):
        if i < len(original_files) - 1:
            file_length = len([f for f in original_files[:i+1]])
        
        enhanced_file_path = os.path.join(output_dir, f"enhanced_{original_file}")
        
      
        if current_idx < len(enhanced_features):
            torch.save(enhanced_features[current_idx:current_idx+1], enhanced_file_path)
            current_idx += 1
    
    logger.info(f"Saved individual enhanced features to {output_dir}")

if __name__ == "__main__":

    main()
