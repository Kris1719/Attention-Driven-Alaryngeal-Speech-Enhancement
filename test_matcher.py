
import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from typing import Union, List, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

from matcher import (
    Config, 
    aSpeechConverter,
    SpeechFeatureDataset,
    FeatureEnhancer,
    AttentionMatcher
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_path = '/workspace/Kris/matcher/checkpoints/best_model.pth'
        #self.output_dir = '/workspace/Kris/knn-vc/LM_detokenizer_package/Eng/SLM_matcher_feat_output'
        self.output_dir = "/workspace/Kris/knn-vc/LM_detokenizer_package/Eng/final_objective_metric_files/unseen/output_feat_ADFM_500"
        self.batch_size = 16 
        #self.alaryngeal_features_dir = '/workspace/Kris/knn-vc/LM_detokenizer_package/Eng/SLM_feat_output'
        self.alaryngeal_features_dir = "/workspace/Kris/knn-vc/LM_detokenizer_package/Eng/final_objective_metric_files/unseen/output_feats_SELM_500"
        self.normal_features_dir = "/workspace/Kris/knn-vc/English_Speech_Feat/augmented_english_alaryngeal/augmented_eng_feat/normal/001"  #eng
        #self.normal_features_dir = "/workspace/Kris/knn-vc/LM_detokenizer_package/Thai/normal_final_train" # thai
        
        #self.normal_features_dir = '/workspace/Kris/objective_evaluation/jimmy/prematched_normal_feat_1'
        
class AlaryngealSpeechInference:
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = InferenceConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.normal_features = None
        self.is_loaded = False
        
    def load_model(self, model_path: Optional[str] = None) -> bool:

        model_path = model_path or self.config.model_path
        
        if not os.path.exists(model_path):
            logger.error(f"Model checkpoint not found: {model_path}")
            return False
            
        try:
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, weights_only = False, map_location=self.device)
            if 'config' in checkpoint:
                checkpoint_config = checkpoint['config']
                for key, value in checkpoint_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            self.model = aSpeechConverter(self.config).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"Best validation loss: {checkpoint.get('val_loss', 'unknown')}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_normal_features(self, normal_features_dir: Optional[str] = None) -> bool:
        #features_dir = normal_features_dir or self.config.normal_features_dir
        features_dir = normal_features_dir
        
        if not os.path.exists(features_dir):
            logger.error(f"Normal features directory not found: {features_dir}")
            return False
            
        try:
            logger.info(f"Loading normal features from {features_dir}")
            feature_files = [f for f in os.listdir(features_dir) 
                           if f.endswith(self.config.feature_extension)]
            
            if not feature_files:
                logger.error(f"No feature files found in {features_dir}")
                return False
                
            features_list = []
            for file in tqdm(sorted(feature_files), desc="Loading normal features"):
                file_path = os.path.join(features_dir, file)
                
                if self.config.feature_extension == '.pt':
                    features = torch.load(file_path, map_location='cpu')
                elif self.config.feature_extension == '.npy':
                    features = torch.tensor(np.load(file_path), dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported feature extension: {self.config.feature_extension}")
                
                features = features.float()
                
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                elif features.dim() == 3:
                    features = features.squeeze(0)
                    
                features_list.append(features)
            
            self.normal_features = torch.cat(features_list, dim=0).float()
            logger.info(f"Loaded {self.normal_features.shape[0]} normal feature vectors")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading normal features: {str(e)}")
            return False
    
    def find_knn_matches(self, alaryngeal_features: torch.Tensor, k: int = None) -> Tuple[torch.Tensor, torch.Tensor]:

        k = k or self.config.k_neighbors
        
        if self.normal_features is None:
            raise ValueError("Normal features not loaded. Call load_normal_features() first.")
            
        device = alaryngeal_features.device
        normal_features_gpu = self.normal_features.to(device).float()
        
        # normalize features for cosine similarity
        alaryngeal_norm = F.normalize(alaryngeal_features, p=2, dim=1)
        normal_norm = F.normalize(normal_features_gpu, p=2, dim=1)

        print('alaryngeal_norm:', alaryngeal_norm.shape)
        print('normal_norm:', normal_norm.shape)
        
        # compute similarity matrix
        similarity_matrix = torch.mm(alaryngeal_norm, normal_norm.t())
        
        # convert to distance (1 - cosine_similarity)
        distance_matrix = 1.0 - similarity_matrix
        
        # get top-k nearest neighbors
        distances, indices = torch.topk(distance_matrix, k=k, dim=1, largest=False)
        
        # get the actual neighbor features
        neighbor_features = normal_features_gpu[indices]  # Shape: [batch_size, k, feature_dim]
        
        return neighbor_features, distances
    
    def infer_single(self, alaryngeal_features: torch.Tensor) -> Dict[str, torch.Tensor]:
       
        
        if self.normal_features is None:
            raise ValueError("Normal features not loaded. Call load_normal_features() first.")
        
        
        if alaryngeal_features.dim() == 1:
            alaryngeal_features = alaryngeal_features.unsqueeze(0)
        
        alaryngeal_features = alaryngeal_features.to(self.device).float()
        
        with torch.no_grad():
            # find k-nearest neighbors
            neighbor_features, distances = self.find_knn_matches(alaryngeal_features)
            
            # perform model inference
            outputs = self.model(alaryngeal_features, neighbor_features)
            
            return {
                'enhanced_alaryngeal': outputs['enhanced_alaryngeal'].cpu(),
                'matched_features': outputs['matched_features'].cpu(),
                'attention_weights': outputs['attention_weights'].cpu()
            }
    
    def infer_batch(self, alaryngeal_features: torch.Tensor, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:

        if self.normal_features is None:
            raise ValueError("Normal features not loaded. Call load_normal_features() first.")
        
        batch_size = batch_size or self.config.batch_size
        alaryngeal_features = alaryngeal_features.float()
        
        all_results = {
            'enhanced_alaryngeal': [],
            'matched_features': [],
            'attention_weights': []
           }
        
        with torch.no_grad():
            for i in tqdm(range(0, len(alaryngeal_features), batch_size), desc="Processing batches"):
                batch_end = min(i + batch_size, len(alaryngeal_features))
                batch_features = alaryngeal_features[i:batch_end].to(self.device)
                
                # find k-nearest neighbors for this batch
                neighbor_features, distances = self.find_knn_matches(batch_features)
                
                outputs = self.model(batch_features, neighbor_features)
                
                all_results['enhanced_alaryngeal'].append(outputs['enhanced_alaryngeal'].cpu())
                all_results['matched_features'].append(outputs['matched_features'].cpu())
                all_results['attention_weights'].append(outputs['attention_weights'].cpu())
               
        
        
        return {key: torch.cat(values, dim=0) for key, values in all_results.items()}
    
    def infer_from_file(self, file_path: str) -> Dict[str, torch.Tensor]:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
        if file_path.endswith('.pt'):
            features = torch.load(file_path, map_location='cpu')
        elif file_path.endswith('.npy'):
            features = torch.tensor(np.load(file_path), dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
        features = features.float()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        elif features.dim() == 3:
            features = features.squeeze(0)
        
        if len(features) == 1:
            return self.infer_single(features)
        else:
            return self.infer_batch(features)
    
    def compute_similarity_improvement(self, results: Dict[str, torch.Tensor]) -> Dict[str, float]:
        original = results['original_alaryngeal']
        enhanced = results['matched_features']
        targets = results['neighbor_features'][:, 0, :]  # closest neighbor as target
        
        sim_before = F.cosine_similarity(original, targets, dim=1)
        sim_after = F.cosine_similarity(enhanced, targets, dim=1)
        
        return {
            'similarity_before_mean': sim_before.mean().item(),
            'similarity_after_mean': sim_after.mean().item(),
            'similarity_improvement_mean': (sim_after - sim_before).mean().item(),
            'similarity_before_std': sim_before.std().item(),
            'similarity_after_std': sim_after.std().item(),
            'improvement_per_sample': (sim_after - sim_before).tolist()
        }
    
    def save_results(self, results: Dict[str, torch.Tensor], output_dir: Optional[str] = None):
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        
        torch.save(results['enhanced_alaryngeal'], os.path.join(output_dir, 'enhanced_features.pt'))
        torch.save(results['matched_features'], os.path.join(output_dir, 'matched_features.pt'))
        return "save results"


def load_features_from_directory(directory: str, feature_extension: str = '.pt') -> torch.Tensor:
    feature_files = [f for f in os.listdir(directory) if f.endswith(feature_extension)]
    features_list = []
    
    for file in tqdm(sorted(feature_files), desc=f"Loading features from {directory}"):
        file_path = os.path.join(directory, file)
        
        if feature_extension == '.pt':
            features = torch.load(file_path, map_location='cpu')
        elif feature_extension == '.npy':
            features = torch.tensor(np.load(file_path), dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported feature extension: {feature_extension}")
        
        features = features.float()
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        elif features.dim() == 3:
            features = features.squeeze(0)
            
        features_list.append(features)
    
    return torch.cat(features_list, dim=0).float()


def main():
    config = InferenceConfig()
    inference = AlaryngealSpeechInference(config)
    
    logger.info("Loading model...")
    if not inference.load_model():
        return
    logger.info("Loading normal features...")
    if not inference.load_normal_features(config.normal_features_dir):
        return
    logger.info("Starting inference on individual files...")
    
    input_dir = config.alaryngeal_features_dir
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    feature_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')] # or .npy
    
    for file_name in tqdm(feature_files, desc="Processing"):
        file_path = os.path.join(input_dir, file_name)
        results = inference.infer_from_file(file_path)
        base_name = os.path.splitext(file_name)[0]
        save_name = f"{base_name}_enhanced.pt"
        save_path = os.path.join(output_dir, save_name)
        
        torch.save(results['matched_features'], save_path)
        
    logger.info(f"Inference completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
