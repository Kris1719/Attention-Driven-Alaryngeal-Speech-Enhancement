from inf_speech_lm import SELMTokenEnhancer, enhance_single_file, batch_enhance
from test_matcher import AlaryngealSpeechInference, InferenceConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import os
import numpy as np
import torchaudio
import argparse
if not hasattr(np, 'complex'):
    np.complex = complex

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ADASE:
    def __init__(self, language="eng", input_alar_dir = None, normal_dir = None, models_dir = None, out_base_dir = None):
        self.language = language
        self.input_alar_dir = input_alar_dir
        self.normal_dir = normal_dir
        self.models_dir = models_dir
        self.out_base_dir = out_base_dir


        if self.language == 'thai':
            exp_dir = f"{self.out_base_dir}/thai"
            self.input_wavs_dir = self.input_alar_dir
            self.selm_model_path = f"{self.models_dir}/last_checkpoint_curriculum.pt"
           
            self.matcher_model_path = f"{self.models_dir}/best_model.pth"
            self.wavlm_feats_dir = f"{exp_dir}/alar_wavlm_feats"
            self.normal_feats_dir= f"{exp_dir}/normal_wavlm_feats"
            self.enhanced_tokens_dir = f"{exp_dir}/enhanced_tokens"
            self.matched_feats_dir = f"{exp_dir}/matched_feats"
            self.output_wavs_dir = f"{exp_dir}/output_wavs"
            self.selm_layer_ids = [6]
            self.selm_num_layers = 4
            self.selm_d_model = 512 
            self.selm_max_length = 500

        else:
            exp_dir = f"{self.out_base_dir}/eng"
            self.input_wavs_dir = self.input_alar_dir
            self.selm_model_path = f"{self.models_dir}/last_checkpoint_curriculum.pt"
           
            self.matcher_model_path = f"{self.models_dir}/best_model.pth"
            self.wavlm_feats_dir = f"{exp_dir}/alar_wavlm_feats"
            self.normal_feats_dir= f"{exp_dir}/normal_wavlm_feats"
            self.enhanced_tokens_dir = f"{exp_dir}/enhanced_tokens"
            self.matched_feats_dir = f"{exp_dir}/matched_feats"
            self.output_wavs_dir = f"{exp_dir}/output_wavs"
            self.selm_layer_ids = [6]
            self.selm_num_layers = 4
            self.selm_d_model = 512 
            self.selm_max_length = 500

    def extract_wavlm_features(self, dwavlm):
        print("STEP 1: Extracting WavLM Features")
        os.makedirs(self.wavlm_feats_dir, exist_ok=True)
        os.makedirs(self.normal_feats_dir, exist_ok=True)
        
        alar_wavs = glob.glob(os.path.join(self.input_wavs_dir, "*.wav"))
        norm_wavs = glob.glob(os.path.join(self.normal_dir, "*.wav"))
        print(f"Found {len(alar_wavs)} wav files")

        ###extract features for alaryngeal speech
        for wav in tqdm(alar_wavs, desc="Extracting Alar features"):
            sig, sample_rate = torchaudio.load(wav)
            sig = torchaudio.functional.resample(sig, sample_rate, dwavlm.sample_rate)
            sig = sig.to(device)
            
            with torch.no_grad():
                feats = dwavlm.sig_to_feats(sig)
            
            feats = feats.squeeze(-1)[0]
            
            file_name = os.path.basename(wav).replace(".wav", ".pt")
            save_path = os.path.join(self.wavlm_feats_dir, file_name)
            torch.save(feats.cpu(), save_path)
        
        print(f"Features saved to: {self.wavlm_feats_dir}")
        ###extract features from normal speech
        for wav in tqdm(norm_wavs, desc="Extracting Norm features"):
            sig, sample_rate = torchaudio.load(wav)
            sig = torchaudio.functional.resample(sig, sample_rate, dwavlm.sample_rate)
            sig = sig.to(device)
            
            with torch.no_grad():
                feats = dwavlm.sig_to_feats(sig)
            
            feats = feats.squeeze(-1)[0]
            
            file_name = os.path.basename(wav).replace(".wav", ".pt")
            save_path = os.path.join(self.normal_feats_dir, file_name)
            torch.save(feats.cpu(), save_path)
        
        print(f"Features saved to: {self.normal_feats_dir}")

    def selm_enhancement(self, dwavlm):
        print("STEP 2: SELM Token Enhancement") 
        os.makedirs(self.enhanced_tokens_dir, exist_ok=True)
        
        print(f"Loading SELM: layers={self.selm_num_layers}, d_model={self.selm_d_model}")
        
        enhancer = SELMTokenEnhancer(
            model_path=self.selm_model_path,
            device=str(device),  
            layer_ids=self.selm_layer_ids,
            num_layers=self.selm_num_layers,
            d_model=self.selm_d_model
        )
        
        feat_files = glob.glob(os.path.join(self.wavlm_feats_dir, "*.pt"))
        print(f"Found {len(feat_files)} feature files to enhance")
        
        for feat_file in tqdm(feat_files, desc="Enhancing features"):
            try:
                enhanced_tokens = enhancer.features_to_enhanced_tokens(feat_file, self.selm_max_length)
                tokens_for_codec = enhanced_tokens.unsqueeze(0).unsqueeze(-1).to(device)
                
                with torch.no_grad():
                    enhanced_feats = dwavlm.toks_to_qfeats(tokens_for_codec)
                
                enhanced_feats = enhanced_feats.squeeze(0).squeeze(-1)
                
                file_name = os.path.basename(feat_file)
                output_name = f"selm_enhanced_{file_name}"
                output_path = os.path.join(self.enhanced_tokens_dir, output_name)
                
                torch.save(enhanced_feats.cpu(), output_path)
                
            except Exception as e:
                print(f"Error: {feat_file}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Enhanced features saved to: {self.enhanced_tokens_dir}")

    def ADFM(self):
        print("STEP 3: Feature Matching (ADFM)")
        os.makedirs(self.matched_feats_dir, exist_ok=True)
        
        inf_config = InferenceConfig()
        inf_config.model_path = self.matcher_model_path
        
        inference = AlaryngealSpeechInference(inf_config)
    
        print("Loading feature matcher model...")
        if not inference.load_model(model_path=self.matcher_model_path):
            raise RuntimeError("Failed to load feature matcher model")
        
        print("Loading normal speech features...")
        # FIX: Instead of manual loop parsing, use the built-in method from test_matcher
        if not inference.load_normal_features(self.normal_feats_dir):
            raise RuntimeError(f"Failed to load normal features from {self.normal_feats_dir}")
       
        # Find files inside the enhanced tokens directory
        feature_files = glob.glob(os.path.join(self.enhanced_tokens_dir, "*.pt"))
        print(f"Found {len(feature_files)} feature files to process")
        
        for feat_file in tqdm(feature_files, desc="Matching features"):
            try:
                results = inference.infer_from_file(feat_file)
                base_name = os.path.basename(feat_file)
                
                if base_name.startswith("selm_enhanced_"):
                    base_name = base_name.replace("selm_enhanced_", "")
                
                output_name = f"matched_{base_name}"
                # FIX: Changed 'config.matched_feats_dir' to 'self.matched_feats_dir'
                output_path = os.path.join(self.matched_feats_dir, output_name)
                
                torch.save(results['matched_features'], output_path)
            except Exception as e:
                print(f"Error processing {feat_file}: {e}")
                
        # FIX: Changed 'config.matched_feats_dir' to 'self.matched_feats_dir'
        print(f"Matched features saved to: {self.matched_feats_dir}")

    # def ADFM(self):
    #     print("STEP 3: Feature Matching (ADFM)")
    #     os.makedirs(self.matched_feats_dir, exist_ok=True)
    #     inf_config = InferenceConfig()
    #     inf_config.model_path = self.matcher_model_path
        
    #     inference = AlaryngealSpeechInference(inf_config)
    #     #inference.load_model(model_path = self.matcher_model_path)
    
    #     print("Loading feature matcher model...")
    #     if not inference.load_model(model_path=self.matcher_model_path):
    #         raise RuntimeError("Failed to load feature matcher model")
        
    #     print("Loading normal speech features from multiple directories...")
    #     all_normal_features = []
        
    #     feat_files = glob.glob(os.path.join(self.normal_feats_dir, "*.pt"))
    #     for f in feat_files:
    #         feat = torch.load(f, map_location='cpu')
    #         if feat.dim() == 2:
    #             all_normal_features.append(feat)
    #         else:
    #             all_normal_features.append(feat.squeeze())
        
    #     if not all_normal_features:
    #         raise RuntimeError("No normal features found in any directory")
        
    #     # inference.normal_features = torch.cat(all_normal_features, dim=0).to(inference.device)
    #     # print(f"Loaded {len(inference.normal_features)} normal feature frames from {len(self.normal_features_dirs)} directories")
       
    #     feature_files = glob.glob(os.path.join(self.enhanced_tokens_dir, "*.pt"))
    #     print(f"Found {len(feature_files)} feature files to process")
    #     print("I am hereeeee")
    #     for feat_file in tqdm(feature_files, desc="Matching features"):
    #         try:
    #             results = inference.infer_from_file(feat_file)
    #             base_name = os.path.basename(feat_file)
    #             if base_name.startswith("selm_enhanced_"):
    #                 base_name = base_name.replace("selm_enhanced_", "")
    #             output_name = f"matched_{base_name}"
    #             output_path = os.path.join(config.matched_feats_dir, output_name)
                
    #             torch.save(results['matched_features'], output_path)
    #         except Exception as e:
    #             print(f"Error processing {feat_file}: {e}")
    #     print(f"Matched features saved to: {config.matched_feats_dir}")
       


    def vocoder(self, dwavlm): 
        print("STEP 4: Vocoder") 
        feat_dir = self.matched_feats_dir 
        feat_files = glob.glob(os.path.join(feat_dir, "*.pt")) 
        os.makedirs(self.output_wavs_dir, exist_ok=True)
       
        
        for feat_path in tqdm(feat_files, desc="Synthesizing audio"): 
            try: 
                feat_from_speech_lm = torch.load(feat_path, map_location='cpu') 
                rec_feats = feat_from_speech_lm.unsqueeze(0).unsqueeze(-1) 
                rec_feats = rec_feats.to(device) 
                
                with torch.no_grad(): 
                    rec_sig = dwavlm.feats_to_sig(rec_feats) 
                    
                base_name = os.path.splitext(os.path.basename(feat_path))[0] 
                wav_output_path = os.path.join(self.output_wavs_dir, f"{base_name}.wav") 
                torchaudio.save(wav_output_path, rec_sig[:, 0].cpu(), dwavlm.sample_rate) 
            except Exception as e: 
                print(f"Error: {feat_path}: {str(e)}") 
                continue


def test_ADASE(language, input_alar_dir, normal_dir, models_dir, out_base_dir):
    print("\nLoading WavLM model...")
    dwavlm = torch.hub.load(
        "lucadellalib/discrete-wavlm-codec", 
        "discrete_wavlm_large", 
        layer_ids=[6]
    )
    dwavlm.eval().requires_grad_(False)
    dwavlm = dwavlm.to(device)

    ADASE_instance = ADASE(language=language, input_alar_dir=input_alar_dir, normal_dir=normal_dir, models_dir=models_dir, out_base_dir=out_base_dir)

    wavlm_feats = ADASE_instance.extract_wavlm_features(dwavlm)
    enhanced_feats = ADASE_instance.selm_enhancement(dwavlm)
    matched_feats = ADASE_instance.ADFM()
    out_wavs = ADASE_instance.vocoder(dwavlm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str, default="thai", help="Path to input alardir")
    parser.add_argument("--input_alar_dir", type=str, required=True, help="Path to input alaryngeal dir")
    parser.add_argument("--normal_dir", type=str, required=True, help="Path to normal directory")
    parser.add_argument("--models_dir", type=str, required=True, help="Path to models directory")
    parser.add_argument("--out_base_dir", type=str, required=True, help="Path to output base directory")

    args = parser.parse_args()
    
    
    test_ADASE(
        language=args.language,
        input_alar_dir = args.input_alar_dir,
        normal_dir=args.normal_dir,
        models_dir=args.models_dir,
        out_base_dir=args.out_base_dir
        
    )
    

          
    
        