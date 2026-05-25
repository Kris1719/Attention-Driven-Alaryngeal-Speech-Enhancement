import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from speech_lm_eng import SELMSpeechLM, CodecTokenizer
import torch.nn as nn
import math


if not hasattr(np, 'complex'):
    np.complex = complex
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        div_term = torch.exp(position)
        pe[:, 0::2] = torch.sin(torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) * div_term[0])
        pe[:, 1::2] = torch.cos(torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) * div_term[0]) 
    
        pass 

def load_selm_model(checkpoint_path, device='cuda', num_layers=4, d_model=512):
   
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint.get('vocab_size', 1024) 
    if 'audio_embedding.weight' in checkpoint['model_state_dict']:
        emb_shape = checkpoint['model_state_dict']['audio_embedding.weight'].shape
        actual_model_vocab_size = emb_shape[0]
        print(f"Detected vocab size from weights: {actual_model_vocab_size}")
        vocab_size = actual_model_vocab_size
        
   
    model = SELMSpeechLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=16,
        num_layers=num_layers,
        dropout=0.0 
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


class SELMTokenEnhancer:
    """
    SELM token enhancer - outputs enhanced tokens only.
    """
    def __init__(self, model_path, device='cuda', layer_ids=[6], 
                 num_layers=4, d_model=512, finetuned_codec_path=None):
        self.device = device
        self.layer_ids = layer_ids
        
        print("Loading SELM components...")
        print(f"Model config: num_layers={num_layers}, d_model={d_model}")    
        self.tokenizer = CodecTokenizer(layer_ids=layer_ids)
        
        self.model = load_selm_model(model_path, device, num_layers=num_layers, d_model=d_model)
        
        print(f"SELM loaded with vocab_size={self.model.vocab_size}, d_model={self.model.d_model}")
    
    def features_to_enhanced_tokens(self, features, max_length=300):
        print("Converting features to tokens...")
        
        if isinstance(features, str):
            features = torch.load(features, map_location='cpu', weights_only=True).float()
        
        original_tokens = self.tokenizer.features_to_tokens(features)
        
        if len(original_tokens.shape) > 1:
            original_tokens = original_tokens.flatten()
        
        print(f"Original tokens shape: {original_tokens.shape}")
        
        print("Enhancing tokens with SELM...")
        enhanced_tokens = self._enhance_tokens(original_tokens, max_length)
        
        print(f"Enhanced tokens shape: {enhanced_tokens.shape}")
        
        return enhanced_tokens
    
    def _enhance_tokens(self, tokens, max_length):
        device = next(self.model.parameters()).device
        if len(tokens) > max_length:
            print(f"Segmenting long sequence (length: {len(tokens)})")
            enhanced_segments = []
            stride = max_length 
            
            for start in range(0, len(tokens), stride):
                end = min(start + max_length, len(tokens))
                segment = tokens[start:end]
                if len(segment) < max_length:
                    pad_size = max_length - len(segment)
                    mask_token_id = self.model.vocab_size - 1
                    padding = torch.full((pad_size,), mask_token_id, dtype=segment.dtype)
                    segment = torch.cat([segment, padding])
                
            
                segment = segment.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = self.model(segment)
                    enhanced_segment = torch.argmax(logits, dim=-1).squeeze(0)
                
                
                if start + max_length > len(tokens):
                    actual_length = len(tokens) - start
                    enhanced_segment = enhanced_segment[:actual_length]
                
                enhanced_segments.append(enhanced_segment.cpu())
        
            enhanced_tokens = torch.cat(enhanced_segments)
            
          
            enhanced_tokens = enhanced_tokens[:len(tokens)]
        else:
           
            if len(tokens) < max_length:
                pad_size = max_length - len(tokens)
                mask_token_id = self.model.vocab_size - 1
                padding = torch.full((pad_size,), mask_token_id, dtype=tokens.dtype)
                padded_tokens = torch.cat([tokens, padding])
            else:
                padded_tokens = tokens
            
            padded_tokens = padded_tokens.unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = self.model(padded_tokens)
                enhanced_padded = torch.argmax(logits, dim=-1).squeeze(0)
            
           
            enhanced_tokens = enhanced_padded[:len(tokens)].cpu()
        
        return enhanced_tokens



def enhance_single_file(enhancer, input_file, output_file, max_length=300):
    print(f"Processing feature file: {input_file}")
    enhanced_tokens = enhancer.features_to_enhanced_tokens(input_file, max_length)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(enhanced_tokens, output_file)
    print(f"Enhanced tokens saved to: {output_file}")
    
    return enhanced_tokens


def batch_enhance(enhancer, input_dir, output_dir, max_length=300):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    feature_files = list(input_dir.glob("*.pt"))
    print(f"Found {len(feature_files)} feature files to process")
    
    for file_path in feature_files:
        try:
            print(f"Processing {file_path.name}...")
            output_file = output_dir / f"selm_enhanced_tokens_{file_path.stem}.pt"
            enhance_single_file(enhancer, str(file_path), str(output_file), max_length)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    parser.add_argument('--model', required=True)
    parser.add_argument('--num_layers', type=int, default=4) # Default to 4
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    enhancer = SELMTokenEnhancer(args.model, args.device, num_layers=args.num_layers, d_model=args.d_model)
    if args.input_file and args.output_file:
        enhance_single_file(enhancer, args.input_file, args.output_file)

if __name__ == "__main__":
    main()




