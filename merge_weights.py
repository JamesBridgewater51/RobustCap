import torch
import os
import argparse
from config import paths
from net.sig_mp_aist_nymeria import Net

def merge_weights(target_net_name=None):
    if target_net_name is None:
        target_net_name = Net.name
        
    print(f"Merging weights for {target_net_name}...")
    base_dir = os.path.join(paths.weight_dir, target_net_name)
    
    # List of stages to merge. Note: rnn6 and rnn7 are also part of the full pipeline if used.
    # rnn8 is optional/experimental often.
    stages = ['rnn2', 'rnn3', 'rnn4', 'rnn6', 'rnn7', 'rnn8']
    
    full_state_dict = {}
    missing_stages = []

    for stage in stages:
        ckpt_path = os.path.join(base_dir, stage, 'best_weights.pt')
        if not os.path.exists(ckpt_path):
            print(f"Warning: Missing checkpoint for {stage} at {ckpt_path}")
            missing_stages.append(stage)
            continue
            
        print(f"Loading {stage} from {ckpt_path}...")
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            
            # Check if it's a full checkpoint dict or just state_dict
            if 'model' in state:
                state = state['model']
            
            # Add prefix because training scripts use sub-modules directly
            # e.g., train_rnn2 uses 'net.rnn2', so saved weights keys are "linear1.weight"
            # But full Net needs "rnn2.linear1.weight"
            for key, value in state.items():
                new_key = f"{stage}.{key}"
                full_state_dict[new_key] = value
                
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")

    if not full_state_dict:
        print("Error: No weights loaded. Merging failed.")
        return

    print(f"Merged state dict has {len(full_state_dict)} keys.")
    if missing_stages:
        print(f"Warning: The following stages were missing: {missing_stages}")
        
    save_path = os.path.join(base_dir, 'best_weights.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(full_state_dict, save_path)
    print(f"Success! Saved merged checkpoint to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge separate RNN checkpoints into one Net checkpoint')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment/net (default: sig_mp_nymeria)')
    args = parser.parse_args()
    
    merge_weights(args.name)
