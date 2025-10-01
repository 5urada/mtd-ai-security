import torch

checkpoint = torch.load('runs/exp3/ckpt_latest.pt', map_location='cpu', weights_only=False)
print("Keys in checkpoint:", checkpoint.keys())
print("\nPolicy state_dict keys:")
for key in checkpoint['policy_state_dict'].keys():
    print(f"  {key}")