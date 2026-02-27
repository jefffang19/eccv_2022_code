import torch
import os
from model import Net

def export_to_onnx():
    save_model_path = "model_weights"
    weights_path = os.path.join(save_model_path, 'weights_used_in_paper.pth')
    
    print("Loading model...")
    model = Net()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    print("Model loaded successfully.")

    # Create dummy input. The model expects (batch_size, 17, 3, 300, 300)
    # where 17 is the number of patches (1 whole image + 16 patches)
    batch_size = 1
    dummy_input = torch.randn(batch_size, 17, 3, 300, 300)
    
    # The forward pass expects (x, cls) but cls is currently unused in the code
    dummy_cls = torch.tensor([0])

    onnx_path = "model.onnx"
    print(f"Exporting to {onnx_path}...")
    
    # We export with dynamic axes to allow different batch sizes during inference
    torch.onnx.export(
        model, 
        (dummy_input, dummy_cls), 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input', 'cls'], 
        output_names=['patch_logits', 'cam_map', 'whole_predict'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'cls': {0: 'batch_size'},
            'patch_logits': {0: 'batch_size'},
            'cam_map': {0: 'batch_size'},
            'whole_predict': {0: 'batch_size'}
        }
    )
    
    print("ONNX export complete.")

if __name__ == "__main__":
    export_to_onnx()
