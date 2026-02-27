import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model import Net

# Fixed batch size — must stay 1 for Jetson Nano TRT 8.2.1 compatibility.
EXPORT_BATCH_SIZE = 1

class NoCamNet(nn.Module):
    """
    Classification-only wrapper for TRT 8.2.1 on Jetson Nano (Tegra X1, SM 5.3).

    Design rules to avoid kgen_prelim_transforms assertion crash:
      1. No ReduceSum — eliminated by using 2D MatMul instead of .sum(dim=1).
      2. No GlobalAveragePool — replaced with explicit AveragePool(kernel=10),
         which exports as ONNX AveragePool rather than GlobalAveragePool.
      3. No 3D MatMul — all Linear layers applied on 2D tensors only.
      4. No CAM branch — never entered.
      5. All shapes are static constants (EXPORT_BATCH_SIZE=1 folded in).
    """
    def __init__(self, net):
        super().__init__()
        self.feature_extract = net.feature_extract
        self.wholeimg_fc    = net.wholeimg_fc
        self.patchwise_fc  = net.patchwise_fc
        self.patch_aggr_fc = net.patch_aggr_fc

    def forward(self, x, cls):
        # ── Reshape: (1, 17, 3, 300, 300) → (17, 3, 300, 300) ─────────────
        x = x.view(17, 3, 300, 300)

        # ── ResNet18 backbone ───────────────────────────────────────────────
        features = self.feature_extract(x)          # (17, 512, 10, 10)

        # ── Spatial pooling — explicit AveragePool(k=10) instead of
        #    GlobalAveragePool to avoid a TRT 8.2.1 myelin lowering bug ─────
        x = F.avg_pool2d(features, kernel_size=10)  # (17, 512,  1,  1)
        x = x.view(17, 512)                         # (17, 512)

        # ── Split into whole-image token and patch tokens ───────────────────
        whole_x = x[0:1, :]    # (1,  512)   — whole image (index 0)
        patch_x = x[1:17, :]   # (16, 512)   — 16 patches  (index 1..16)

        # ── Whole-image classifier: standard 2D Gemm ───────────────────────
        whole_logits = self.wholeimg_fc(whole_x)    # (1, 2)

        # ── Patch classifier: 2D Gemm on (16,512) ──────────────────────────
        patch_logits_2d      = self.patchwise_fc(patch_x)      # (16, 2)
        patch_logits_collect = patch_logits_2d.view(1, 16, 2)  # (1, 16, 2)

        # ── Attention weights via Softmax ───────────────────────────────────
        scores  = patch_logits_collect[0, :, 1].view(1, 16)    # (1, 16)
        weights = F.softmax(scores, dim=1)                      # (1, 16)

        # ── Weighted aggregation via 2D MatMul — NO ReduceSum! ──────────────
        # (1, 16) @ (16, 512) = (1, 512)
        patch_features = torch.mm(weights, patch_x)            # (1, 512)

        # ── Patch-aggregated classifier: standard 2D Gemm ──────────────────
        patch_logists = self.patch_aggr_fc(patch_features)     # (1, 2)

        whole_predict = whole_logits + patch_logists            # (1, 2)

        return patch_logits_collect, whole_predict


def export_to_onnx():
    save_model_path = "model_weights"
    weights_path    = os.path.join(save_model_path, 'weights_used_in_paper.pth')

    print("Loading model...")
    base_model = Net()
    base_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    base_model.eval()

    model = NoCamNet(base_model)
    model.eval()
    print("Model loaded (no ReduceSum, no GlobalAveragePool, no 3D MatMul, no CAM).")

    B             = EXPORT_BATCH_SIZE
    dummy_input   = torch.randn(B, 17, 3, 300, 300)
    dummy_cls     = torch.zeros(B, dtype=torch.long)

    # Sanity-check output shapes
    with torch.no_grad():
        patch_logits, whole_predict = model(dummy_input, dummy_cls)
    print(f"patch_logits shape : {patch_logits.shape}")    # (1, 16, 2)
    print(f"whole_predict shape: {whole_predict.shape}")   # (1,  2)

    onnx_path = "model.onnx"
    print(f"Exporting to {onnx_path} (static batch={B}, no dynamic axes)...")

    torch.onnx.export(
        model,
        (dummy_input, dummy_cls),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input', 'cls'],
        output_names=['patch_logits', 'whole_predict'],
        # Fully static — no dynamic_axes
    )

    print("ONNX export complete.")
    print(f"Batch size is fixed at {B}. Set batch=1 in your test DataLoader.")


if __name__ == "__main__":
    export_to_onnx()
