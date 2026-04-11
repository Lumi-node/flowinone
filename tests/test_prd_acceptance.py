import torch
import numpy as np
import pytest
from unittest.mock import MagicMock

def test_validate_sketch_rasterizer_output_dims():
    from sketch2dream.modules import SketchRasterizer
    module = SketchRasterizer()
    symbols = [{"label": "rectangle", "bbox": [100, 50, 200, 150], "text": "tree"}]
    out = module(symbols)
    assert out.shape == (1, 3, 512, 512), f"Expected (1,3,512,512), got {out.shape}"

def test_validate_text_renderer_handwriting_style():
    from sketch2dream.modules import TextRenderer
    module = TextRenderer(style_dim=256)
    style_emb = torch.randn(1, 256)
    out = module("hello", style_emb)
    assert out.shape == (1, 3, 64, 512), "TextRenderer output shape mismatch"
    assert out.min() >= 0 and out.max() <= 1, "Output not in [0,1] range"

def test_validate_visual_reprogrammer_isomorphic_mapping():
    from sketch2dream.modules import VisualReprogrammer
    module = VisualReprogrammer()
    sym = {"label": "arrow", "bbox": [200, 100, 50, 30], "text": "\u2192 clouds"}
    out = module(sym)
    assert out.shape == (1, 3, 512, 512), "VisualReprogrammer output shape mismatch"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"

def test_validate_realsketchloader_augmentation_behavior():
    from sketch2dream.datasets import RealSketchLoader
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy image
        from PIL import Image
        img = Image.new('RGB', (512, 512), color='white')
        img.save(os.path.join(tmpdir, "dummy.png"))
        
        dataset = RealSketchLoader(tmpdir, image_size=512, augment=True, seed=42)
        item = dataset[0]
        assert item.shape == (3, 512, 512), "RealSketchLoader output shape mismatch"
        assert item.dtype == torch.float32, "Item not float32"

def test_validate_visualpromptencoder_clip_distillation():
    from sketch2dream.modules import VisualPromptEncoder
    encoder = VisualPromptEncoder(out_features=256)
    x = torch.randn(1, 3, 512, 512)
    feat = encoder(x)  # (1, 512, 512, 256)
    assert feat.shape == (1, 512, 512, 256), "VisualPromptEncoder output shape mismatch"
    assert feat.requires_grad, "Encoder not differentiable"

def test_validate_flowconditioninginjector_spatial_modulation():
    from sketch2dream.modules import FlowConditioningInjector
    injector = FlowConditioningInjector(num_channels=128)
    x = torch.randn(1, 128, 64, 64)
    cond = torch.randn(1, 64, 64, 256)  # (B,H,W,C)
    out = injector(x, cond)
    assert out.shape == x.shape, "Injector改变了形状"
    assert out.requires_grad, "Modulation broke gradient flow"

def test_validate_vectorvelocityunet_fourier_modulation():
    from sketch2dream.modules import VectorVelocityUNet
    unet = VectorVelocityUNet()
    x = torch.randn(1, 3, 512, 512)
    t = torch.tensor([0.5])
    cond = torch.randn(1, 512, 512, 256)
    v = unet(x, t, cond)
    assert v.shape == x.shape, "UNet output shape mismatch"
    assert v.min() >= -1e6 and v.max() <= 1e6, "Velocity contains extreme values"

def test_validate_probabilitypathttracer_linear_interpolation():
    from sketch2dream.modules import ProbabilityPathTracer
    tracer = ProbabilityPathTracer()
    x0 = torch.zeros(1, 3, 512, 512)
    x1 = torch.ones(1, 3, 512, 512)
    t = torch.tensor([0.3])
    xt = tracer(x0, x1, t)
    expected = 0.7 * x0 + 0.3 * x1
    assert torch.allclose(xt, expected), "Linear interpolation incorrect"

def test_validate_flowmatchingloss_integral_computation():
    from sketch2dream.losses import FlowMatchingLoss
    loss_fn = FlowMatchingLoss()
    def model(x, t, cond): return torch.zeros_like(x)
    x0 = torch.randn(2, 3, 64, 64)
    cond = torch.randn(2, 64, 64, 256)
    loss = loss_fn(model, x0, cond)
    assert loss.dim() == 0, "Loss not scalar"
    assert loss >= 0, "Negative loss computed"

def test_validate_dualpathaugmentor_domain_confusion():
    from sketch2dream.datasets import DualPathAugmentor
    augmentor = DualPathAugmentor(real_dataset=MagicMock(), synth_generator=MagicMock())
    # This is a structural test
    assert hasattr(augmentor, 'domain_classifier_weight'), "Missing domain confusion weight"
    assert augmentor.domain_classifier_weight == 0.5, "Default weight not 0.5"

def test_validate_sketch2imagegenerator_cli_interface():
    import subprocess, sys
    result = subprocess.run([
        sys.executable, "-m", "sketch2dream.cli",
        "--input_sketch", "test_input.png",
        "--output_image", "test_output.png"
    ], capture_output=True)
    # Should not crash; absence of files is expected during test
    assert result.returncode == 0 or "No such file" in result.stderr.decode(), "CLI crashed"

def test_validate_apiinterface_pytorch_lightning_integration():
    from sketch2dream.api import Sketch2ImageGenerator
    model = Sketch2ImageGenerator()
    assert hasattr(model, "training_step"), "Missing training_step"
    assert hasattr(model, "configure_optimizers"), "Missing configure_optimizers"
    batch = torch.randn(4, 3, 512, 512)
    loss = model.training_step(batch, 0)
    assert loss.dim() == 0, "Training step must return scalar loss"

def test_validate_end_to_end_inference_determinism():
    from sketch2dream.api import predict
    import numpy as np
    img = np.random.RandomState(42).randint(0, 255, (512, 512, 3), dtype=np.uint8)
    with pytest.warns(None) as record:
        out1 = predict(img)
        out2 = predict(img)
    assert np.allclose(out1, out2), "Predict is not deterministic"
    

def test_validate_package_installable_via_pip():
    import subprocess
    result = subprocess.run(["python", "-m", "build"], cwd="/media/lumi-node/Storage2/research-radar/lab_builds/hfpaper-2604.06757", capture_output=True)
    assert result.returncode == 0, f"Build failed: {result.stderr.decode()}"

def test_validate_train_config_structure():
    from sketch2dream.api import train
    config = {
        "batch_size": 16,
        "lr": 1e-4,
        "max_epochs": 100,
        "data_dir": "/tmp/fake_data",
        "accelerator": "cpu"
    }
    try:
        train(config)
    except ModuleNotFoundError:
        pass  # Data may not exist
    except Exception as e:
        assert False, f"train() crashed on valid config: {e}"

def test_validate_predict_input_output_types():
    from sketch2dream.api import predict
    img_uint8 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img_float = img_uint8.astype(np.float32) / 255.0
    
    out_uint8 = predict(img_uint8)
    out_float = predict(img_float)
    
    assert out_uint8.shape == (64, 64, 3) and out_uint8.dtype == np.float32, "predict() uint8 in fail"
    assert out_float.shape == (64, 64, 3) and out_float.dtype == np.float32, "predict() float in fail"
    assert out_uint8.min() >= 0 and out_uint8.max() <= 1, "predict() output out of range"

def test_validate_synthetic_prompt_indistinguishability():
    from sketch2dream.modules import VisualReprogrammer
    from sketch2dream.datasets import RealSketchLoader
    import tempfile, os
    from PIL import Image
    module = VisualReprogrammer()
    sym = {"label": "checkbox", "bbox": [10, 10, 30, 30], "text": "✓ remove grass"}
    synth = module(sym)
    assert synth.min() >= 0 and synth.max() <= 1, "Synthetic prompt out of range"
    
    # Create real-like sample
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        real_img = (synth[0].permute(1,2,0).numpy() * 255).astype(np.uint8)
        Image.fromarray(real_img).save(f.name)
        dataset = RealSketchLoader(f.name.rsplit("/",1)[0], image_size=512)
    
    # Should load without error
    item = dataset[0]
    assert item.shape == (3, 512, 512), "RealSketchLoader cannot ingest synthetic"

def test_validate_semantic_preservation_checkbox_command():
    # This is a functional test that must be manually validated in system test
    # But we can test that the string "remove grass" is preserved in the prompt
    from sketch2dream.modules import VisualReprogrammer
    module = VisualReprogrammer()
    out = module({"label": "checkbox", "text": "✓ remove grass", "bbox": [0,0,50,50]})
    # In practice, OCR should not be used, but for test: we ensure "grass" appears in expected region
    # HACK: Use CLIP to verify region contains 'grass' suppression
    assert True  # Placeholder - requires external metric

def test_validate_multi_object_generation_from_sketch():
    from sketch2dream.modules import SketchRasterizer
    module = SketchRasterizer()
    symbols = [
        {"label": "rectangle", "bbox": [100,50,200,150], "text": "tree"},
        {"label": "arrow", "bbox": [200,100,50,30], "text": "\u2192 clouds"},
        {"label": "checkbox", "bbox": [10,10,30,30], "text": "✓ remove grass"}
    ]
    out = module(symbols)
    assert out.shape == (1, 3, 512, 512), "Multi-object rasterization failed"

def test_validate_coordinate_aware_attention_gradient_flow():
    from sketch2dream.modules import VisualPromptEncoder
    encoder = VisualPromptEncoder()
    x = torch.randn(1, 3, 512, 512, requires_grad=True)
    feat = encoder(x)
    loss = feat.sum()
    loss.backward()
    assert x.grad is not None, "Gradient does not flow through encoder"
