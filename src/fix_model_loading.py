import torch
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent))

print("[*] Testing model loading...")

try:
    from backend.services.vfn_model import load_vfn_model
    
    model, preprocess = load_vfn_model()
    print("[✓] Model loaded successfully!")
    print(f"    Model type: {type(model)}")
    print(f"    Has backbone: {hasattr(model, 'backbone')}")
    

    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(test_input)
        print(f"    Output keys: {output.keys()}")
        

        if hasattr(model, 'backbone'):
            features = model.backbone(test_input)
            print(f"    Feature shape: {features.shape}")
            print("[✓] Feature extraction works!")
    
except Exception as e:
    print(f"[✗] Error: {e}")
    import traceback
    traceback.print_exc()



