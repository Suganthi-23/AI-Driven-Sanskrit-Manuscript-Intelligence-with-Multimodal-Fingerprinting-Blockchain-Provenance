import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.services.textual_service import extract_text_features, get_text_embedding
from backend.services.fingerprint_service import extract_fingerprint

def test_text_extraction(image_path: str):
    print(f"\n{'='*60}")
    print("TEST 1: Text Extraction (OCR)")
    print(f"{'='*60}")
    
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        print(f"[*] Processing image: {image_path}")
        text_features = extract_text_features(image_bytes)
        
        print(f"\n[✓] Text Extraction Results:")
        print(f"    Extracted text: {text_features['text'][:100]}..." if len(text_features['text']) > 100 else f"    Extracted text: {text_features['text']}")
        print(f"    Confidence: {text_features['text_confidence']:.3f}")
        print(f"    Has text: {text_features['has_text']}")
        print(f"    Embedding dimension: {len(text_features['text_embedding'])}")
        
        return text_features
    except Exception as e:
        print(f"[!] Error: {e}")
        return None


def test_text_embedding(text: str):
    print(f"\n{'='*60}")
    print("TEST 2: Text Embedding Generation")
    print(f"{'='*60}")
    
    try:
        print(f"[*] Generating embedding for text: '{text[:50]}...'")
        embedding = get_text_embedding(text)
        
        print(f"\n[✓] Embedding Results:")
        print(f"    Dimension: {len(embedding)}")
        print(f"    Norm: {sum(embedding**2)**0.5:.4f}")
        print(f"    Sample values: {embedding[:5]}")
        
        return embedding
    except Exception as e:
        print(f"[!] Error: {e}")
        return None


def test_multimodal_fingerprint(image_path: str):
    print(f"\n{'='*60}")
    print("TEST 3: Multimodal Fingerprinting")
    print(f"{'='*60}")
    
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        print(f"[*] Extracting image-only fingerprint...")
        fp_image = extract_fingerprint(image_bytes, use_multimodal=False)
        print(f"    Dimension: {len(fp_image)}")
        print(f"    Norm: {sum(fp_image**2)**0.5:.4f}")
        
        print(f"\n[*] Extracting multimodal fingerprint...")
        fp_multimodal = extract_fingerprint(image_bytes, use_multimodal=True)
        print(f"    Dimension: {len(fp_multimodal)}")
        print(f"    Norm: {sum(fp_multimodal**2)**0.5:.4f}")
        
        print(f"\n[✓] Comparison:")
        print(f"    Image-only: {len(fp_image)}-dim")
        print(f"    Multimodal: {len(fp_multimodal)}-dim")
        print(f"    Difference: {len(fp_multimodal) - len(fp_image)} dimensions added")
        
        return fp_image, fp_multimodal
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MULTIMODAL FEATURES TEST SUITE")
    print("="*60)
    
    # Find a test image
    test_images = [
        "data/processed/images_1024/hf_sanskrit_ocr/train/10.png",
        "data/processed/images_1024/hf_sanskrit_ocr/validation/1.png",
    ]
    
    test_image = None
    for img_path in test_images:
        if Path(img_path).exists():
            test_image = img_path
            break
    
    if not test_image:
        print("[!] No test image found. Please provide an image path.")
        print("    Usage: python test_multimodal.py <image_path>")
        return
    
    print(f"\n[*] Using test image: {test_image}")
    
    # Test 1: Text extraction
    text_features = test_text_extraction(test_image)
    
    # Test 2: Text embedding (if text extracted)
    if text_features and text_features['has_text']:
        test_text_embedding(text_features['text'])
    else:
        # Test with sample Sanskrit text
        sample_text = "अहं ब्रह्मास्मि"  # "I am Brahman" in Sanskrit
        test_text_embedding(sample_text)
    
    # Test 3: Multimodal fingerprinting
    test_multimodal_fingerprint(test_image)
    
    print(f"\n{'='*60}")
    print("TEST SUITE COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Use provided image path
        test_image = sys.argv[1]
        test_multimodal_fingerprint(test_image)
    else:
        main()



