import os
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_resnet_features_batched(image_dir, output_dir, batch_size=32):
    """
    Extract 512-D ResNet-18 features from optical flow images.
    Matches EgoEgo's expected format: (512,) vectors saved as .npy files.
    """
    print(f"Using device: {DEVICE}")
    
    # Load pre-trained ResNet-18 (outputs 512-D features)
    model = models.resnet18(pretrained=True)
    # Remove final classification layer to get features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(DEVICE)
    model.eval()
    
    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # Get all flow images
    images = sorted(glob.glob(os.path.join(image_dir, '*.png')) + 
                   glob.glob(os.path.join(image_dir, '*.jpg')))
    
    if len(images) == 0:
        print(f"ERROR: No images found in {image_dir}")
        return
    
    print(f"Found {len(images)} optical flow images")
    print(f"Extracting ResNet-18 features with batch_size={batch_size}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            batch_paths = images[batch_start:batch_end]
            
            # Load and preprocess batch
            batch_tensors = []
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = preprocess(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if len(batch_tensors) == 0:
                continue
            
            # Stack into batch and ensure contiguous
            batch = torch.stack(batch_tensors).to(DEVICE).contiguous()
            
            # Extract features for entire batch
            features = model(batch).squeeze()  # Shape: (batch_size, 512)
            
            # Handle single image case
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            
            # Save individual features with EgoEgo naming: 00000.npy, 00001.npy, etc.
            for i, idx in enumerate(range(batch_start, batch_start + len(batch_tensors))):
                output_path = os.path.join(output_dir, f"{idx:05d}.npy")
                feature_vec = features[i].cpu().numpy()
                np.save(output_path, feature_vec)
            
            # Progress update
            if (batch_end) % 100 == 0 or batch_end == len(images):
                print(f"  Processed {batch_end}/{len(images)} images")
            
            # Cleanup
            del batch, features
            torch.cuda.empty_cache()
    
    print(f"\n✓ Done! Extracted features for {len(images)} images")
    print(f"✓ Features saved to: {output_dir}/")
    print(f"✓ Feature shape: (512,) per image")
    print(f"✓ Format: 00000.npy, 00001.npy, ..., {len(images)-1:05d}.npy")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract ResNet-18 features from optical flow images for EgoEgo'
    )
    parser.add_argument('--image_dir', required=True, 
                       help='Directory with optical flow images (from RAFT)')
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save .npy feature files')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for feature extraction (default: 64)')
    args = parser.parse_args()
    
    extract_resnet_features_batched(args.image_dir, args.output_dir, args.batch_size)
