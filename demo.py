import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = 'cuda'

def load_image(imfile, max_dim=1024):
    img = Image.open(imfile)
    orig_width, orig_height = img.size
    
    if max(orig_width, orig_height) > max_dim:
        if orig_width > orig_height:
            new_width = max_dim
            new_height = int((max_dim / orig_width) * orig_height)
        else:
            new_height = max_dim
            new_width = int((max_dim / orig_height) * orig_width)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_flow_viz(img, flo, output_path):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(output_path, flo[:, :, [2,1,0]])


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    images = sorted(glob.glob(os.path.join(args.path, '*.png')) + 
                   glob.glob(os.path.join(args.path, '*.jpg')))
    
    print(f"Found {len(images)} images, processing {len(images)-1} pairs with batch_size={args.batch_size}")
    
    with torch.no_grad():
        pairs = list(zip(images[:-1], images[1:]))
        
        for batch_start in range(0, len(pairs), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(pairs))
            batch_pairs = pairs[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//args.batch_size + 1}: pairs {batch_start}-{batch_end-1}")
            
            # Load all images in batch
            images1 = []
            images2 = []
            for imfile1, imfile2 in batch_pairs:
                images1.append(load_image(imfile1, max_dim=args.max_dim))
                images2.append(load_image(imfile2, max_dim=args.max_dim))
            
            # Stack into batch
            batch_img1 = torch.cat(images1, dim=0)
            batch_img2 = torch.cat(images2, dim=0)
            
            # Pad batch
            padder = InputPadder(batch_img1.shape)
            batch_img1, batch_img2 = padder.pad(batch_img1, batch_img2)
            
            # Process entire batch at once
            flow_low, flow_up = model(batch_img1, batch_img2, iters=20, test_mode=True)
            
            # Save results
            for i, (imfile1, _) in enumerate(batch_pairs):
                idx = batch_start + i
                output_name = f"flow_{idx:04d}_{os.path.basename(imfile1).split('.')[0]}.png"
                output_path = os.path.join(args.output_dir, output_name)
                save_flow_viz(batch_img1[i:i+1], flow_up[i:i+1], output_path)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Saved {idx + 1}/{len(pairs)} pairs")
            
            # Cleanup
            del batch_img1, batch_img2, flow_low, flow_up
            torch.cuda.empty_cache()
    
    print(f"\nAll done! Flow visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation')
    parser.add_argument('--output_dir', default='flow_output', help='output directory')
    parser.add_argument('--max_dim', type=int, default=1024, help='maximum dimension')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for processing')
    args = parser.parse_args()

    demo(args)
