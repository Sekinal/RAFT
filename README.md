Here's a thorough README README on how to run the RAFT and ResNet optical flow feature extraction pipeline for EgoEgo, assuming no docker-compose and using your Docker setup.

***

# README: RAFT + ResNet Optical Flow Feature Extraction for EgoEgo

This guide explains how to run the RAFT optical flow feature extraction and ResNet-based feature embedding steps for preparation of EgoEgo full-body pose estimation from egocentric videos.

***

## Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 3090)
- Docker installed with NVIDIA Container Toolkit for GPU support
- RAFT repo cloned and Docker image `raft-pixi:latest` built as per previous instructions
- Your egocentric video frames (original RGB images)
- SLAM poses corresponding to the video frames (for EgoEgo, not covered here)

***

## Step 1: Run RAFT Optical Flow Estimation with Docker (Batched)

This step computes optical flow images between consecutive video frames.

```bash
sudo docker run --gpus all --rm \
  -v $(pwd):/workspace/raft \
  -w /workspace/raft \
  raft-pixi:latest \
  pixi run python demo.py \
    --model models/raft-things.pth \
    --path output_frames \
    --output_dir flow_results_only \
    --batch_size 8 \
    --max_dim 512
```

- `--path output_frames` points to your original RGB frames directory
- `--output_dir flow_results_only` is where RAFT will save **only the optical flow visualization images** (no original frames concatenated)
- `--batch_size 8` speeds processing but you can tune based on your VRAM
- `--max_dim 512` resizes large frames to avoid out-of-memory errors

***

## Step 2: Download Pre-trained ResNet Weights to Cache (Host Machine)

To extract features with ResNet, download pretrained weights to avoid Docker SSL errors:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints/
wget -O ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
```

This prevents errors when running the Docker container.

***

## Step 3: Extract ResNet-18 Features from Optical Flow Images (Batched)

Use this step to extract 512-dimensional feature vectors from the RAFT optical flow images.

Create `extract_resnet_features.py` using the provided script.

Run it inside Docker:

```bash
sudo docker run --gpus all --rm \
  -v $(pwd):/workspace/raft \
  -v ~/.cache/torch:/root/.cache/torch \
  -w /workspace/raft \
  raft-pixi:latest \
  pixi run python extract_resnet_features.py \
    --image_dir flow_results_only \
    --output_dir raft_of_feats \
    --batch_size 64
```

- `--image_dir flow_results_only` points to the RAFT optical flow images from Step 1
- `--output_dir raft_of_feats` is where 512-D `.npy` feature vectors will be saved matching EgoEgo format
- Mount your local torch cache `~/.cache/torch` to avoid re-downloading weights

***

## Step 4: Validate Outputs

- You should have optical flow visualizations in `flow_results_only/` (only flow images)
- You should have extracted ResNet features in `raft_of_feats/` as `.npy` files:
  - Named `00000.npy`, `00001.npy`, ...
  - Each file is a numpy array of shape `(512,)`

***

## Additional Tips

- Tune `--batch_size` in both steps to fully utilize your GPU memory but avoid OOM errors.
- Use `--max_dim` in RAFT to resize large images.
- Make sure the SLAM poses you have are formatted correctly for EgoEgo inputs.
- Download and place the required pretrained EgoEgo models as per their instructions.
- To generate body pose estimations, follow the EgoEgo repo pipeline using these features and your SLAM data.

***

## Summary Pipeline

```
Original Frames → RAFT (optical flow images) → ResNet-18 (512-D features) → EgoEgo pose prediction
```

***

This workflow ensures you get the exact feature format EgoEgo expects from your custom egocentric video.

If you want, assistance on the next EgoEgo steps is available!

***

Thank you for working through this end-to-end! Enjoy your full-body egocentric pose estimation!

[1](https://github.com/DominicBreuker/resnet_50_docker)
[2](https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18)
[3](https://v-iashin.github.io/video_features/models/resnet/)
[4](https://github.com/v-iashin/video_features)
[5](https://discuss.pytorch.org/t/how-should-i-extract-feature-from-resnet-50-pytorch-pre-trained-model-for-deep-learning-coloring/69558)
[6](https://v-iashin.github.io/video_features/models/raft/)
[7](https://hub.docker.com/r/challisa/easyocr)