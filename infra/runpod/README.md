# RunPod Training Guide

Train FiDel ML models on RunPod GPU instances with data from AWS S3.

## Quick Start

### 1. Create RunPod Account
1. Go to [runpod.io](https://runpod.io)
2. Create account and add credits ($10-20 is enough to start)

### 2. Launch GPU Pod
1. Click "Deploy" → "GPU Pods"
2. Select a GPU:
   - **Budget**: RTX 3090 (~$0.22/hr)
   - **Performance**: RTX 4090 (~$0.40/hr)
   - **Memory**: A100 40GB (~$1.00/hr) for large models
3. Select template: **RunPod Pytorch 2.0** (or similar)
4. Set volume: 20GB is enough
5. Click "Deploy"

### 3. Connect to Pod
1. Click "Connect" → "Start Web Terminal" or use SSH

### 4. Set Up Environment

```bash
# Install dependencies
pip install boto3 pandas pyarrow torch scikit-learn xgboost lightgbm

# Set AWS credentials (get from AWS Console → IAM → Security Credentials)
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Upload or clone the training script
# Option A: Clone repo
git clone https://github.com/YOUR_USER/FiDel.git
cd FiDel

# Option B: Upload runpod_train.py manually via File Manager
```

### 5. Run Training

```bash
# XGBoost (fast, ~5 min)
python scripts/runpod_train.py --model xgboost --days 7

# LSTM (GPU accelerated, ~15-30 min)
python scripts/runpod_train.py --model lstm --days 14 --epochs 100

# Transformer (GPU accelerated, ~30-60 min)
python scripts/runpod_train.py --model transformer --days 14 --epochs 100 --hidden-size 128
```

### 6. Results
- Models are automatically uploaded to S3: `s3://fidel-models-xxx/models/`
- Training logs show accuracy/loss metrics
- **Stop the pod when done** to avoid charges!

## Command Reference

```bash
python scripts/runpod_train.py \
    --model transformer \      # xgboost, lightgbm, lstm, gru, transformer
    --symbols BTCUSDT \        # Trading symbols
    --days 14 \                # Days of data (or use --start-date/--end-date)
    --epochs 100 \             # Training epochs (neural nets only)
    --batch-size 256 \         # Batch size
    --sequence-length 60 \     # Sequence length for LSTM/Transformer
    --hidden-size 128 \        # Hidden layer size
    --num-layers 2 \           # Number of layers
    --learning-rate 0.001 \    # Learning rate
    --label-type direction     # direction or regression
```

## Cost Estimates

| GPU | Price/hr | 7-day XGBoost | 14-day Transformer |
|-----|----------|---------------|-------------------|
| RTX 3090 | $0.22 | ~$0.02 (5 min) | ~$0.15 (40 min) |
| RTX 4090 | $0.40 | ~$0.03 (5 min) | ~$0.25 (35 min) |
| A100 | $1.00 | ~$0.08 (5 min) | ~$0.50 (30 min) |

## Troubleshooting

### "No data files found"
- Check AWS credentials are correct
- Verify data exists in S3: `aws s3 ls s3://fidel-ml-ready-xxx/features/`
- Ensure the Lambda has processed raw data into parquet

### "CUDA out of memory"
- Reduce batch size: `--batch-size 128`
- Reduce hidden size: `--hidden-size 64`
- Use a GPU with more memory (A100)

### "AWS credentials not found"
```bash
# Verify credentials are set
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Test connection
pip install boto3
python -c "import boto3; print(boto3.client('sts').get_caller_identity())"
```

## Security Notes

- Never commit AWS credentials to git
- Use IAM user with minimal permissions (S3 read/write only)
- Delete the pod when done to avoid charges
- Consider using RunPod secrets for credentials
