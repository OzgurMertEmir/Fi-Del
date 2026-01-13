#!/bin/bash
# ec2-gpu-training.sh - Launch EC2 Spot GPU instance for ML training
#
# This script:
# 1. Requests a g4dn.xlarge Spot instance
# 2. Configures the instance with ML dependencies
# 3. Runs the training job
# 4. Auto-terminates on completion
#
# Usage:
#   ./ec2-gpu-training.sh --symbol BTCUSDT --days 7
#   ./ec2-gpu-training.sh --symbol BTCUSDT --date-range 2025-01-01:2025-01-07

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="g4dn.xlarge"
SYMBOL="BTCUSDT"
DAYS=7
DATE_RANGE=""
MODEL_TYPE="lightgbm"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            REGION="$2"
            shift 2
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --days)
            DAYS="$2"
            shift 2
            ;;
        --date-range)
            DATE_RANGE="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Load bucket names
if [ -f "${SCRIPT_DIR}/bucket-names.env" ]; then
    source "${SCRIPT_DIR}/bucket-names.env"
else
    echo "ERROR: bucket-names.env not found. Run s3-setup.sh first."
    exit 1
fi

echo "=============================================="
echo "FiDel GPU Training Job"
echo "=============================================="
echo "Region: $REGION"
echo "Instance type: $INSTANCE_TYPE"
echo "Symbol: $SYMBOL"
echo ""

# Check AWS CLI
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Status action
if [ "$ACTION" = "status" ]; then
    echo "Checking training instances..."
    instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=fidel-training" \
                  "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType,LaunchTime]' \
        --output table \
        --region "$REGION")

    if [ -z "$instances" ] || [ "$instances" = "None" ]; then
        echo "No running training instances found."
    else
        echo "$instances"
    fi
    exit 0
fi

# Stop action
if [ "$ACTION" = "stop" ]; then
    echo "Stopping training instances..."
    instance_ids=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=fidel-training" \
                  "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[*].Instances[*].InstanceId' \
        --output text \
        --region "$REGION")

    if [ -z "$instance_ids" ]; then
        echo "No running training instances found."
    else
        aws ec2 terminate-instances --instance-ids $instance_ids --region "$REGION"
        echo "Terminated instances: $instance_ids"
    fi
    exit 0
fi

# Get Deep Learning AMI (has CUDA and ML libraries pre-installed)
echo "Finding Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Amazon Linux 2*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text \
    --region "$REGION")

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo "Deep Learning AMI not found, falling back to standard AMI..."
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
                  "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text \
        --region "$REGION")
fi

echo "Using AMI: $AMI_ID"

# Get security group
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=fidel-ingestor-sg" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region "$REGION" 2>/dev/null || echo "")

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "fidel-training-sg" \
        --description "Security group for FiDel training" \
        --region "$REGION" \
        --query 'GroupId' \
        --output text)

    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "$REGION"
fi

echo "Security Group: $SG_ID"

# Build training args
TRAINING_ARGS="--symbol $SYMBOL --model-type $MODEL_TYPE"
if [ -n "$DATE_RANGE" ]; then
    TRAINING_ARGS="$TRAINING_ARGS --date-range $DATE_RANGE"
else
    TRAINING_ARGS="$TRAINING_ARGS --days $DAYS"
fi

echo "Training args: $TRAINING_ARGS"
echo ""

# Create user data script
USER_DATA=$(cat << 'USERDATA_START'
#!/bin/bash
set -e

exec > >(tee /var/log/training-setup.log) 2>&1
echo "Starting training setup at $(date)"

# Install dependencies
yum update -y
pip3 install --upgrade pip
pip3 install boto3 polars lightgbm scikit-learn joblib

# Create app directory
mkdir -p /opt/fidel
cd /opt/fidel

# Download training script
cat > training_job.py << 'TRAINING_SCRIPT'
USERDATA_START
)

# Append the training script
USER_DATA="${USER_DATA}
$(cat "${PROJECT_ROOT}/src/deployment/training_job.py")
TRAINING_SCRIPT"

# Continue user data
USER_DATA="${USER_DATA}

# Set environment variables
export FIDEL_PROCESSED_BUCKET='${FIDEL_PROCESSED_BUCKET}'
export FIDEL_MODELS_BUCKET='${FIDEL_MODELS_BUCKET}'
export AWS_REGION='${REGION}'

# Run training
echo 'Starting training job at \$(date)'
python3 /opt/fidel/training_job.py ${TRAINING_ARGS}

echo 'Training complete at \$(date)'
"

# Get current spot price
SPOT_PRICE=$(aws ec2 describe-spot-price-history \
    --instance-types "$INSTANCE_TYPE" \
    --product-descriptions "Linux/UNIX" \
    --query 'SpotPriceHistory[0].SpotPrice' \
    --output text \
    --region "$REGION")

echo "Current spot price for $INSTANCE_TYPE: \$$SPOT_PRICE/hr"
echo ""

# Request Spot instance
echo "Requesting Spot instance..."

LAUNCH_SPEC=$(cat << EOF
{
    "ImageId": "$AMI_ID",
    "InstanceType": "$INSTANCE_TYPE",
    "IamInstanceProfile": {
        "Name": "fidel-ec2-profile"
    },
    "SecurityGroupIds": ["$SG_ID"],
    "UserData": "$(echo "$USER_DATA" | base64 | tr -d '\n')",
    "BlockDeviceMappings": [
        {
            "DeviceName": "/dev/xvda",
            "Ebs": {
                "VolumeSize": 100,
                "VolumeType": "gp3",
                "DeleteOnTermination": true
            }
        }
    ]
}
EOF
)

# Request Spot instance
SPOT_REQUEST=$(aws ec2 request-spot-instances \
    --spot-price "$SPOT_PRICE" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "$LAUNCH_SPEC" \
    --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
    --output text \
    --region "$REGION")

echo "Spot request: $SPOT_REQUEST"

# Wait for spot request to be fulfilled
echo "Waiting for Spot instance..."
aws ec2 wait spot-instance-request-fulfilled \
    --spot-instance-request-ids "$SPOT_REQUEST" \
    --region "$REGION"

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
    --spot-instance-request-ids "$SPOT_REQUEST" \
    --query 'SpotInstanceRequests[0].InstanceId' \
    --output text \
    --region "$REGION")

echo "Instance ID: $INSTANCE_ID"

# Tag the instance
aws ec2 create-tags \
    --resources "$INSTANCE_ID" \
    --tags "Key=Name,Value=fidel-training" \
           "Key=Project,Value=FiDel" \
           "Key=Symbol,Value=$SYMBOL" \
    --region "$REGION"

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Get instance details
INSTANCE_INFO=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].[PublicIpAddress,PrivateIpAddress]' \
    --output text \
    --region "$REGION")

PUBLIC_IP=$(echo "$INSTANCE_INFO" | cut -f1)

echo ""
echo "=============================================="
echo "Training Instance Started!"
echo "=============================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Instance Type: $INSTANCE_TYPE"
echo "Public IP: $PUBLIC_IP"
echo "Spot Price: \$$SPOT_PRICE/hr"
echo ""
echo "The training job will start automatically after setup (~5-10 min)."
echo "Instance will auto-terminate on completion."
echo ""
echo "To monitor progress:"
echo "  ssh ec2-user@$PUBLIC_IP 'tail -f /var/log/training-setup.log'"
echo ""
echo "To check training status:"
echo "  ./ec2-gpu-training.sh --status"
echo ""
echo "To manually stop:"
echo "  ./ec2-gpu-training.sh --stop"
echo ""
echo "Models will be saved to:"
echo "  s3://${FIDEL_MODELS_BUCKET}/checkpoints/${SYMBOL}/"
