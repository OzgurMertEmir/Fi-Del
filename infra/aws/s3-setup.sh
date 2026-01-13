#!/bin/bash
# s3-setup.sh - Create S3 buckets with lifecycle policies for FiDel
#
# Creates:
# - fidel-raw-data-{account_id}: Raw streaming data
# - fidel-processed-data-{account_id}: Feature-engineered data
# - fidel-models-{account_id}: Trained model artifacts
# - fidel-logs-{account_id}: Application logs
#
# Usage:
#   ./s3-setup.sh [--region us-east-1]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            REGION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "FiDel S3 Setup"
echo "=============================================="
echo "Region: $REGION"
echo ""

# Check AWS CLI
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: $ACCOUNT_ID"
echo ""

# Define bucket names (globally unique with account ID)
RAW_BUCKET="fidel-raw-data-${ACCOUNT_ID}"
PROCESSED_BUCKET="fidel-processed-data-${ACCOUNT_ID}"
MODELS_BUCKET="fidel-models-${ACCOUNT_ID}"
LOGS_BUCKET="fidel-logs-${ACCOUNT_ID}"

# Function to create bucket
create_bucket() {
    local bucket_name=$1
    local description=$2

    echo "Creating bucket: $bucket_name ($description)"

    if aws s3api head-bucket --bucket "$bucket_name" 2>/dev/null; then
        echo "  Bucket already exists, skipping creation"
    else
        # Create bucket (different syntax for us-east-1)
        if [ "$REGION" = "us-east-1" ]; then
            aws s3api create-bucket \
                --bucket "$bucket_name" \
                --region "$REGION"
        else
            aws s3api create-bucket \
                --bucket "$bucket_name" \
                --region "$REGION" \
                --create-bucket-configuration LocationConstraint="$REGION"
        fi
        echo "  Created bucket: $bucket_name"
    fi

    # Enable versioning for data safety
    aws s3api put-bucket-versioning \
        --bucket "$bucket_name" \
        --versioning-configuration Status=Enabled
    echo "  Enabled versioning"

    # Block public access
    aws s3api put-public-access-block \
        --bucket "$bucket_name" \
        --public-access-block-configuration \
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
    echo "  Blocked public access"
}

# ============================================
# Create Buckets
# ============================================

create_bucket "$RAW_BUCKET" "raw streaming data"
create_bucket "$PROCESSED_BUCKET" "processed/feature-engineered data"
create_bucket "$MODELS_BUCKET" "trained model artifacts"
create_bucket "$LOGS_BUCKET" "application logs"

echo ""

# ============================================
# Lifecycle Policies
# ============================================
echo "Configuring lifecycle policies..."

# Raw data: Standard -> IA (30 days) -> Glacier (90 days) -> Delete (365 days)
cat > /tmp/raw-lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "ID": "TransitionToIA",
      "Status": "Enabled",
      "Filter": {
        "Prefix": ""
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 365
      },
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 30
      }
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket "$RAW_BUCKET" \
    --lifecycle-configuration file:///tmp/raw-lifecycle.json
echo "  Applied lifecycle to $RAW_BUCKET (IA:30d, Glacier:90d, Delete:365d)"

# Processed data: Standard -> IA (60 days) -> Delete (180 days)
cat > /tmp/processed-lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "ID": "TransitionToIA",
      "Status": "Enabled",
      "Filter": {
        "Prefix": ""
      },
      "Transitions": [
        {
          "Days": 60,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "Expiration": {
        "Days": 180
      },
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 14
      }
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket "$PROCESSED_BUCKET" \
    --lifecycle-configuration file:///tmp/processed-lifecycle.json
echo "  Applied lifecycle to $PROCESSED_BUCKET (IA:60d, Delete:180d)"

# Models: Keep indefinitely, cleanup old versions
cat > /tmp/models-lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "ID": "CleanupOldVersions",
      "Status": "Enabled",
      "Filter": {
        "Prefix": ""
      },
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      }
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket "$MODELS_BUCKET" \
    --lifecycle-configuration file:///tmp/models-lifecycle.json
echo "  Applied lifecycle to $MODELS_BUCKET (keep indefinitely, old versions:90d)"

# Logs: Delete after 30 days
cat > /tmp/logs-lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "ID": "DeleteOldLogs",
      "Status": "Enabled",
      "Filter": {
        "Prefix": ""
      },
      "Expiration": {
        "Days": 30
      }
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket "$LOGS_BUCKET" \
    --lifecycle-configuration file:///tmp/logs-lifecycle.json
echo "  Applied lifecycle to $LOGS_BUCKET (Delete:30d)"

echo ""

# ============================================
# Create folder structure
# ============================================
echo "Creating folder structure..."

# Create placeholder folders in raw bucket
aws s3api put-object --bucket "$RAW_BUCKET" --key "trades/" --content-length 0
aws s3api put-object --bucket "$RAW_BUCKET" --key "orderbook/" --content-length 0
aws s3api put-object --bucket "$RAW_BUCKET" --key "signals/" --content-length 0
echo "  Created folders in $RAW_BUCKET: trades/, orderbook/, signals/"

# Create placeholder folders in processed bucket
aws s3api put-object --bucket "$PROCESSED_BUCKET" --key "features/" --content-length 0
aws s3api put-object --bucket "$PROCESSED_BUCKET" --key "datasets/" --content-length 0
echo "  Created folders in $PROCESSED_BUCKET: features/, datasets/"

# Create placeholder folders in models bucket
aws s3api put-object --bucket "$MODELS_BUCKET" --key "checkpoints/" --content-length 0
aws s3api put-object --bucket "$MODELS_BUCKET" --key "production/" --content-length 0
echo "  Created folders in $MODELS_BUCKET: checkpoints/, production/"

echo ""

# ============================================
# Summary
# ============================================
echo "=============================================="
echo "S3 Setup Complete!"
echo "=============================================="
echo ""
echo "Created buckets:"
echo "  - s3://$RAW_BUCKET (raw streaming data)"
echo "  - s3://$PROCESSED_BUCKET (feature-engineered data)"
echo "  - s3://$MODELS_BUCKET (model artifacts)"
echo "  - s3://$LOGS_BUCKET (application logs)"
echo ""
echo "Lifecycle policies:"
echo "  - Raw data: Standard -> IA (30d) -> Glacier (90d) -> Delete (365d)"
echo "  - Processed: Standard -> IA (60d) -> Delete (180d)"
echo "  - Models: Keep indefinitely"
echo "  - Logs: Delete after 30 days"
echo ""
echo "Estimated monthly cost for 100GB: ~\$3-5"
echo ""
echo "Next step: Run ec2-ingestor.sh to set up data ingestion"

# Save bucket names for other scripts
cat > "${SCRIPT_DIR}/bucket-names.env" << EOF
# Auto-generated bucket names for FiDel
export FIDEL_RAW_BUCKET=$RAW_BUCKET
export FIDEL_PROCESSED_BUCKET=$PROCESSED_BUCKET
export FIDEL_MODELS_BUCKET=$MODELS_BUCKET
export FIDEL_LOGS_BUCKET=$LOGS_BUCKET
export FIDEL_ACCOUNT_ID=$ACCOUNT_ID
export FIDEL_REGION=$REGION
EOF
echo "Saved bucket names to: ${SCRIPT_DIR}/bucket-names.env"

# Cleanup
rm -f /tmp/raw-lifecycle.json /tmp/processed-lifecycle.json
rm -f /tmp/models-lifecycle.json /tmp/logs-lifecycle.json
