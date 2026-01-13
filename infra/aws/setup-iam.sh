#!/bin/bash
# setup-iam.sh - Create IAM roles and policies for FiDel pipeline
#
# This script creates:
# - fidel-lambda-role: For Lambda functions (feature engineering)
# - fidel-ec2-role: For EC2 instances (ingestor, GPU training)
# - fidel-instance-profile: To attach EC2 role to instances
#
# Prerequisites:
# - AWS CLI configured with admin credentials
# - Run: aws configure
#
# Usage:
#   ./setup-iam.sh [--region us-east-1]

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
echo "FiDel IAM Setup"
echo "=============================================="
echo "Region: $REGION"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    echo ""
    echo "You'll need:"
    echo "  - AWS Access Key ID"
    echo "  - AWS Secret Access Key"
    echo "  - Default region (e.g., us-east-1)"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: $ACCOUNT_ID"
echo ""

# ============================================
# Lambda Role
# ============================================
echo "Creating Lambda execution role..."

# Create trust policy file
cat > /tmp/lambda-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role (ignore if exists)
if aws iam get-role --role-name fidel-lambda-role &>/dev/null; then
    echo "  Role fidel-lambda-role already exists, skipping creation"
else
    aws iam create-role \
        --role-name fidel-lambda-role \
        --assume-role-policy-document file:///tmp/lambda-trust-policy.json \
        --description "Lambda execution role for FiDel feature engineering" \
        --region "$REGION"
    echo "  Created role: fidel-lambda-role"
fi

# Create and attach Lambda policy
cat > /tmp/lambda-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ReadRawData",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::fidel-raw-data-${ACCOUNT_ID}",
        "arn:aws:s3:::fidel-raw-data-${ACCOUNT_ID}/*"
      ]
    },
    {
      "Sid": "S3WriteProcessedData",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::fidel-processed-data-${ACCOUNT_ID}",
        "arn:aws:s3:::fidel-processed-data-${ACCOUNT_ID}/*"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:${ACCOUNT_ID}:log-group:/aws/lambda/fidel-*"
    }
  ]
}
EOF

if aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelLambdaPolicy" &>/dev/null; then
    echo "  Policy FiDelLambdaPolicy already exists, updating..."
    # Get current version and delete old versions if needed
    VERSIONS=$(aws iam list-policy-versions --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelLambdaPolicy" --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text)
    for v in $VERSIONS; do
        aws iam delete-policy-version --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelLambdaPolicy" --version-id "$v" 2>/dev/null || true
    done
    aws iam create-policy-version \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelLambdaPolicy" \
        --policy-document file:///tmp/lambda-policy.json \
        --set-as-default
else
    aws iam create-policy \
        --policy-name FiDelLambdaPolicy \
        --policy-document file:///tmp/lambda-policy.json \
        --description "S3 and CloudWatch access for FiDel Lambda"
    echo "  Created policy: FiDelLambdaPolicy"
fi

# Attach policy to role
aws iam attach-role-policy \
    --role-name fidel-lambda-role \
    --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelLambdaPolicy" 2>/dev/null || true
echo "  Attached FiDelLambdaPolicy to fidel-lambda-role"

# Attach basic Lambda execution policy
aws iam attach-role-policy \
    --role-name fidel-lambda-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
echo "  Attached AWSLambdaBasicExecutionRole"

echo ""

# ============================================
# EC2 Role
# ============================================
echo "Creating EC2 instance role..."

# Create trust policy for EC2
cat > /tmp/ec2-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
if aws iam get-role --role-name fidel-ec2-role &>/dev/null; then
    echo "  Role fidel-ec2-role already exists, skipping creation"
else
    aws iam create-role \
        --role-name fidel-ec2-role \
        --assume-role-policy-document file:///tmp/ec2-trust-policy.json \
        --description "EC2 instance role for FiDel ingestor and training" \
        --region "$REGION"
    echo "  Created role: fidel-ec2-role"
fi

# Create and attach EC2 policy
cat > /tmp/ec2-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3FullAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::fidel-raw-data-${ACCOUNT_ID}",
        "arn:aws:s3:::fidel-raw-data-${ACCOUNT_ID}/*",
        "arn:aws:s3:::fidel-processed-data-${ACCOUNT_ID}",
        "arn:aws:s3:::fidel-processed-data-${ACCOUNT_ID}/*",
        "arn:aws:s3:::fidel-models-${ACCOUNT_ID}",
        "arn:aws:s3:::fidel-models-${ACCOUNT_ID}/*",
        "arn:aws:s3:::fidel-logs-${ACCOUNT_ID}",
        "arn:aws:s3:::fidel-logs-${ACCOUNT_ID}/*"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:*:${ACCOUNT_ID}:log-group:/fidel/*"
    },
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*"
    },
    {
      "Sid": "EC2Describe",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeSpotInstanceRequests",
        "ec2:DescribeInstances",
        "ec2:DescribeTags"
      ],
      "Resource": "*"
    }
  ]
}
EOF

if aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelEC2Policy" &>/dev/null; then
    echo "  Policy FiDelEC2Policy already exists, updating..."
    VERSIONS=$(aws iam list-policy-versions --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelEC2Policy" --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text)
    for v in $VERSIONS; do
        aws iam delete-policy-version --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelEC2Policy" --version-id "$v" 2>/dev/null || true
    done
    aws iam create-policy-version \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelEC2Policy" \
        --policy-document file:///tmp/ec2-policy.json \
        --set-as-default
else
    aws iam create-policy \
        --policy-name FiDelEC2Policy \
        --policy-document file:///tmp/ec2-policy.json \
        --description "S3 and CloudWatch access for FiDel EC2 instances"
    echo "  Created policy: FiDelEC2Policy"
fi

# Attach policy to role
aws iam attach-role-policy \
    --role-name fidel-ec2-role \
    --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/FiDelEC2Policy" 2>/dev/null || true
echo "  Attached FiDelEC2Policy to fidel-ec2-role"

# ============================================
# EC2 Instance Profile
# ============================================
echo ""
echo "Creating EC2 instance profile..."

if aws iam get-instance-profile --instance-profile-name fidel-ec2-profile &>/dev/null; then
    echo "  Instance profile fidel-ec2-profile already exists"
else
    aws iam create-instance-profile --instance-profile-name fidel-ec2-profile
    echo "  Created instance profile: fidel-ec2-profile"
fi

# Add role to instance profile (ignore if already added)
aws iam add-role-to-instance-profile \
    --instance-profile-name fidel-ec2-profile \
    --role-name fidel-ec2-role 2>/dev/null || true
echo "  Attached fidel-ec2-role to fidel-ec2-profile"

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "IAM Setup Complete!"
echo "=============================================="
echo ""
echo "Created resources:"
echo "  - Role: fidel-lambda-role (for Lambda functions)"
echo "  - Role: fidel-ec2-role (for EC2 instances)"
echo "  - Instance Profile: fidel-ec2-profile"
echo "  - Policy: FiDelLambdaPolicy"
echo "  - Policy: FiDelEC2Policy"
echo ""
echo "Lambda Role ARN: arn:aws:iam::${ACCOUNT_ID}:role/fidel-lambda-role"
echo "EC2 Role ARN: arn:aws:iam::${ACCOUNT_ID}:role/fidel-ec2-role"
echo ""
echo "Next step: Run s3-setup.sh to create S3 buckets"

# Cleanup
rm -f /tmp/lambda-trust-policy.json /tmp/lambda-policy.json
rm -f /tmp/ec2-trust-policy.json /tmp/ec2-policy.json
