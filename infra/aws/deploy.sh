#!/bin/bash
# deploy.sh - Master deployment script for FiDel AWS infrastructure
#
# This script runs all setup scripts in order to deploy the complete
# FiDel trading data pipeline to AWS.
#
# Prerequisites:
# - AWS CLI installed and configured (aws configure)
# - Docker installed (for Lambda layer build)
# - Sufficient IAM permissions
#
# Usage:
#   ./deploy.sh --email your-email@example.com [--region us-east-1]
#   ./deploy.sh --step 3  # Start from step 3
#   ./deploy.sh --dry-run # Show what would be done

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REGION="${AWS_REGION:-us-east-1}"
EMAIL=""
START_STEP=1
DRY_RUN=false
SKIP_INGESTOR=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            REGION="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --step)
            START_STEP="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-ingestor)
            SKIP_INGESTOR=true
            shift
            ;;
        -h|--help)
            cat << EOF
FiDel AWS Deployment Script

Usage: ./deploy.sh [OPTIONS]

Options:
  --region REGION     AWS region (default: us-east-1)
  --email EMAIL       Email for CloudWatch alerts (recommended)
  --step N            Start from step N (1-6)
  --skip-ingestor     Skip starting the ingestor (Step 5)
  --dry-run           Show what would be done without executing
  -h, --help          Show this help message

Steps:
  1. IAM Setup        - Create roles and policies
  2. S3 Setup         - Create buckets with lifecycle policies
  3. Lambda Deploy    - Deploy feature processing function
  4. CloudWatch       - Set up monitoring and alerts
  5. Start Ingestor   - Launch EC2 Spot ingestor (optional)
  6. Summary          - Show deployment summary

Example:
  ./deploy.sh --email alerts@example.com --region us-west-2
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo ""
    echo -e "${BLUE}=============================================="
    echo "Step $1: $2"
    echo -e "==============================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================
# Pre-flight checks
# ============================================
echo ""
echo -e "${BLUE}=============================================="
echo "FiDel AWS Deployment"
echo -e "==============================================${NC}"
echo ""
echo "Region: $REGION"
echo "Email: ${EMAIL:-'(not set - alerts will not be sent)'}"
echo "Start from step: $START_STEP"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No changes will be made"
    echo ""
fi

# Check AWS CLI
echo "Checking prerequisites..."

if ! command -v aws &>/dev/null; then
    print_error "AWS CLI not installed. Install with: brew install awscli"
    exit 1
fi
print_success "AWS CLI installed"

if ! aws sts get-caller-identity &>/dev/null; then
    print_error "AWS CLI not configured. Run: aws configure"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_USER=$(aws sts get-caller-identity --query Arn --output text)
print_success "AWS authenticated as: $AWS_USER"

# Check Docker (optional for Lambda)
if command -v docker &>/dev/null; then
    print_success "Docker installed (will use for Lambda layer)"
else
    print_warning "Docker not installed (Lambda layer may have compatibility issues)"
fi

echo ""
read -p "Continue with deployment? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Make all scripts executable
chmod +x "${SCRIPT_DIR}"/*.sh

# ============================================
# Step 1: IAM Setup
# ============================================
if [ $START_STEP -le 1 ]; then
    print_step 1 "IAM Setup"

    if [ "$DRY_RUN" = true ]; then
        echo "Would run: setup-iam.sh --region $REGION"
    else
        "${SCRIPT_DIR}/setup-iam.sh" --region "$REGION"
    fi
fi

# ============================================
# Step 2: S3 Setup
# ============================================
if [ $START_STEP -le 2 ]; then
    print_step 2 "S3 Storage Setup"

    if [ "$DRY_RUN" = true ]; then
        echo "Would run: s3-setup.sh --region $REGION"
    else
        "${SCRIPT_DIR}/s3-setup.sh" --region "$REGION"
    fi
fi

# ============================================
# Step 3: Lambda Deploy
# ============================================
if [ $START_STEP -le 3 ]; then
    print_step 3 "Lambda Function Deployment"

    if [ "$DRY_RUN" = true ]; then
        echo "Would run: lambda-deploy.sh --region $REGION"
    else
        "${SCRIPT_DIR}/lambda-deploy.sh" --region "$REGION"
    fi
fi

# ============================================
# Step 4: CloudWatch Setup
# ============================================
if [ $START_STEP -le 4 ]; then
    print_step 4 "CloudWatch Monitoring Setup"

    if [ "$DRY_RUN" = true ]; then
        echo "Would run: cloudwatch-setup.sh --region $REGION --email $EMAIL"
    else
        if [ -n "$EMAIL" ]; then
            "${SCRIPT_DIR}/cloudwatch-setup.sh" --region "$REGION" --email "$EMAIL"
        else
            "${SCRIPT_DIR}/cloudwatch-setup.sh" --region "$REGION"
        fi
    fi
fi

# ============================================
# Step 5: Start Ingestor (Optional)
# ============================================
if [ $START_STEP -le 5 ] && [ "$SKIP_INGESTOR" = false ]; then
    print_step 5 "Data Ingestor (Optional)"

    echo "Do you want to start the data ingestor now?"
    echo "This will launch an EC2 Spot instance (~\$3/month) to stream Binance data."
    echo ""
    read -p "Start ingestor? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would run: ec2-ingestor.sh --region $REGION --symbol BTCUSDT"
        else
            "${SCRIPT_DIR}/ec2-ingestor.sh" --region "$REGION" --symbol BTCUSDT
        fi
    else
        echo "Skipping ingestor. You can start it later with:"
        echo "  ./ec2-ingestor.sh --symbol BTCUSDT"
    fi
fi

# ============================================
# Step 6: Summary
# ============================================
print_step 6 "Deployment Summary"

# Reload bucket names
if [ -f "${SCRIPT_DIR}/bucket-names.env" ]; then
    source "${SCRIPT_DIR}/bucket-names.env"
fi

cat << EOF
${GREEN}=============================================="
Deployment Complete!
==============================================${NC}

${BLUE}AWS Account:${NC} $ACCOUNT_ID
${BLUE}Region:${NC} $REGION

${BLUE}S3 Buckets:${NC}
  - Raw data:      s3://${FIDEL_RAW_BUCKET:-fidel-raw-data-$ACCOUNT_ID}
  - Processed:     s3://${FIDEL_PROCESSED_BUCKET:-fidel-processed-data-$ACCOUNT_ID}
  - Models:        s3://${FIDEL_MODELS_BUCKET:-fidel-models-$ACCOUNT_ID}
  - Logs:          s3://${FIDEL_LOGS_BUCKET:-fidel-logs-$ACCOUNT_ID}

${BLUE}Lambda Function:${NC}
  - Name:          fidel-feature-processor
  - Trigger:       S3 ObjectCreated on trades/*.json.gz

${BLUE}IAM Roles:${NC}
  - Lambda:        fidel-lambda-role
  - EC2:           fidel-ec2-role

${BLUE}CloudWatch:${NC}
  - Dashboard:     https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=FiDel-Pipeline
  - Log groups:    /fidel/ingestor, /fidel/training, /aws/lambda/fidel-feature-processor

${BLUE}Estimated Monthly Cost:${NC}
  - S3 Storage:        \$3-5
  - EC2 Ingestor:      \$3-5 (if running)
  - Lambda:            \$0 (free tier)
  - CloudWatch:        \$0 (free tier)
  - GPU Training:      \$3-10 (on-demand)
  - Total:             \$10-30/month

${YELLOW}Next Steps:${NC}

1. Start data ingestion (if not already):
   ./ec2-ingestor.sh --symbol BTCUSDT

2. Monitor the pipeline:
   Open CloudWatch dashboard in AWS Console

3. Run training job (after data accumulates):
   ./ec2-gpu-training.sh --symbol BTCUSDT --days 7

4. Check ingestor status:
   ./ec2-ingestor.sh --status

5. Stop ingestor when not needed:
   ./ec2-ingestor.sh --stop

${BLUE}Documentation:${NC}
  See docs/AWS_DEPLOYMENT.md for detailed usage instructions.

EOF

# Create quick reference guide
cat > "${SCRIPT_DIR}/QUICK_REFERENCE.md" << 'QUICKREF'
# FiDel AWS Quick Reference

## Common Commands

### Data Ingestion
```bash
# Start ingestor for a symbol
./ec2-ingestor.sh --symbol BTCUSDT

# Check ingestor status
./ec2-ingestor.sh --status

# Stop ingestor
./ec2-ingestor.sh --stop
```

### ML Training
```bash
# Train on last 7 days
./ec2-gpu-training.sh --symbol BTCUSDT --days 7

# Train on specific date range
./ec2-gpu-training.sh --symbol BTCUSDT --date-range 2025-01-01:2025-01-07

# Check training status
./ec2-gpu-training.sh --status

# Stop training instance
./ec2-gpu-training.sh --stop
```

### Lambda
```bash
# Update Lambda code
./lambda-deploy.sh --update-code

# View Lambda logs
aws logs tail /aws/lambda/fidel-feature-processor --follow
```

### Monitoring
```bash
# View CloudWatch dashboard
open "https://console.aws.amazon.com/cloudwatch/home#dashboards:name=FiDel-Pipeline"

# Check S3 bucket sizes
aws s3 ls s3://fidel-raw-data-YOUR_ACCOUNT_ID --summarize --recursive
```

## Troubleshooting

### Ingestor not receiving data
1. Check security group allows outbound connections
2. Verify IAM role has S3 write permissions
3. Check instance logs: `ssh ec2-user@IP 'journalctl -u fidel-ingestor'`

### Lambda not processing files
1. Check S3 trigger configuration
2. Verify Lambda has read access to raw bucket
3. Check CloudWatch logs for errors

### Training job fails
1. Check instance has enough disk space (100GB)
2. Verify processed data exists in S3
3. Check training logs: `ssh ec2-user@IP 'cat /var/log/training-setup.log'`
QUICKREF

print_success "Quick reference saved to: ${SCRIPT_DIR}/QUICK_REFERENCE.md"
echo ""
