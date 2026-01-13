#!/bin/bash
# cloudwatch-setup.sh - Set up CloudWatch monitoring and alerts
#
# This script:
# 1. Creates CloudWatch log groups
# 2. Sets up metric alarms
# 3. Configures SNS notifications
# 4. Creates a CloudWatch dashboard
#
# Usage:
#   ./cloudwatch-setup.sh --email your-email@example.com [--region us-east-1]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"
EMAIL=""

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
echo "FiDel CloudWatch Setup"
echo "=============================================="
echo "Region: $REGION"
echo ""

# Check AWS CLI
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# ============================================
# Create Log Groups
# ============================================
echo "Creating CloudWatch Log Groups..."

LOG_GROUPS=(
    "/fidel/ingestor"
    "/fidel/training"
    "/aws/lambda/fidel-feature-processor"
)

for log_group in "${LOG_GROUPS[@]}"; do
    if aws logs describe-log-groups --log-group-name-prefix "$log_group" --query 'logGroups[0].logGroupName' --output text --region "$REGION" 2>/dev/null | grep -q "$log_group"; then
        echo "  Log group exists: $log_group"
    else
        aws logs create-log-group --log-group-name "$log_group" --region "$REGION"
        echo "  Created log group: $log_group"
    fi

    # Set retention to 30 days
    aws logs put-retention-policy \
        --log-group-name "$log_group" \
        --retention-in-days 30 \
        --region "$REGION" 2>/dev/null || true
done

echo ""

# ============================================
# Create SNS Topic for Alerts
# ============================================
echo "Creating SNS topic for alerts..."

TOPIC_ARN=$(aws sns create-topic \
    --name "fidel-alerts" \
    --query 'TopicArn' \
    --output text \
    --region "$REGION")

echo "  Topic ARN: $TOPIC_ARN"

# Subscribe email if provided
if [ -n "$EMAIL" ]; then
    aws sns subscribe \
        --topic-arn "$TOPIC_ARN" \
        --protocol email \
        --notification-endpoint "$EMAIL" \
        --region "$REGION"
    echo "  Subscribed: $EMAIL (check your email to confirm)"
fi

echo ""

# ============================================
# Create Metric Alarms
# ============================================
echo "Creating CloudWatch alarms..."

# Lambda error alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "FiDel-Lambda-Errors" \
    --alarm-description "Lambda function errors for FiDel feature processor" \
    --metric-name Errors \
    --namespace AWS/Lambda \
    --statistic Sum \
    --period 300 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --evaluation-periods 1 \
    --dimensions "Name=FunctionName,Value=fidel-feature-processor" \
    --alarm-actions "$TOPIC_ARN" \
    --region "$REGION"
echo "  Created alarm: FiDel-Lambda-Errors"

# Lambda duration alarm (approaching timeout)
aws cloudwatch put-metric-alarm \
    --alarm-name "FiDel-Lambda-Duration" \
    --alarm-description "Lambda function approaching timeout" \
    --metric-name Duration \
    --namespace AWS/Lambda \
    --statistic Average \
    --period 300 \
    --threshold 240000 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --evaluation-periods 2 \
    --dimensions "Name=FunctionName,Value=fidel-feature-processor" \
    --alarm-actions "$TOPIC_ARN" \
    --region "$REGION"
echo "  Created alarm: FiDel-Lambda-Duration"

# S3 bucket size alarm (cost control)
aws cloudwatch put-metric-alarm \
    --alarm-name "FiDel-S3-RawData-Size" \
    --alarm-description "Raw data bucket size exceeds 50GB" \
    --metric-name BucketSizeBytes \
    --namespace AWS/S3 \
    --statistic Average \
    --period 86400 \
    --threshold 53687091200 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --evaluation-periods 1 \
    --dimensions "Name=BucketName,Value=${FIDEL_RAW_BUCKET}" "Name=StorageType,Value=StandardStorage" \
    --alarm-actions "$TOPIC_ARN" \
    --region "$REGION"
echo "  Created alarm: FiDel-S3-RawData-Size"

echo ""

# ============================================
# Create Dashboard
# ============================================
echo "Creating CloudWatch dashboard..."

DASHBOARD_BODY=$(cat << EOF
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "title": "Lambda Invocations & Errors",
                "metrics": [
                    ["AWS/Lambda", "Invocations", "FunctionName", "fidel-feature-processor", {"label": "Invocations"}],
                    [".", "Errors", ".", ".", {"label": "Errors", "color": "#d62728"}]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "${REGION}",
                "period": 300
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "title": "Lambda Duration",
                "metrics": [
                    ["AWS/Lambda", "Duration", "FunctionName", "fidel-feature-processor", {"stat": "Average", "label": "Avg"}],
                    ["...", {"stat": "Maximum", "label": "Max"}]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "${REGION}",
                "period": 300
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "title": "S3 Bucket Sizes",
                "metrics": [
                    ["AWS/S3", "BucketSizeBytes", "BucketName", "${FIDEL_RAW_BUCKET}", "StorageType", "StandardStorage", {"label": "Raw Data"}],
                    ["...", "${FIDEL_PROCESSED_BUCKET}", ".", ".", {"label": "Processed Data"}],
                    ["...", "${FIDEL_MODELS_BUCKET}", ".", ".", {"label": "Models"}]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "${REGION}",
                "period": 86400
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "title": "S3 Object Counts",
                "metrics": [
                    ["AWS/S3", "NumberOfObjects", "BucketName", "${FIDEL_RAW_BUCKET}", "StorageType", "AllStorageTypes", {"label": "Raw Data"}],
                    ["...", "${FIDEL_PROCESSED_BUCKET}", ".", ".", {"label": "Processed Data"}],
                    ["...", "${FIDEL_MODELS_BUCKET}", ".", ".", {"label": "Models"}]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "${REGION}",
                "period": 86400
            }
        },
        {
            "type": "log",
            "x": 0,
            "y": 12,
            "width": 24,
            "height": 6,
            "properties": {
                "title": "Recent Lambda Logs",
                "query": "SOURCE '/aws/lambda/fidel-feature-processor' | fields @timestamp, @message | sort @timestamp desc | limit 50",
                "region": "${REGION}",
                "view": "table"
            }
        },
        {
            "type": "text",
            "x": 0,
            "y": 18,
            "width": 24,
            "height": 2,
            "properties": {
                "markdown": "## FiDel Trading Pipeline Dashboard\\n\\nThis dashboard monitors the FiDel data pipeline including data ingestion, feature processing, and model training."
            }
        }
    ]
}
EOF
)

aws cloudwatch put-dashboard \
    --dashboard-name "FiDel-Pipeline" \
    --dashboard-body "$DASHBOARD_BODY" \
    --region "$REGION"

echo "  Created dashboard: FiDel-Pipeline"

echo ""

# ============================================
# Summary
# ============================================
echo "=============================================="
echo "CloudWatch Setup Complete!"
echo "=============================================="
echo ""
echo "Created resources:"
echo "  - Log groups: ${LOG_GROUPS[*]}"
echo "  - SNS topic: $TOPIC_ARN"
echo "  - Alarms: Lambda errors, duration, S3 size"
echo "  - Dashboard: FiDel-Pipeline"
echo ""
echo "View dashboard:"
echo "  https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=FiDel-Pipeline"
echo ""
if [ -n "$EMAIL" ]; then
    echo "IMPORTANT: Check your email ($EMAIL) to confirm the SNS subscription!"
    echo ""
fi
echo "Next step: Run deploy.sh to complete the full deployment"

# Save SNS topic ARN for other scripts
echo "export FIDEL_SNS_TOPIC=$TOPIC_ARN" >> "${SCRIPT_DIR}/bucket-names.env"
