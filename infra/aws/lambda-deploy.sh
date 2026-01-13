#!/bin/bash
# lambda-deploy.sh - Deploy Lambda function for feature engineering
#
# This script:
# 1. Packages the Lambda function with dependencies
# 2. Creates/updates the Lambda function
# 3. Configures S3 trigger
#
# Prerequisites:
# - Docker (for building Lambda layer)
# - AWS CLI configured
#
# Usage:
#   ./lambda-deploy.sh [--region us-east-1]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REGION="${AWS_REGION:-us-east-1}"
FUNCTION_NAME="fidel-feature-processor"
RUNTIME="python3.11"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            REGION="$2"
            shift 2
            ;;
        --update-code)
            UPDATE_ONLY=true
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
echo "FiDel Lambda Deployment"
echo "=============================================="
echo "Region: $REGION"
echo "Function: $FUNCTION_NAME"
echo ""

# Check AWS CLI
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create build directory
BUILD_DIR=$(mktemp -d)
echo "Build directory: $BUILD_DIR"

# ============================================
# Create Lambda Layer with dependencies
# ============================================
create_layer() {
    echo ""
    echo "Creating Lambda layer with dependencies..."

    LAYER_DIR="${BUILD_DIR}/layer"
    mkdir -p "${LAYER_DIR}/python"

    # Check if Docker is available
    if command -v docker &>/dev/null; then
        echo "Using Docker for compatible builds..."

        # Create requirements file
        cat > "${BUILD_DIR}/requirements.txt" << EOF
polars==0.20.3
numpy==1.26.3
EOF

        # Build in Docker with Lambda-compatible environment
        docker run --rm \
            -v "${BUILD_DIR}:/build" \
            -w /build \
            public.ecr.aws/lambda/python:3.11 \
            pip install -r requirements.txt -t /build/layer/python --no-cache-dir

    else
        echo "Docker not available, using local pip (may have compatibility issues)..."
        pip install polars numpy -t "${LAYER_DIR}/python" --no-cache-dir
    fi

    # Create layer zip
    cd "${LAYER_DIR}"
    zip -r9 "${BUILD_DIR}/layer.zip" python/

    # Check if layer exists
    LAYER_NAME="fidel-dependencies"
    LAYER_ARN=$(aws lambda list-layer-versions \
        --layer-name "$LAYER_NAME" \
        --query 'LayerVersions[0].LayerVersionArn' \
        --output text \
        --region "$REGION" 2>/dev/null || echo "None")

    # Publish new layer version
    echo "Publishing Lambda layer..."
    LAYER_VERSION_ARN=$(aws lambda publish-layer-version \
        --layer-name "$LAYER_NAME" \
        --description "Dependencies for FiDel Lambda functions" \
        --zip-file "fileb://${BUILD_DIR}/layer.zip" \
        --compatible-runtimes python3.11 \
        --query 'LayerVersionArn' \
        --output text \
        --region "$REGION")

    echo "Layer published: $LAYER_VERSION_ARN"
    echo "$LAYER_VERSION_ARN" > "${BUILD_DIR}/layer_arn.txt"
}

# ============================================
# Package Lambda function
# ============================================
package_function() {
    echo ""
    echo "Packaging Lambda function..."

    FUNC_DIR="${BUILD_DIR}/function"
    mkdir -p "$FUNC_DIR"

    # Copy Lambda handler
    cp "${PROJECT_ROOT}/src/deployment/lambda_processor.py" "${FUNC_DIR}/lambda_function.py"

    # Create zip
    cd "$FUNC_DIR"
    zip -r9 "${BUILD_DIR}/function.zip" .

    echo "Function packaged: ${BUILD_DIR}/function.zip"
}

# ============================================
# Deploy Lambda function
# ============================================
deploy_function() {
    echo ""
    echo "Deploying Lambda function..."

    LAYER_ARN=$(cat "${BUILD_DIR}/layer_arn.txt")
    ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/fidel-lambda-role"

    # Check if function exists
    if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" &>/dev/null; then
        echo "Updating existing function..."

        # Update code
        aws lambda update-function-code \
            --function-name "$FUNCTION_NAME" \
            --zip-file "fileb://${BUILD_DIR}/function.zip" \
            --region "$REGION"

        # Wait for update to complete
        aws lambda wait function-updated --function-name "$FUNCTION_NAME" --region "$REGION"

        # Update configuration
        aws lambda update-function-configuration \
            --function-name "$FUNCTION_NAME" \
            --layers "$LAYER_ARN" \
            --environment "Variables={PROCESSED_BUCKET=${FIDEL_PROCESSED_BUCKET}}" \
            --region "$REGION"

    else
        echo "Creating new function..."

        aws lambda create-function \
            --function-name "$FUNCTION_NAME" \
            --runtime "$RUNTIME" \
            --role "$ROLE_ARN" \
            --handler "lambda_function.handler" \
            --zip-file "fileb://${BUILD_DIR}/function.zip" \
            --layers "$LAYER_ARN" \
            --timeout 300 \
            --memory-size 512 \
            --environment "Variables={PROCESSED_BUCKET=${FIDEL_PROCESSED_BUCKET}}" \
            --description "FiDel feature engineering processor" \
            --region "$REGION"
    fi

    echo "Function deployed: $FUNCTION_NAME"
}

# ============================================
# Configure S3 trigger
# ============================================
configure_trigger() {
    echo ""
    echo "Configuring S3 trigger..."

    FUNCTION_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"

    # Add permission for S3 to invoke Lambda
    aws lambda add-permission \
        --function-name "$FUNCTION_NAME" \
        --statement-id "S3InvokeFunction" \
        --action "lambda:InvokeFunction" \
        --principal s3.amazonaws.com \
        --source-arn "arn:aws:s3:::${FIDEL_RAW_BUCKET}" \
        --source-account "$ACCOUNT_ID" \
        --region "$REGION" 2>/dev/null || true

    # Create S3 notification configuration
    cat > "${BUILD_DIR}/notification.json" << EOF
{
    "LambdaFunctionConfigurations": [
        {
            "Id": "FiDelFeatureProcessing",
            "LambdaFunctionArn": "${FUNCTION_ARN}",
            "Events": ["s3:ObjectCreated:*"],
            "Filter": {
                "Key": {
                    "FilterRules": [
                        {
                            "Name": "prefix",
                            "Value": "trades/"
                        },
                        {
                            "Name": "suffix",
                            "Value": ".json.gz"
                        }
                    ]
                }
            }
        }
    ]
}
EOF

    # Apply notification configuration
    aws s3api put-bucket-notification-configuration \
        --bucket "$FIDEL_RAW_BUCKET" \
        --notification-configuration "file://${BUILD_DIR}/notification.json" \
        --region "$REGION"

    echo "S3 trigger configured for s3://${FIDEL_RAW_BUCKET}/trades/*.json.gz"
}

# ============================================
# Main execution
# ============================================

if [ "$UPDATE_ONLY" = true ]; then
    package_function
    # Just update code, skip layer
    echo "Updating function code only..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://${BUILD_DIR}/function.zip" \
        --region "$REGION"
else
    create_layer
    package_function
    deploy_function
    configure_trigger
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Lambda Deployment Complete!"
echo "=============================================="
echo ""
echo "Function: $FUNCTION_NAME"
echo "Runtime: $RUNTIME"
echo "Memory: 512 MB"
echo "Timeout: 300 seconds"
echo ""
echo "Trigger: S3 ObjectCreated on s3://${FIDEL_RAW_BUCKET}/trades/*.json.gz"
echo "Output: s3://${FIDEL_PROCESSED_BUCKET}/features/"
echo ""
echo "To test manually:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"test\": true}' output.json"
echo ""
echo "To view logs:"
echo "  aws logs tail /aws/lambda/$FUNCTION_NAME --follow"
echo ""
echo "Next step: Run ec2-gpu-training.sh to set up ML training"

# Cleanup
rm -rf "$BUILD_DIR"
