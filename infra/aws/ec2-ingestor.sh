#!/bin/bash
# ec2-ingestor.sh - Deploy data ingestor to EC2 Spot instance
#
# This script:
# 1. Creates a launch template for the ingestor
# 2. Requests a Spot instance
# 3. Configures the instance with the ingestor service
#
# Usage:
#   ./ec2-ingestor.sh --symbol BTCUSDT [--region us-east-1]
#   ./ec2-ingestor.sh --stop  # Stop running ingestor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="t3.micro"
SYMBOL="BTCUSDT"
ACTION="start"

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
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --status)
            ACTION="status"
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
echo "FiDel EC2 Ingestor"
echo "=============================================="
echo "Region: $REGION"
echo "Action: $ACTION"
echo ""

# Check AWS CLI
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Function to get the latest Amazon Linux 2 AMI
get_latest_ami() {
    aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
                  "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text \
        --region "$REGION"
}

# Function to get or create security group
get_security_group() {
    local sg_name="fidel-ingestor-sg"

    # Check if exists
    local sg_id=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$sg_name" \
        --query 'SecurityGroups[0].GroupId' \
        --output text \
        --region "$REGION" 2>/dev/null || echo "None")

    if [ "$sg_id" = "None" ] || [ -z "$sg_id" ]; then
        echo "Creating security group..." >&2
        sg_id=$(aws ec2 create-security-group \
            --group-name "$sg_name" \
            --description "Security group for FiDel ingestor" \
            --region "$REGION" \
            --query 'GroupId' \
            --output text)

        # Allow SSH from anywhere (you may want to restrict this)
        aws ec2 authorize-security-group-ingress \
            --group-id "$sg_id" \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0 \
            --region "$REGION"

        echo "Created security group: $sg_id" >&2
    fi

    echo "$sg_id"
}

# Status action
if [ "$ACTION" = "status" ]; then
    echo "Checking ingestor status..."
    instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=fidel-ingestor" \
                  "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
        --output text \
        --region "$REGION")

    if [ -z "$instances" ]; then
        echo "No running ingestor instances found."
    else
        echo "Running ingestor instances:"
        echo "$instances"
    fi
    exit 0
fi

# Stop action
if [ "$ACTION" = "stop" ]; then
    echo "Stopping ingestor instances..."
    instance_ids=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=fidel-ingestor" \
                  "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[*].Instances[*].InstanceId' \
        --output text \
        --region "$REGION")

    if [ -z "$instance_ids" ]; then
        echo "No running ingestor instances found."
    else
        aws ec2 terminate-instances --instance-ids $instance_ids --region "$REGION"
        echo "Terminated instances: $instance_ids"
    fi
    exit 0
fi

# Start action
echo "Starting ingestor for symbol: $SYMBOL"
echo ""

# Get AMI and security group
AMI_ID=$(get_latest_ami)
SG_ID=$(get_security_group)

echo "Using AMI: $AMI_ID"
echo "Security Group: $SG_ID"
echo ""

# Create user data script
USER_DATA=$(cat << EOF
#!/bin/bash
set -e

# Update system
yum update -y

# Install Python 3.9 and pip
amazon-linux-extras install python3.8 -y
pip3 install --upgrade pip

# Install dependencies
pip3 install boto3 websockets aiohttp

# Create app directory
mkdir -p /opt/fidel
cd /opt/fidel

# Download ingestor service from S3 or inline
cat > ingestor_service.py << 'PYEOF'
$(cat "${PROJECT_ROOT}/src/deployment/ingestor_service.py")
PYEOF

# Create systemd service
cat > /etc/systemd/system/fidel-ingestor.service << 'SVCEOF'
[Unit]
Description=FiDel Data Ingestor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/fidel
Environment="FIDEL_RAW_BUCKET=${FIDEL_RAW_BUCKET}"
Environment="AWS_REGION=${REGION}"
ExecStart=/usr/bin/python3 /opt/fidel/ingestor_service.py --symbol ${SYMBOL}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCEOF

# Start service
systemctl daemon-reload
systemctl enable fidel-ingestor
systemctl start fidel-ingestor

# Log startup
echo "FiDel Ingestor started for ${SYMBOL}" | logger
EOF
)

# Create launch template
TEMPLATE_NAME="fidel-ingestor-template"

# Check if template exists and delete it
if aws ec2 describe-launch-templates --launch-template-names "$TEMPLATE_NAME" --region "$REGION" &>/dev/null; then
    aws ec2 delete-launch-template --launch-template-name "$TEMPLATE_NAME" --region "$REGION"
fi

# Create new template
echo "Creating launch template..."
aws ec2 create-launch-template \
    --launch-template-name "$TEMPLATE_NAME" \
    --version-description "FiDel ingestor v1" \
    --launch-template-data "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"IamInstanceProfile\": {
            \"Name\": \"fidel-ec2-profile\"
        },
        \"SecurityGroupIds\": [\"$SG_ID\"],
        \"UserData\": \"$(echo "$USER_DATA" | base64 | tr -d '\n')\",
        \"TagSpecifications\": [
            {
                \"ResourceType\": \"instance\",
                \"Tags\": [
                    {\"Key\": \"Name\", \"Value\": \"fidel-ingestor\"},
                    {\"Key\": \"Project\", \"Value\": \"FiDel\"},
                    {\"Key\": \"Symbol\", \"Value\": \"$SYMBOL\"}
                ]
            }
        ],
        \"InstanceMarketOptions\": {
            \"MarketType\": \"spot\",
            \"SpotOptions\": {
                \"SpotInstanceType\": \"persistent\",
                \"InstanceInterruptionBehavior\": \"stop\"
            }
        }
    }" \
    --region "$REGION"

echo ""

# Request Spot instance
echo "Requesting Spot instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --launch-template "LaunchTemplateName=$TEMPLATE_NAME" \
    --query 'Instances[0].InstanceId' \
    --output text \
    --region "$REGION")

echo "Instance requested: $INSTANCE_ID"

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
PRIVATE_IP=$(echo "$INSTANCE_INFO" | cut -f2)

echo ""
echo "=============================================="
echo "Ingestor Started!"
echo "=============================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Private IP: $PRIVATE_IP"
echo ""
echo "The ingestor will start automatically and begin streaming"
echo "$SYMBOL data to s3://${FIDEL_RAW_BUCKET}/trades/"
echo ""
echo "To check status:"
echo "  ssh ec2-user@$PUBLIC_IP 'sudo systemctl status fidel-ingestor'"
echo ""
echo "To view logs:"
echo "  ssh ec2-user@$PUBLIC_IP 'sudo journalctl -u fidel-ingestor -f'"
echo ""
echo "To stop:"
echo "  ./ec2-ingestor.sh --stop"
echo ""
echo "Estimated cost: ~\$0.003/hr (~\$2-3/month)"
