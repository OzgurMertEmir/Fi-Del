#!/bin/bash
# ec2-java-streamer.sh - Deploy Binance Data Streamer (Java) to AWS EC2
#
# This script:
# 1. Uploads the JAR to S3
# 2. Launches an EC2 Spot instance with Java 17
# 3. Runs the streamer and syncs data to S3
#
# Usage:
#   ./ec2-java-streamer.sh [--region us-east-1]
#   ./ec2-java-streamer.sh --stop
#   ./ec2-java-streamer.sh --status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STREAMER_DIR="${PROJECT_ROOT}/binance-data-streamer"
JAR_FILE="${STREAMER_DIR}/target/binance-data-streamer-0.0.1-SNAPSHOT.jar"
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="t3.small"  # 2 vCPU, 2GB RAM - good for Java
ACTION="start"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            REGION="$2"
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
        --logs)
            ACTION="logs"
            shift
            ;;
        --build)
            ACTION="build"
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
echo "FiDel Binance Data Streamer (Java)"
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

# Build action
if [ "$ACTION" = "build" ]; then
    echo "Building JAR..."
    cd "$STREAMER_DIR"
    ./mvnw clean package -DskipTests
    echo "Build complete: $JAR_FILE"
    exit 0
fi

# Status action
if [ "$ACTION" = "status" ]; then
    echo "Checking streamer status..."
    instances=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=fidel-java-streamer" \
                  "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress,LaunchTime]' \
        --output table \
        --region "$REGION")

    if [ -z "$instances" ] || [ "$instances" = "None" ]; then
        echo "No running streamer instances found."
    else
        echo "$instances"
    fi
    exit 0
fi

# Logs action
if [ "$ACTION" = "logs" ]; then
    echo "Fetching recent logs..."
    INSTANCE_IP=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=fidel-java-streamer" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --region "$REGION")

    if [ -z "$INSTANCE_IP" ] || [ "$INSTANCE_IP" = "None" ]; then
        echo "No running instance found."
        exit 1
    fi

    echo "Instance IP: $INSTANCE_IP"
    echo "SSH to view logs: ssh ec2-user@$INSTANCE_IP 'sudo journalctl -u fidel-streamer -f'"
    exit 0
fi

# Stop action
if [ "$ACTION" = "stop" ]; then
    echo "Stopping streamer instances..."
    instance_ids=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=fidel-java-streamer" \
                  "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[*].Instances[*].InstanceId' \
        --output text \
        --region "$REGION")

    if [ -z "$instance_ids" ]; then
        echo "No running streamer instances found."
    else
        aws ec2 terminate-instances --instance-ids $instance_ids --region "$REGION"
        echo "Terminated instances: $instance_ids"
    fi
    exit 0
fi

# ============================================
# Start Action
# ============================================

# Check if JAR exists
if [ ! -f "$JAR_FILE" ]; then
    echo "JAR file not found. Building..."
    cd "$STREAMER_DIR"
    ./mvnw clean package -DskipTests
fi

echo "JAR file: $JAR_FILE"
echo "Size: $(ls -lh "$JAR_FILE" | awk '{print $5}')"
echo ""

# Upload JAR to S3
JAR_S3_KEY="artifacts/binance-data-streamer.jar"
echo "Uploading JAR to S3..."
aws s3 cp "$JAR_FILE" "s3://${FIDEL_MODELS_BUCKET}/${JAR_S3_KEY}" --region "$REGION"
echo "Uploaded to: s3://${FIDEL_MODELS_BUCKET}/${JAR_S3_KEY}"
echo ""

# Get Amazon Linux 2023 AMI (has newer packages)
echo "Finding AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-*-x86_64" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text \
    --region "$REGION")

echo "Using AMI: $AMI_ID"

# Get or create security group
SG_NAME="fidel-streamer-sg"
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region "$REGION" 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Security group for FiDel Java streamer" \
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
echo ""

# Create user data script
USER_DATA=$(cat << 'USERDATA_EOF'
#!/bin/bash
set -e

exec > >(tee /var/log/streamer-setup.log) 2>&1
echo "Starting streamer setup at $(date)"

# Install Java 17
dnf install -y java-17-amazon-corretto-headless

# Create app directory
mkdir -p /opt/fidel/persistence
cd /opt/fidel

# Download JAR from S3
USERDATA_EOF
)

USER_DATA="${USER_DATA}
aws s3 cp s3://${FIDEL_MODELS_BUCKET}/${JAR_S3_KEY} /opt/fidel/binance-data-streamer.jar --region ${REGION}

# Create application.properties
cat > /opt/fidel/application.properties << 'PROPS'
spring.main.web-application-type=none

binancedatastreamer.exchange=BINANCE
binancedatastreamer.messageproducer=SIMPLE

# Symbols to stream
binancedatastreamer.symbols=BTCUSDT:BTCUSDT:1,ETHUSDT:ETHUSDT:1,SOLUSDT:SOLUSDT:1,BNBUSDT:BNBUSDT:1,XRPUSDT:XRPUSDT:1,DOGEUSDT:DOGEUSDT:1

binancedatastreamer.persistence.folder=/opt/fidel/persistence
binancedatastreamer.persistence.resetDays=6,7
binancedatastreamer.persistence.mode=CRYPTO

binancedatastreamer.enableVolumeTracking=true
binance.volume.reset-cron=0 0 0 * * *
binance.volume.timezone=UTC

binancedatastreamer.marketdata.binance.websocket-url=wss://fstream.binance.com/stream
PROPS

# Create S3 sync script
cat > /opt/fidel/sync-to-s3.sh << 'SYNC'
#!/bin/bash
# Sync persistence files to S3 raw data bucket
BUCKET=\"${FIDEL_RAW_BUCKET}\"
REGION=\"${REGION}\"

for file in /opt/fidel/persistence/*.store; do
    if [ -f \"\$file\" ]; then
        symbol=\$(basename \"\$file\" .store)
        date_str=\$(date +%Y-%m-%d)
        hour_str=\$(date +%H)

        # Upload with timestamp
        aws s3 cp \"\$file\" \"s3://\${BUCKET}/trades/\${symbol}/\${date_str}/\${hour_str}/\$(date +%Y%m%d_%H%M%S).store\" --region \"\${REGION}\"
    fi
done
SYNC
chmod +x /opt/fidel/sync-to-s3.sh

# Set environment variables
export FIDEL_RAW_BUCKET='${FIDEL_RAW_BUCKET}'
export REGION='${REGION}'

# Update sync script with actual values
sed -i \"s|\\\${FIDEL_RAW_BUCKET}|${FIDEL_RAW_BUCKET}|g\" /opt/fidel/sync-to-s3.sh
sed -i \"s|\\\${REGION}|${REGION}|g\" /opt/fidel/sync-to-s3.sh

# Create systemd service
cat > /etc/systemd/system/fidel-streamer.service << 'SVC'
[Unit]
Description=FiDel Binance Data Streamer
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/fidel
ExecStart=/usr/bin/java -Xmx1g -jar /opt/fidel/binance-data-streamer.jar --spring.config.location=/opt/fidel/application.properties
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVC

# Create S3 sync timer (every 5 minutes)
cat > /etc/systemd/system/fidel-sync.service << 'SYNCSVC'
[Unit]
Description=FiDel S3 Sync

[Service]
Type=oneshot
ExecStart=/opt/fidel/sync-to-s3.sh
SYNCSVC

cat > /etc/systemd/system/fidel-sync.timer << 'SYNCTIMER'
[Unit]
Description=FiDel S3 Sync Timer

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
SYNCTIMER

# Start services
systemctl daemon-reload
systemctl enable fidel-streamer
systemctl start fidel-streamer
systemctl enable fidel-sync.timer
systemctl start fidel-sync.timer

echo 'FiDel Binance Streamer started at \$(date)' | logger
echo 'Setup complete!'
"

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
                "VolumeSize": 30,
                "VolumeType": "gp3",
                "DeleteOnTermination": true
            }
        }
    ]
}
EOF
)

# Get current spot price
SPOT_PRICE=$(aws ec2 describe-spot-price-history \
    --instance-types "$INSTANCE_TYPE" \
    --product-descriptions "Linux/UNIX" \
    --query 'SpotPriceHistory[0].SpotPrice' \
    --output text \
    --region "$REGION")

echo "Current spot price for $INSTANCE_TYPE: \$$SPOT_PRICE/hr"

# Add 10% buffer to spot price
SPOT_BID=$(echo "$SPOT_PRICE * 1.1" | bc -l | xargs printf "%.4f")
echo "Bidding: \$$SPOT_BID/hr"
echo ""

# Request Spot instance
SPOT_REQUEST=$(aws ec2 request-spot-instances \
    --spot-price "$SPOT_BID" \
    --instance-count 1 \
    --type "persistent" \
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
    --tags "Key=Name,Value=fidel-java-streamer" \
           "Key=Project,Value=FiDel" \
           "Key=Component,Value=BinanceStreamer" \
    --region "$REGION"

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Get instance IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --region "$REGION")

echo ""
echo "=============================================="
echo "Binance Data Streamer Deployed!"
echo "=============================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Instance Type: $INSTANCE_TYPE"
echo "Public IP: $PUBLIC_IP"
echo "Spot Price: \$$SPOT_PRICE/hr (~\$$(echo "$SPOT_PRICE * 720" | bc -l | xargs printf "%.0f")/month)"
echo ""
echo "The streamer will start automatically after setup (~2-3 min)."
echo "Data will be synced to S3 every 5 minutes."
echo ""
echo "Streaming symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT"
echo ""
echo "Data will be saved to:"
echo "  s3://${FIDEL_RAW_BUCKET}/trades/{SYMBOL}/{DATE}/{HOUR}/"
echo ""
echo "To check status:"
echo "  ./ec2-java-streamer.sh --status"
echo ""
echo "To view logs:"
echo "  ssh ec2-user@$PUBLIC_IP 'sudo journalctl -u fidel-streamer -f'"
echo ""
echo "To stop:"
echo "  ./ec2-java-streamer.sh --stop"
