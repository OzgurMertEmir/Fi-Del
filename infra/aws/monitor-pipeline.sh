#!/bin/bash
# monitor-pipeline.sh - Monitor FiDel AWS Pipeline
#
# Usage:
#   ./monitor-pipeline.sh           # Full status check
#   ./monitor-pipeline.sh --watch   # Continuous monitoring
#   ./monitor-pipeline.sh --logs    # View streamer logs
#   ./monitor-pipeline.sh --s3      # Check S3 data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"

# Load bucket names
if [ -f "${SCRIPT_DIR}/bucket-names.env" ]; then
    source "${SCRIPT_DIR}/bucket-names.env"
fi

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

print_status() {
    if [ "$2" = "ok" ]; then
        echo -e "  ${GREEN}✓${NC} $1"
    elif [ "$2" = "warn" ]; then
        echo -e "  ${YELLOW}⚠${NC} $1"
    else
        echo -e "  ${RED}✗${NC} $1"
    fi
}

check_ec2() {
    print_header "EC2 Instances"

    # Java Streamer
    STREAMER=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=fidel-java-streamer" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' \
        --output text \
        --region "$REGION" 2>/dev/null)

    if [ -n "$STREAMER" ] && [ "$STREAMER" != "None" ]; then
        INSTANCE_ID=$(echo "$STREAMER" | cut -f1)
        PUBLIC_IP=$(echo "$STREAMER" | cut -f2)
        print_status "Java Streamer: $INSTANCE_ID ($PUBLIC_IP)" "ok"
        echo "$PUBLIC_IP" > /tmp/streamer_ip.txt
    else
        print_status "Java Streamer: Not running" "error"
    fi

    # Ingestor (Python)
    INGESTOR=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=fidel-ingestor" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' \
        --output text \
        --region "$REGION" 2>/dev/null)

    if [ -n "$INGESTOR" ] && [ "$INGESTOR" != "None" ]; then
        print_status "Python Ingestor: $(echo $INGESTOR | cut -f1)" "ok"
    else
        print_status "Python Ingestor: Not running" "warn"
    fi
}

check_s3() {
    print_header "S3 Buckets"

    # Raw data
    RAW_COUNT=$(aws s3 ls "s3://${FIDEL_RAW_BUCKET}/trades/" --recursive 2>/dev/null | wc -l | tr -d ' ')
    RAW_SIZE=$(aws s3 ls "s3://${FIDEL_RAW_BUCKET}/trades/" --recursive --summarize 2>/dev/null | grep "Total Size" | awk '{print $3}')

    if [ "$RAW_COUNT" -gt 1 ]; then
        print_status "Raw Data: $RAW_COUNT files (${RAW_SIZE:-0} bytes)" "ok"
    else
        print_status "Raw Data: No data yet (sync every 5 min)" "warn"
    fi

    # Processed data
    PROC_COUNT=$(aws s3 ls "s3://${FIDEL_PROCESSED_BUCKET}/features/" --recursive 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PROC_COUNT" -gt 0 ]; then
        print_status "Processed Data: $PROC_COUNT files" "ok"
    else
        print_status "Processed Data: No data yet (waiting for Lambda)" "warn"
    fi

    # Models
    MODEL_COUNT=$(aws s3 ls "s3://${FIDEL_MODELS_BUCKET}/" --recursive 2>/dev/null | wc -l | tr -d ' ')
    print_status "Models Bucket: $MODEL_COUNT files" "ok"
}

check_lambda() {
    print_header "Lambda Function"

    LAMBDA_STATE=$(aws lambda get-function \
        --function-name fidel-feature-processor \
        --query 'Configuration.State' \
        --output text \
        --region "$REGION" 2>/dev/null)

    if [ "$LAMBDA_STATE" = "Active" ]; then
        print_status "fidel-feature-processor: Active" "ok"

        # Check recent invocations
        INVOCATIONS=$(aws cloudwatch get-metric-statistics \
            --namespace AWS/Lambda \
            --metric-name Invocations \
            --dimensions Name=FunctionName,Value=fidel-feature-processor \
            --start-time "$(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ)" \
            --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --period 3600 \
            --statistics Sum \
            --query 'Datapoints[0].Sum' \
            --output text \
            --region "$REGION" 2>/dev/null || echo "0")

        if [ "$INVOCATIONS" != "None" ] && [ "$INVOCATIONS" != "0" ]; then
            print_status "Invocations (last hour): $INVOCATIONS" "ok"
        else
            print_status "Invocations (last hour): 0 (waiting for S3 data)" "warn"
        fi
    else
        print_status "fidel-feature-processor: $LAMBDA_STATE" "error"
    fi
}

check_cloudwatch() {
    print_header "CloudWatch Alarms"

    ALARMS=$(aws cloudwatch describe-alarms \
        --alarm-name-prefix "FiDel" \
        --query 'MetricAlarms[*].[AlarmName,StateValue]' \
        --output text \
        --region "$REGION" 2>/dev/null)

    if [ -n "$ALARMS" ]; then
        echo "$ALARMS" | while read -r name state; do
            if [ "$state" = "OK" ]; then
                print_status "$name: OK" "ok"
            elif [ "$state" = "ALARM" ]; then
                print_status "$name: ALARM" "error"
            else
                print_status "$name: $state" "warn"
            fi
        done
    else
        print_status "No alarms configured" "warn"
    fi
}

show_recent_data() {
    print_header "Recent Data Files"

    echo "  Raw data (last 10 files):"
    aws s3 ls "s3://${FIDEL_RAW_BUCKET}/trades/" --recursive 2>/dev/null | tail -10 | while read -r line; do
        echo "    $line"
    done

    echo ""
    echo "  Processed data (last 10 files):"
    aws s3 ls "s3://${FIDEL_PROCESSED_BUCKET}/features/" --recursive 2>/dev/null | tail -10 | while read -r line; do
        echo "    $line"
    done
}

show_logs() {
    print_header "Streamer Logs"

    if [ -f /tmp/streamer_ip.txt ]; then
        IP=$(cat /tmp/streamer_ip.txt)
    else
        IP=$(aws ec2 describe-instances \
            --filters "Name=tag:Name,Values=fidel-java-streamer" \
                      "Name=instance-state-name,Values=running" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text \
            --region "$REGION" 2>/dev/null)
    fi

    if [ -n "$IP" ] && [ "$IP" != "None" ]; then
        echo "  Connecting to $IP..."
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ec2-user@$IP \
            'sudo journalctl -u fidel-streamer -n 50 --no-pager' 2>/dev/null || \
            echo "  SSH connection failed - instance may still be initializing"
    else
        echo "  No running streamer instance found"
    fi
}

show_lambda_logs() {
    print_header "Lambda Logs (last 5 minutes)"

    aws logs filter-log-events \
        --log-group-name "/aws/lambda/fidel-feature-processor" \
        --start-time $(($(date +%s) * 1000 - 300000)) \
        --query 'events[*].message' \
        --output text \
        --region "$REGION" 2>/dev/null | head -30 || echo "  No recent logs"
}

print_urls() {
    print_header "AWS Console Links"

    echo "  CloudWatch Dashboard:"
    echo "    https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=FiDel-Pipeline"
    echo ""
    echo "  S3 Raw Data:"
    echo "    https://s3.console.aws.amazon.com/s3/buckets/${FIDEL_RAW_BUCKET}?region=${REGION}&prefix=trades/"
    echo ""
    echo "  Lambda Logs:"
    echo "    https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#logsV2:log-groups/log-group/\$252Faws\$252Flambda\$252Ffidel-feature-processor"
    echo ""
    echo "  EC2 Instances:"
    echo "    https://${REGION}.console.aws.amazon.com/ec2/v2/home?region=${REGION}#Instances:"
}

# Main
case "${1:-}" in
    --watch)
        while true; do
            clear
            echo "FiDel Pipeline Monitor - $(date)"
            check_ec2
            check_s3
            check_lambda
            check_cloudwatch
            echo ""
            echo "Refreshing in 30 seconds... (Ctrl+C to exit)"
            sleep 30
        done
        ;;
    --logs)
        show_logs
        ;;
    --lambda-logs)
        show_lambda_logs
        ;;
    --s3)
        show_recent_data
        ;;
    --urls)
        print_urls
        ;;
    *)
        echo "FiDel Pipeline Monitor - $(date)"
        check_ec2
        check_s3
        check_lambda
        check_cloudwatch
        print_urls
        echo ""
        echo -e "${YELLOW}Tips:${NC}"
        echo "  ./monitor-pipeline.sh --watch       # Continuous monitoring"
        echo "  ./monitor-pipeline.sh --logs        # View streamer logs"
        echo "  ./monitor-pipeline.sh --lambda-logs # View Lambda logs"
        echo "  ./monitor-pipeline.sh --s3          # View recent S3 data"
        echo "  ./monitor-pipeline.sh --urls        # Show AWS console links"
        ;;
esac
