#!/usr/bin/env python3
"""
ingestor_service.py - Data ingestion service for AWS EC2

This service runs on an EC2 Spot instance and:
1. Connects to data sources (WebSocket streams, REST APIs)
2. Buffers incoming data locally
3. Uploads to S3 in configurable batches
4. Handles reconnection on disconnects
5. Gracefully handles Spot interruption warnings

Usage:
    python ingestor_service.py --config config.yaml
    python ingestor_service.py --symbol BTCUSDT --source binance

Environment variables:
    FIDEL_RAW_BUCKET: S3 bucket for raw data
    AWS_REGION: AWS region (default: us-east-1)
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ingestor")


@dataclass
class IngestorConfig:
    """Configuration for the data ingestor."""

    # Data source
    source: str = "binance"  # binance, custom_ws, rest_api
    symbol: str = "BTCUSDT"
    stream_type: str = "trade"  # trade, depth, aggTrade

    # S3 settings
    bucket: str = ""
    prefix: str = "trades"
    region: str = "us-east-1"

    # Buffering
    buffer_size: int = 1000  # messages before flush
    flush_interval: float = 5.0  # seconds
    compress: bool = True

    # Connection
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10

    # Spot instance handling
    check_spot_termination: bool = True
    termination_check_interval: float = 5.0


@dataclass
class DataBuffer:
    """In-memory buffer for incoming data."""

    messages: list[dict] = field(default_factory=list)
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None

    def add(self, message: dict, timestamp: Optional[datetime] = None) -> None:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        self.messages.append(message)

    def clear(self) -> list[dict]:
        messages = self.messages
        self.messages = []
        self.first_timestamp = None
        self.last_timestamp = None
        return messages

    def __len__(self) -> int:
        return len(self.messages)


class S3Uploader:
    """Handles uploading data to S3."""

    def __init__(self, config: IngestorConfig):
        self.config = config
        self.s3 = boto3.client("s3", region_name=config.region)
        self.bucket = config.bucket

    def upload_batch(
        self,
        messages: list[dict],
        symbol: str,
        timestamp: datetime,
    ) -> str:
        """Upload a batch of messages to S3."""
        # Create S3 key with partitioning
        date_str = timestamp.strftime("%Y-%m-%d")
        hour_str = timestamp.strftime("%H")
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")

        key = f"{self.config.prefix}/{symbol}/{date_str}/{hour_str}/{ts_str}"

        # Serialize data
        data = json.dumps(messages, default=str)

        if self.config.compress:
            key += ".json.gz"
            body = gzip.compress(data.encode("utf-8"))
            content_type = "application/gzip"
        else:
            key += ".json"
            body = data.encode("utf-8")
            content_type = "application/json"

        # Upload to S3
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
            Metadata={
                "symbol": symbol,
                "record_count": str(len(messages)),
                "source": self.config.source,
            },
        )

        logger.info(f"Uploaded {len(messages)} records to s3://{self.bucket}/{key}")
        return key


class SpotTerminationMonitor:
    """Monitors for EC2 Spot instance termination notices."""

    METADATA_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"

    def __init__(self):
        self.termination_time: Optional[datetime] = None
        self._session = None

    async def check_termination(self) -> bool:
        """Check if spot termination notice has been issued."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.METADATA_URL, timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        action = data.get("action")
                        action_time = data.get("time")
                        if action in ("terminate", "stop", "hibernate"):
                            self.termination_time = datetime.fromisoformat(
                                action_time.replace("Z", "+00:00")
                            )
                            logger.warning(
                                f"Spot termination notice: {action} at {action_time}"
                            )
                            return True
        except Exception:
            # No termination notice (404) or not running on EC2
            pass
        return False


class BinanceWebSocketClient:
    """WebSocket client for Binance streams."""

    WS_BASE_URL = "wss://stream.binance.com:9443/ws"

    def __init__(
        self,
        symbol: str,
        stream_type: str,
        on_message: Callable[[dict], None],
    ):
        self.symbol = symbol.lower()
        self.stream_type = stream_type
        self.on_message = on_message
        self._ws = None
        self._running = False

    @property
    def stream_name(self) -> str:
        return f"{self.symbol}@{self.stream_type}"

    @property
    def ws_url(self) -> str:
        return f"{self.WS_BASE_URL}/{self.stream_name}"

    async def connect(self) -> None:
        """Connect to WebSocket stream."""
        import websockets

        logger.info(f"Connecting to {self.ws_url}")
        self._ws = await websockets.connect(self.ws_url)
        self._running = True
        logger.info(f"Connected to {self.stream_name}")

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def listen(self) -> None:
        """Listen for messages."""
        if not self._ws:
            raise RuntimeError("Not connected")

        while self._running:
            try:
                message = await self._ws.recv()
                data = json.loads(message)
                self.on_message(data)
            except Exception as e:
                if self._running:
                    logger.error(f"WebSocket error: {e}")
                    raise


class DataIngestor:
    """Main data ingestion service."""

    def __init__(self, config: IngestorConfig):
        self.config = config
        self.buffer = DataBuffer()
        self.uploader = S3Uploader(config)
        self.spot_monitor = SpotTerminationMonitor()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._spot_task: Optional[asyncio.Task] = None
        self._ws_client: Optional[BinanceWebSocketClient] = None

        # Stats
        self.messages_received = 0
        self.batches_uploaded = 0
        self.start_time: Optional[datetime] = None

    def _on_message(self, data: dict) -> None:
        """Handle incoming message."""
        self.messages_received += 1
        timestamp = datetime.now(timezone.utc)

        # Add to buffer
        self.buffer.add(data, timestamp)

        # Check if buffer is full
        if len(self.buffer) >= self.config.buffer_size:
            asyncio.create_task(self._flush_buffer())

    async def _flush_buffer(self) -> None:
        """Flush buffer to S3."""
        if len(self.buffer) == 0:
            return

        messages = self.buffer.clear()
        timestamp = datetime.now(timezone.utc)

        try:
            self.uploader.upload_batch(messages, self.config.symbol, timestamp)
            self.batches_uploaded += 1
        except ClientError as e:
            logger.error(f"Failed to upload batch: {e}")
            # Re-add messages to buffer for retry
            for msg in messages:
                self.buffer.add(msg)

    async def _periodic_flush(self) -> None:
        """Periodically flush buffer."""
        while self._running:
            await asyncio.sleep(self.config.flush_interval)
            if len(self.buffer) > 0:
                await self._flush_buffer()

    async def _monitor_spot_termination(self) -> None:
        """Monitor for Spot termination."""
        while self._running:
            if await self.spot_monitor.check_termination():
                logger.warning("Spot termination detected! Flushing and shutting down...")
                await self._flush_buffer()
                await self.stop()
                break
            await asyncio.sleep(self.config.termination_check_interval)

    async def start(self) -> None:
        """Start the ingestor service."""
        logger.info(f"Starting ingestor for {self.config.symbol}")
        logger.info(f"Source: {self.config.source}")
        logger.info(f"S3 bucket: s3://{self.config.bucket}/{self.config.prefix}")

        self._running = True
        self.start_time = datetime.now(timezone.utc)

        # Start background tasks
        self._flush_task = asyncio.create_task(self._periodic_flush())

        if self.config.check_spot_termination:
            self._spot_task = asyncio.create_task(self._monitor_spot_termination())

        # Connect to data source
        reconnect_attempts = 0
        while self._running:
            try:
                if self.config.source == "binance":
                    self._ws_client = BinanceWebSocketClient(
                        symbol=self.config.symbol,
                        stream_type=self.config.stream_type,
                        on_message=self._on_message,
                    )
                    await self._ws_client.connect()
                    await self._ws_client.listen()
                else:
                    raise ValueError(f"Unknown source: {self.config.source}")

            except Exception as e:
                reconnect_attempts += 1
                if reconnect_attempts > self.config.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached: {e}")
                    break

                logger.warning(
                    f"Connection lost ({e}), reconnecting in {self.config.reconnect_delay}s "
                    f"(attempt {reconnect_attempts}/{self.config.max_reconnect_attempts})"
                )
                await asyncio.sleep(self.config.reconnect_delay)

    async def stop(self) -> None:
        """Stop the ingestor service."""
        logger.info("Stopping ingestor...")
        self._running = False

        # Disconnect WebSocket
        if self._ws_client:
            await self._ws_client.disconnect()

        # Cancel background tasks
        if self._flush_task:
            self._flush_task.cancel()
        if self._spot_task:
            self._spot_task.cancel()

        # Final flush
        await self._flush_buffer()

        # Print stats
        runtime = (
            (datetime.now(timezone.utc) - self.start_time).total_seconds()
            if self.start_time
            else 0
        )
        logger.info(f"Ingestor stopped after {runtime:.1f}s")
        logger.info(f"Messages received: {self.messages_received:,}")
        logger.info(f"Batches uploaded: {self.batches_uploaded}")


async def main():
    parser = argparse.ArgumentParser(description="FiDel Data Ingestor Service")
    parser.add_argument(
        "--source",
        default="binance",
        choices=["binance"],
        help="Data source (default: binance)",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--stream-type",
        default="trade",
        choices=["trade", "aggTrade", "depth"],
        help="Stream type (default: trade)",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("FIDEL_RAW_BUCKET", ""),
        help="S3 bucket for raw data",
    )
    parser.add_argument(
        "--prefix",
        default="trades",
        help="S3 prefix (default: trades)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Messages per batch (default: 1000)",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=5.0,
        help="Flush interval in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable gzip compression",
    )
    parser.add_argument(
        "--no-spot-check",
        action="store_true",
        help="Disable Spot termination checking",
    )
    args = parser.parse_args()

    if not args.bucket:
        logger.error("S3 bucket not specified. Use --bucket or set FIDEL_RAW_BUCKET")
        sys.exit(1)

    config = IngestorConfig(
        source=args.source,
        symbol=args.symbol,
        stream_type=args.stream_type,
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        buffer_size=args.buffer_size,
        flush_interval=args.flush_interval,
        compress=not args.no_compress,
        check_spot_termination=not args.no_spot_check,
    )

    ingestor = DataIngestor(config)

    # Handle shutdown signals
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(ingestor.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    await ingestor.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
