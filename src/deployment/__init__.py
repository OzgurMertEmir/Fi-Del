"""
FiDel AWS Deployment Module

This module contains Python services for AWS deployment:
- ingestor_service: Real-time data ingestion to S3
- lambda_processor: Feature engineering Lambda function
- training_job: ML training job for EC2 GPU instances

For deployment scripts, see infra/aws/
"""

from .ingestor_service import DataIngestor, IngestorConfig
from .lambda_processor import handler as lambda_handler, compute_features
from .training_job import train_model, download_features

__all__ = [
    "DataIngestor",
    "IngestorConfig",
    "lambda_handler",
    "compute_features",
    "train_model",
    "download_features",
]
