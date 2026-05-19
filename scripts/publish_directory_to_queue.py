from __future__ import annotations

import argparse
from pathlib import Path

import boto3

from picca_search.domain import SUPPORTED_IMAGE_EXTENSIONS
from picca_search.infrastructure.rabbitmq_queue import ImageJobMessage, RabbitMqImageJobQueue


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a directory to SeaweedFS S3 and publish image jobs.")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--rabbitmq-url", default="amqp://guest:guest@localhost:5672/%2F")
    parser.add_argument("--queue", default="image_jobs")
    parser.add_argument("--s3-endpoint-url", default="http://localhost:8333")
    parser.add_argument("--s3-bucket", default="images")
    parser.add_argument("--s3-access-key-id", default="seaweedfs")
    parser.add_argument("--s3-secret-access-key", default="seaweedfs")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    s3_client = boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint_url,
        aws_access_key_id=args.s3_access_key_id,
        aws_secret_access_key=args.s3_secret_access_key,
    )
    queue = RabbitMqImageJobQueue(args.rabbitmq_url, args.queue)
    try:
        for image_path in sorted(args.image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            object_key = f"{args.prefix}{image_path.name}"
            s3_client.upload_file(str(image_path), args.s3_bucket, object_key)
            queue.publish(ImageJobMessage(image_id=object_key))
            print(f"published\t{object_key}")
    finally:
        queue.close()


if __name__ == "__main__":
    main()
