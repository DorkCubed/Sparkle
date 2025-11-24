import s3fs
import os
import json
import logging
import random
from sparkle.configs.config import Config
from sparkle.utils import get_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class S3SampleDataFetcher:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.fs = s3fs.S3FileSystem()

    def load_manifest_json(self, path):
        """
        Load an existing manifest.json that already contains fields:
        packet, header, field, direction.
        """
        logger.info(f"Loading existing manifest from {path}")
        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Manifest JSON must contain a list of entries.")

        logger.info(f"Loaded {len(data)} entries.")
        return data

    def sample_manifest(self, manifest, k=1000):
        if len(manifest) < k:
            raise ValueError(f"Manifest contains only {len(manifest)} entries but k={k}.")

        logger.info(f"Sampling {k} entries from {len(manifest)} available.")
        random.seed(42)
        return random.sample(manifest, k)

    def save_manifest_to_json(self, manifest, output_path):
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Saved sampled manifest to {output_path}")


if __name__ == "__main__":
    bucket_name = "netml-s3-bucket"
    fetcher = S3SampleDataFetcher(bucket_name)

    config = Config()

    current_dir = os.path.dirname(__file__)
    project_dir = get_project_root()

    original_manifest_path = config.manifest_path

    sample_length = 10

    sampled_manifest_path = os.path.join(project_dir, "manifest", f"manifest_{sample_length}.json")

    manifest = fetcher.load_manifest_json(original_manifest_path)
    sampled = fetcher.sample_manifest(manifest, k=sample_length)
    fetcher.save_manifest_to_json(sampled, sampled_manifest_path)
