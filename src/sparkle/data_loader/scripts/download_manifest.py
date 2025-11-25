import json
import os
import s3fs
from urllib.parse import urlparse
from sparkle.configs.config import Config
from sparkle.utils import get_project_root

config = Config()
INPUT_MANIFEST = config.manifest_path
project_root = get_project_root()
OUTPUT_MANIFEST = os.path.join(project_root, "manifest", "local_manifest_10.json")

LOCAL_BASE = os.path.join(project_root, "dataset", "local_10")

fs = s3fs.S3FileSystem()


def download_s3_uri(uri: str, base: str) -> str:
    p = urlparse(uri)
    bucket = p.netloc
    key = p.path.lstrip("/")

    local_path = os.path.join(base, key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    fs.get(f"{bucket}/{key}", local_path)
    return local_path


def process_manifest():
    with open(INPUT_MANIFEST, "r") as f:
        manifest = json.load(f)

    new_manifest = []
    fields = ["packet", "header", "field", "direction"]

    for item in manifest:
        out = {"flow": item["flow"]}
        for f_name in fields:
            out[f_name] = download_s3_uri(item[f_name], LOCAL_BASE)
        new_manifest.append(out)

    with open(OUTPUT_MANIFEST, "w") as f:
        json.dump(new_manifest, f, indent=4)


if __name__ == "__main__":
    process_manifest()
