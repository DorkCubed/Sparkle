import s3fs
import os
import json
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class S3DataFetcher:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.fs = s3fs.S3FileSystem()

    def list_split_objects(self, parents, split):
        """
        Given a list of parent folders and a split name, returns a list of tuples
        containing the paths to the packet, header, field, and direction files for
        each split.

        Args:
            parents (list): List of parent folder names.
            split (str): Name of the split.

        Returns:
            list: List of tuples containing the paths to the packet, header, field,
            and direction files for each split.
        """
        files = []
        for parent in parents:
            base_path = f"{self.bucket_name}/{parent}/{split}"

            packet_files = sorted(self.fs.glob(f"{base_path}/packets/*.txt"))
            header_files = sorted(self.fs.glob(f"{base_path}/header/*.txt"))
            field_files = sorted(self.fs.glob(f"{base_path}/fields/*.txt"))
            direction_files = sorted(self.fs.glob(f"{base_path}/direction/*.txt"))

            for p, h, f, d in zip(
                packet_files, header_files, field_files, direction_files
            ):
                files.append((p, h, f, d))
        return files

    def build_manifest(self, parents):
        manifest = []

        total_flows = 0
        for parent in parents:
            flows = self.fs.ls(parent)
            total_flows += len(flows)

        logger.info(f"Found total {total_flows} flow directories under {parents}")
        for parent in parents:
            flows = self.fs.ls(parent)
            logger.info(f"Found {len(flows)} flow directories under {parent}")

            for i, flow in enumerate(tqdm(flows, desc=f"Processing flows in {parent}")):
                if not self.fs.isdir(flow):
                    continue

                logger.debug(f"[{i}/{len(flows)}] Processing flow: {flow}")

                packets = self.fs.glob(f"{flow}/packets/*.txt")
                headers = self.fs.glob(f"{flow}/header/*.txt")
                fields = self.fs.glob(f"{flow}/fields/*.txt")
                directions = self.fs.glob(f"{flow}/direction/*.txt")

                for p, h, f, d in zip(
                    sorted(packets), sorted(headers), sorted(fields), sorted(directions)
                ):
                    manifest.append(
                        {
                            "flow": os.path.basename(flow),
                            "packet": p,
                            "header": h,
                            "field": f,
                            "direction": d,
                        }
                    )

        return manifest

    def save_manifest_to_json(self, manifest, output_path):
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=4)

        print(f"Manifest saved to {output_path}")

    def read_files(self, s3_path):
        """
        Reads the contents of a file from an S3 path.
        """
        with self.fs.open(s3_path, "r") as f:
            return f.read()


if __name__ == "__main__":
    bucket_name = "netml-s3-bucket"
    # folder1 = 'Working_folder/input_aws/Wireshark_Sample_PCAPs/split'
    # folder2 = 'Working_folder/input_aws'
    folder = "Finetuning/iot_intrusion_dataset/split"
    fetcher = S3DataFetcher(bucket_name)
    manifest = fetcher.build_manifest(parents=[f"s3://{bucket_name}/{folder}"])

    current_dir = os.path.dirname(__file__)
    project_dir = os.path.abspath(
        os.path.join(current_dir, os.pardir, os.pardir, os.pardir, os.pardir)
    )
    manifest_path = os.path.join(project_dir, "manifest", "eval_manifest.json")

    fetcher.save_manifest_to_json(manifest, manifest_path)
