import s3fs

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

            for p, h, f, d in zip(packet_files, header_files, field_files, direction_files):
                files.append((p, h, f, d))
        return files

    def read_files(self, s3_path):
        """
        Reads the contents of a file from an S3 path.
        """
        with self.fs.open(s3_path, 'r') as f:
            return f.read()
