import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def count_s3_objects(bucket_name, prefix):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    logger.info(f"Counting objects in bucket '{bucket_name}' with prefix '{prefix}'")

    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    count = 0
    page_num = 0
    for page in page_iterator:
        page_num += 1
        contents = page.get('Contents', [])
        page_count = len(contents)
        count += page_count
        logger.info(f"Page {page_num}: Found {page_count} objects (running total: {count})")

    logger.info(f"Total number of objects: {count}")
    return count


def list_s3_folders(bucket_name, prefix):
    s3 = boto3.client('s3')

    logger.info(f"Listing folders in bucket '{bucket_name}' with prefix '{prefix}'")

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter='/'
    )

    folders = []

    for page in page_iterator:
        common_prefixes = page.get('CommonPrefixes', [])
        for cp in common_prefixes:
            folder_name = cp['Prefix']
            logger.info(f"Found folder: {folder_name}")
            folders.append(folder_name)

    logger.info(f"Total folders found: {len(folders)}")
    return folders


def list_s3_folders_recursive(bucket_name, prefix):
    """
    Recursively lists all folders under the given prefix in an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): The folder path prefix to search within

    Returns:
        list: List of all folder paths found recursively
    """
    s3 = boto3.client('s3')
    folders = []

    def _list_folders(current_prefix):
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=current_prefix,
            Delimiter='/'
        )

        for page in page_iterator:
            # Add current level folders
            for cp in page.get('CommonPrefixes', []):
                folder_name = cp['Prefix']
                folders.append(folder_name)
                logger.info(f"Found folder: {folder_name}")
                # Recursively list subfolders
                _list_folders(folder_name)

    logger.info(f"Starting recursive folder listing in bucket '{bucket_name}' with prefix '{prefix}'")
    _list_folders(prefix)
    logger.info(f"Total folders found: {len(folders)}")
    return folders

# Example usage
if __name__ == "__main__":
    bucket = 'netml-s3-bucket'
    folder = 'Working_folder/input_aws/Wireshark_Sample_PCAPs/split/'  # Include trailing slash if needed
    folders = list_s3_folders_recursive(bucket, folder)
    print("Folders:")
    for f in sorted(folders):
        print(f)
