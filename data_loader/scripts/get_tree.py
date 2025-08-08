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

# Example usage
if __name__ == "__main__":
    bucket = 'netml-s3-bucket'
    folder = 'Working_folder/input_aws/split/'  # Include trailing slash if needed
    folders = list_s3_folders(bucket, folder)
    print("Folders:")
    for f in folders:
        print(f)
