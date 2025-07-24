from urllib.parse import urljoin
from bs4 import BeautifulSoup
import os
import boto3

client = boto3.client(
    's3',
    aws_access_key_id="AKIA2UC3CH6ZCCUNPZVY",
    aws_secret_access_key="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"
)
BUCKET = "netml-s3-bucket"

base_url = 'Finetuning/USTC-TFC2016-master/USTC-TFC2016-master/Malware'
path = r"C:\Users\Artemis\Downloads\Pcaps"

for file in os.listdir(path):
    if not file.endswith('.pcap'):
        continue
    print(file)
    open(f'{path}/{file}', 'rb')
    print(f'Uploading {file}')
    client.upload_file(f'{path}/{file}', BUCKET,
                       f"{base_url}/{file}")
