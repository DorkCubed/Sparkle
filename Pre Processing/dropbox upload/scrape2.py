from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests
import boto3

client = boto3.client(
    's3',
    aws_access_key_id="AKIA2UC3CH6ZCCUNPZVY",
    aws_secret_access_key="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"
)
BUCKET = "netml-s3-bucket"

website_url = 'http://205.174.165.80/CICDataset/DoHBrw-2020/Dataset/PCAPs/DoHMalicious/'
base_url = 'Dropbox/BITS WILP Research Group/DoHBrw-2020/DoHMalicious'
response = requests.get(website_url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')

    for link in links:
        file_url = urljoin(website_url, link.get('href'))
        year = link.get('href')

        if file_url.endswith('.zip'):
            request = requests.get(
                "https://drive.usercontent.google.com/download?id=1fTp7fWNGeQAF9LuA1_l2lM1CTBoaB5U2&export=download&authuser=0&confirm=t&uuid=005a41bb-a274-4fa1-bc10-de9bce5bfabb&at=APZUnTWNdjF3ssbtbMI7as0eeHrk:1724167481896", stream=True)
            open(f'./dump.file', 'wb').write(request.content)
            print(f'Uploading {file_url} to {file_url}')
            # client.upload_file(f'./dump.file', BUCKET, f"{base_url}/{link.get('href')}")


else:
    print(f'Failed to retrieve the website. Status code: {
          response.status_code}')
