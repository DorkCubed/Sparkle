from urllib.parse import urljoin
from bs4 import BeautifulSoup
import dropbox
import requests


def upload_file_from_url(dbx, url, dest_path):
    res = requests.get(url, stream=True)
    dbx.files_upload(res.content, dest_path)


dbx = dropbox.Dropbox(oauth2_access_token='sl.BwHUFL63BhxwSY_3JFaVzK0Deq2W5e2tv3BZaf6vQKr2NVxbpUaIqmMUHRMUGbZPd0lB2bM_5bIy5ZGpMSg70t1pXV3jxIJBufeUJMAEYCAdkP8EGTBoxb_eVyTI0wbpDOBD6ip3_Jmydku5odrB',
                      app_key='19agponew3b055g', app_secret='y43dp1rsleuqypq')


website_url = 'https://www.wireshark.org/download/automated/captures/'
response = requests.get(website_url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')

    for link in links:
        file_url = urljoin(website_url, link.get('href'))
        year = link.get('href')
        years = year.split('-')

        if len(years) > 1:
            year = int(years[1])
        else:
            year = 0

        if year >= 2017:
            if file_url.endswith('.pcap'):
                path = '/BITS WILP Research Group/Wireshark Sample PCAPs/' + \
                    link.get('href')
                print(f'Uploading {file_url} to {path}')

                try:
                    dbx.files_get_metadata(path)
                    print('File already exists, skipping upload\n')

                except:
                    try:
                        upload_file_from_url(dbx, file_url, path)
                        print('Upload of ' + str(file_url) + ' complete\n')
                    except Exception as e:
                        print(f'Error downloading file: {e}\n')

else:
    print(f'Failed to retrieve the website. Status code: {
          response.status_code}')
