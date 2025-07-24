import dropbox
import requests

def upload_file_from_url(dbx, url, dest_path):
    res = requests.get(url, stream=True)
    dbx.files_upload(res.content, dest_path)


dbx = dropbox.Dropbox(oauth2_access_token='sl.Bv_sEgXZ--IcKVBIydMd5hFRJge0cwY1kPKugk-NibOGtzLNsZd0j9sKjkfHEAeOLZkoysGY9P1L_h_29fxM4ggSDH85TdA6J8z7mthg38Qr9LDxMKMUb3SehkK-1CXZKjIHBZJ20BUP89jr2Y80', app_key='19agponew3b055g', app_secret='y43dp1rsleuqypq')
url = "https://cdn.discordapp.com/attachments/1085549674963943424/1207775056193781800/GGZuywHWIAAehYw.jpg?ex=65e0df2c&is=65ce6a2c&hm=765e8df21f1998aee948a3aa5fbfdf2ce7431fef09ca57e5dfb7cce5a29752b0&"
dest_path = '/my-file.jpg'
upload_file_from_url(dbx, url, dest_path)