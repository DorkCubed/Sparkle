import subprocess


def unzip(zip_file):
    dest_dir = zip_file.rsplit(".", 1)[0]
    subprocess.run(["7z", "e", zip_file, f"-o{dest_dir}"])
