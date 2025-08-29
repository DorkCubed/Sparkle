import os
import json

current_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
manifest_path = os.path.join(project_dir, "manifest", "manifest.json")

with open(manifest_path, "r") as f:
    manifest = json.load(f)

files = [(m["packet"], m["header"], m["field"], m["direction"]) for m in manifest]

print(files[1])
