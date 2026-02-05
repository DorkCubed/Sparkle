import json
import os
from tqdm import tqdm
from sparkle.model.utils.direction_encoder import encode_file
from sparkle.utils import get_project_root


OUTPUT_MANIFEST = os.path.join(get_project_root(), "manifest", "local_manifest.json")

def process_direction(output_manifest_path: str) -> None:
    with open(output_manifest_path, "r") as f:
        manifest = json.load(f)

    cleaned_manifest = []
    fields_removed = 0

    for item in tqdm(manifest, desc="Encoding direction"):
        direction_file = item.get("direction")
        if direction_file is None:
            continue

        try:
            encode_file(direction_file)
            cleaned_manifest.append(item)
        except ValueError:
            if os.path.exists(direction_file):
                fields_removed += 1

    with open(output_manifest_path, "w") as f:
        json.dump(cleaned_manifest, f, indent=4)
    
    print(f"Removed {fields_removed} fields. {len(cleaned_manifest)} entries now in manifest.")

    
if __name__ == '__main__':
    try:
        process_direction(OUTPUT_MANIFEST)
    except KeyboardInterrupt:
        print("Interrupted. Exiting.")