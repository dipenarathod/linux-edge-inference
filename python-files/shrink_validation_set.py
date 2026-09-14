import os
from pathlib import Path

# 1. Root folder containing one subfolder per class (e.g. "0", "1", ..., "999")
VAL_DIR = Path("./imagenetv2-top-images-format-val")

# 2. Which image extensions count as images to keep/delete
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

print(f"Scanning class folders under: {VAL_DIR.resolve()}")

# 3. Iterate over every class subfolder (skip anything that isn't a directory,
# in case stray files like a README end up at the top level).
class_dirs = sorted([d for d in VAL_DIR.iterdir() if d.is_dir()])
print(f"Found {len(class_dirs)} class folders.")

kept_count = 0
deleted_count = 0
skipped_empty = 0

for class_dir in class_dirs:
    # 4. List image files in this class folder, sorted for a deterministic pick.
    images = sorted(
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        # Folder has no images at all (e.g. already emptied by something else,
        # or wasn't a class folder in the first place). Don't touch it.
        skipped_empty += 1
        continue

    # 5. Always keep the first image (alphabetically) - this is what makes the
    # script safe to re-run: if only one image remains, it's already the "kept"
    # one, so there's nothing left to delete and the folder never goes empty.
    image_to_keep = images[0]
    images_to_delete = images[1:]

    for image_path in images_to_delete:
        os.remove(image_path)
        deleted_count += 1

    kept_count += 1

print("\n--- Done ---")
print(f"Class folders processed: {len(class_dirs)}")
print(f"Images kept (1 per folder): {kept_count}")
print(f"Images deleted: {deleted_count}")
if skipped_empty:
    print(f"Folders skipped (no images found): {skipped_empty}")