"""Preprocess iLIDS-VID dataset for TF-CLIP."""
import os
import os.path as osp
import shutil
import json
import scipy.io as sio
from glob import glob

root = '/data1/lgf/TF-CLIP/data/iLIDS-VID'
raw_dir = osp.join(root, 'raw')
exdir = osp.join(raw_dir, 'i-LIDS-VID')
images_dir = osp.join(root, 'images')

# Step 1: Parse sequence filenames and build identities
print("Step 1: Parsing sequence filenames...")
fpaths = sorted(glob(osp.join(exdir, 'sequences', '*/*/*.png')))
print(f"  Found {len(fpaths)} frames")

# identities[pid][cam] = list of filenames
# Original format: cam{C}_person{NNN}_{FFFF}.png
identities_raw = [[[] for _ in range(2)] for _ in range(320)]  # max 319 persons

for fpath in fpaths:
    fname = osp.basename(fpath)
    if fname.startswith('.'):
        continue
    # Parse: cam1_person001_00317.png
    parts = fname.replace('.png', '').split('_')
    cam = int(parts[0].replace('cam', ''))  # 1 or 2
    pid = int(parts[1].replace('person', ''))  # 1-319
    frame = int(parts[2])  # frame number
    identities_raw[pid][cam - 1].append((frame, fpath))

# Sort frames within each person-camera
for pid in range(len(identities_raw)):
    for cam in range(2):
        identities_raw[pid][cam].sort(key=lambda x: x[0])

# Filter out empty identities
identities_clean = []
for pid in range(len(identities_raw)):
    cams = identities_raw[pid]
    if any(len(c) > 0 for c in cams):
        identities_clean.append(cams)

print(f"  Found {len(identities_clean)} identities")

# Step 2: Create images directory and copy/rename files
print("Step 2: Copying and renaming images...")
os.makedirs(images_dir, exist_ok=True)

identities_images = []
new_pid = 0
for pid_idx, cams in enumerate(identities_clean):
    new_cams = []
    for cam_idx, frame_list in enumerate(cams):
        cam_files = []
        for frame_idx, (orig_frame, fpath) in enumerate(frame_list):
            new_fname = f'{new_pid:08d}_{cam_idx:02d}_{frame_idx:04d}.png'
            dst = osp.join(images_dir, new_fname)
            if not osp.exists(dst):
                shutil.copy2(fpath, dst)
            cam_files.append(new_fname)
        new_cams.append(cam_files)
    identities_images.append(new_cams)
    new_pid += 1

print(f"  Total identities: {len(identities_images)}")
total_frames = sum(sum(len(c) for c in cams) for cams in identities_images)
print(f"  Total frames: {total_frames}")

# Step 3: Create meta.json
print("Step 3: Creating meta.json...")
meta = {
    'name': 'iLIDS-sequence',
    'shot': 'sequence',
    'num_cameras': 2,
    'identities': identities_images
}
with open(osp.join(root, 'meta.json'), 'w') as f:
    json.dump(meta, f)

# Step 4: Create splits.json from .mat file
print("Step 4: Creating splits.json...")
mat_path = osp.join(raw_dir, 'train-test people splits', 'train_test_splits_ilidsvid.mat')
data = sio.loadmat(mat_path)
person_list = data['ls_set']
num = len(identities_images)
print(f"  .mat split matrix shape: {person_list.shape}")

splits = []
for i in range(person_list.shape[0]):
    pids = (person_list[i] - 1).tolist()  # convert to 0-indexed
    # Filter to only valid pids
    pids = [p for p in pids if p < num]
    trainval_pids = sorted(pids[:num // 2])
    test_pids = sorted(pids[num // 2:])
    split = {
        'trainval': trainval_pids,
        'query': test_pids,
        'gallery': test_pids
    }
    splits.append(split)
    print(f"  Split {i}: trainval={len(trainval_pids)}, test={len(test_pids)}")

with open(osp.join(root, 'splits.json'), 'w') as f:
    json.dump(splits, f)

# Step 5: Verify
print("\nStep 5: Verification...")
assert osp.isdir(images_dir), "images/ dir missing"
assert osp.isfile(osp.join(root, 'meta.json')), "meta.json missing"
assert osp.isfile(osp.join(root, 'splits.json')), "splits.json missing"
print("  images/ dir: OK")
print("  meta.json: OK")
print("  splits.json: OK")
print("\nDone! Dataset ready at:", root)
