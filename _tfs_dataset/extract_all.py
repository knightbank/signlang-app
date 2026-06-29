"""Extract MediaPipe landmarks from ASL-HG processed dataset for 8 selected
letters + Mendeley Thai 'bor' images. Map each ASL letter to its Thai equivalent.

Output: extracted.json  ({label, landmark} list normalized via wrist-relative + 3D scale)
"""
import os, sys, json, math, glob, zipfile, time
import cv2
import mediapipe as mp
# Force UTF-8 on stdout (Windows defaults to cp874 with Thai locale)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.abspath(__file__))

# Thai → ASL mapping (high-confidence handshape matches)
THAI_ASL = {
    'อ': 'A',
    'บ': 'B',
    'ล': 'L',
    'ม': 'M',
    'น': 'N',
    'ส': 'S',
    'ว': 'W',
    'ย': 'Y',
}
ASL_THAI = {v: k for k, v in THAI_ASL.items()}

PER_CLASS_TARGET = 250  # usable landmark samples per Thai letter
mp_hands = mp.solutions.hands

def normalize(landmarks):
    if len(landmarks) != 21:
        return None
    w = landmarks[0]
    m9 = landmarks[9]
    dx, dy, dz = m9.x - w.x, m9.y - w.y, m9.z - w.z
    s = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
    out = []
    for p in landmarks:
        out.append((p.x - w.x) / s)
        out.append((p.y - w.y) / s)
        out.append((p.z - w.z) / s)
    return out

def extract_image(path, hands_solver):
    img = cv2.imread(path)
    if img is None:
        return None, 'cv2 load failed'
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands_solver.process(rgb)
    if not res.multi_hand_landmarks:
        return None, 'no hand'
    flat = normalize(res.multi_hand_landmarks[0].landmark)
    if flat is None:
        return None, 'normalize failed'
    return flat, None

def find_asl_letter_folders(unzip_root, letter):
    """Walk unzip_root to find ALL folders for given ASL letter (test + train)."""
    out = []
    for root, dirs, files in os.walk(unzip_root):
        base = os.path.basename(root)
        if base.upper() == letter.upper() and any(
            f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files
        ):
            out.append(root)
    return out

def unzip_if_needed(zip_path, dest):
    if os.path.isdir(dest) and any(os.scandir(dest)):
        print(f'  already unzipped at {dest}')
        return
    print(f'  unzipping {zip_path} ...')
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    print('  done')

def main():
    asl_zip = os.path.join(ROOT, 'ASL_Processed_Images.zip')
    asl_unzip = os.path.join(ROOT, 'asl_unzip')
    if os.path.exists(asl_zip):
        unzip_if_needed(asl_zip, asl_unzip)

    all_samples = []
    summary = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=1,
    ) as hands:
        # 1) Mendeley Thai บ images (verified Thai source)
        bor_folder = os.path.join(ROOT, 'bor')
        if os.path.isdir(bor_folder):
            n_ok, n_rej = 0, 0
            for p in sorted(glob.glob(os.path.join(bor_folder, '*.jpg'))):
                flat, err = extract_image(p, hands)
                if flat:
                    all_samples.append({'label': 'บ', 'landmark': flat, 'src': f'mendeley:{os.path.basename(p)}'})
                    n_ok += 1
                else:
                    n_rej += 1
            print(f'[บ Mendeley] ok={n_ok} rejected={n_rej}')
            summary.append({'label': 'บ', 'source': 'mendeley', 'ok': n_ok, 'rejected': n_rej})

        # 2) ASL-HG for each Thai letter via mapped ASL letter
        if os.path.isdir(asl_unzip):
            for thai, asl in THAI_ASL.items():
                folders = find_asl_letter_folders(asl_unzip, asl)
                if not folders:
                    print(f'[{thai}<->{asl}] folder NOT FOUND in ASL-HG')
                    summary.append({'label': thai, 'source': f'asl-hg:{asl}', 'ok': 0, 'rejected': 0, 'note': 'folder missing'})
                    continue
                imgs = []
                for fld in folders:
                    imgs.extend(sorted(
                        glob.glob(os.path.join(fld, '*.jpg')) +
                        glob.glob(os.path.join(fld, '*.jpeg')) +
                        glob.glob(os.path.join(fld, '*.png'))
                    ))
                n_ok, n_rej = 0, 0
                t0 = time.time()
                for p in imgs:
                    if n_ok >= PER_CLASS_TARGET:
                        break
                    flat, err = extract_image(p, hands)
                    if flat:
                        all_samples.append({'label': thai, 'landmark': flat, 'src': f'asl-hg:{asl}/{os.path.basename(p)}'})
                        n_ok += 1
                    else:
                        n_rej += 1
                dt = time.time() - t0
                print(f'[{thai}<->{asl}] folders={len(folders)} pool={len(imgs)} ok={n_ok} rejected={n_rej} ({dt:.1f}s)')
                summary.append({'label': thai, 'source': f'asl-hg:{asl}', 'ok': n_ok, 'rejected': n_rej, 'pool': len(imgs)})

    out = {
        'dataset': [{'label': s['label'], 'landmark': s['landmark']} for s in all_samples],
        'sources': [{'label': s['label'], 'src': s['src']} for s in all_samples],
        'summary': summary,
        'mapping': THAI_ASL,
    }
    out_path = os.path.join(ROOT, 'extracted.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    print(f'\nWrote {out_path}: {len(all_samples)} total samples across {len(set(s["label"] for s in all_samples))} classes')

if __name__ == '__main__':
    main()
