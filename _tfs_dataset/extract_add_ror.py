"""Extract ASL R landmarks and merge into existing extracted.json to add a new
'ร' class without re-extracting the existing 8 classes (which are already in
extracted.json). Output: extracted_v2.json with 9 classes.

Run from _tfs_dataset/. Requires signlang_env (mediapipe 0.10.18, opencv-python).
"""
import os, sys, json, math, glob, time
import cv2
import mediapipe as mp

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.abspath(__file__))
PER_CLASS_TARGET = 250  # match the other classes
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
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands_solver.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    return normalize(res.multi_hand_landmarks[0].landmark)

def main():
    # Load existing baseline
    src_path = os.path.join(ROOT, 'extracted.json')
    with open(src_path, 'r', encoding='utf-8') as f:
        base = json.load(f)
    print(f'Existing baseline: {len(base["dataset"])} samples across {len(set(s["label"] for s in base["dataset"]))} classes')

    # Gather R images from train + test folders
    folders = [
        os.path.join(ROOT, 'asl_unzip', 'asl_processed', 'test', 'R'),
        os.path.join(ROOT, 'asl_unzip', 'asl_processed', 'train', 'R'),
    ]
    imgs = []
    for fld in folders:
        if os.path.isdir(fld):
            imgs.extend(sorted(glob.glob(os.path.join(fld, '*.jpg'))))
    print(f'Pool of ASL R images: {len(imgs)}')

    new_samples = []
    rejected = 0
    t0 = time.time()
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=1,
    ) as hands:
        for p in imgs:
            if len(new_samples) >= PER_CLASS_TARGET:
                break
            flat = extract_image(p, hands)
            if flat:
                new_samples.append({'label': 'ร', 'landmark': flat, 'src': f'asl-hg:R/{os.path.basename(p)}'})
            else:
                rejected += 1
    dt = time.time() - t0
    print(f'[ร<->R] accepted={len(new_samples)} rejected={rejected} ({dt:.1f}s)')

    # Append to baseline dataset + sources
    combined_dataset = base['dataset'] + [{'label': s['label'], 'landmark': s['landmark']} for s in new_samples]
    combined_sources = base.get('sources', []) + [{'label': s['label'], 'src': s['src']} for s in new_samples]

    summary = base.get('summary', []) + [{
        'label': 'ร',
        'source': 'asl-hg:R',
        'ok': len(new_samples),
        'rejected': rejected,
        'pool': len(imgs),
    }]
    mapping = base.get('mapping', {})
    mapping['ร'] = 'R'

    out = {
        'dataset': combined_dataset,
        'sources': combined_sources,
        'summary': summary,
        'mapping': mapping,
    }
    out_path = os.path.join(ROOT, 'extracted_v2.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    labels = sorted(set(s['label'] for s in combined_dataset))
    print(f'\nWrote {out_path}: total {len(combined_dataset)} samples / {len(labels)} classes')
    print(f'Classes: {labels}')

if __name__ == '__main__':
    main()
