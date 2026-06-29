"""Add ASL J landmarks as the จ class on top of extracted_v2.json.

Intentional negative-result experiment: จ is a 2-stroke motion sign in TSL,
ASL J is also a motion sign. Feeding multi-pose static frames into a
single-frame Dense MLP is expected to underperform vs the 1-stroke letters
and possibly degrade some of them via confusion. The result will be
documented in the paper as motivation for the GRU sequence model.
"""
import os, sys, json, math, glob, time
import cv2
import mediapipe as mp

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.abspath(__file__))
PER_CLASS_TARGET = 250
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
    # Read v2 baseline (8 classes + ร = 9 classes)
    src_path = os.path.join(ROOT, 'extracted_v2.json')
    with open(src_path, 'r', encoding='utf-8') as f:
        base = json.load(f)
    print(f'Baseline (v2): {len(base["dataset"])} samples / {len(set(s["label"] for s in base["dataset"]))} classes')

    folders = [
        os.path.join(ROOT, 'asl_unzip', 'asl_processed', 'test', 'J'),
        os.path.join(ROOT, 'asl_unzip', 'asl_processed', 'train', 'J'),
    ]
    imgs = []
    for fld in folders:
        if os.path.isdir(fld):
            imgs.extend(sorted(glob.glob(os.path.join(fld, '*.jpg'))))
    print(f'ASL J pool: {len(imgs)} images')

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
                new_samples.append({'label': 'จ', 'landmark': flat, 'src': f'asl-hg:J/{os.path.basename(p)}'})
            else:
                rejected += 1
    dt = time.time() - t0
    print(f'[จ<->J] accepted={len(new_samples)} rejected={rejected} ({dt:.1f}s)')
    print('  NOTE: ASL J is a motion sign. Static frames span the J trajectory --')
    print('        expect inconsistent handshapes within this class.')

    combined_dataset = base['dataset'] + [{'label': s['label'], 'landmark': s['landmark']} for s in new_samples]
    combined_sources = base.get('sources', []) + [{'label': s['label'], 'src': s['src']} for s in new_samples]
    summary = base.get('summary', []) + [{
        'label': 'จ',
        'source': 'asl-hg:J',
        'ok': len(new_samples),
        'rejected': rejected,
        'pool': len(imgs),
        'note': 'motion-bearing ASL letter, static frames mix start/mid/end of J trajectory',
    }]
    mapping = base.get('mapping', {})
    mapping['จ'] = 'J'

    out = {
        'dataset': combined_dataset,
        'sources': combined_sources,
        'summary': summary,
        'mapping': mapping,
    }
    out_path = os.path.join(ROOT, 'extracted_v3.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    labels = sorted(set(s['label'] for s in combined_dataset))
    print(f'\nWrote {out_path}: total {len(combined_dataset)} samples / {len(labels)} classes')
    print(f'Classes: {labels}')

if __name__ == '__main__':
    main()
