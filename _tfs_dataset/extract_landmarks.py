"""Extract MediaPipe Hands landmarks from images and save normalized features.

Normalization matches the JS composable (useHandLandmarks.js):
- translate: subtract wrist (idx 0) from every point
- scale: divide by 3D Euclidean distance from wrist to middle-MCP (idx 9)
- flatten to 63 floats (21 points x [x,y,z])
"""
import os, sys, json, math, glob
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def normalize(landmarks):
    """landmarks: list of 21 (x,y,z) tuples or NormalizedLandmark protos.
    Returns flat list of 63 floats or None."""
    if len(landmarks) != 21:
        return None
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    dz = middle_mcp.z - wrist.z
    scale = math.sqrt(dx*dx + dy*dy + dz*dz)
    if scale == 0:
        scale = 1.0
    out = []
    for p in landmarks:
        out.append((p.x - wrist.x) / scale)
        out.append((p.y - wrist.y) / scale)
        out.append((p.z - wrist.z) / scale)
    return out

def extract_from_folder(folder, label, hands_solver):
    """Return list of {label, landmark} for detectable images."""
    paths = sorted(glob.glob(os.path.join(folder, '*.jpg'))) + \
            sorted(glob.glob(os.path.join(folder, '*.png')))
    samples = []
    rejected = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            rejected.append((p, 'cv2 load failed'))
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands_solver.process(rgb)
        if not res.multi_hand_landmarks:
            rejected.append((p, 'no hand'))
            continue
        # Use first detected hand (matches frontend behavior)
        norm = normalize(res.multi_hand_landmarks[0].landmark)
        if norm is None:
            rejected.append((p, 'normalize failed'))
            continue
        samples.append({'label': label, 'landmark': norm, 'source': os.path.basename(p)})
    return samples, rejected

def main():
    folders = [
        ('bor', 'บ'),
        ('or', 'อ'),
    ]
    all_samples = []
    summary = []
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=1,
    ) as hands:
        for folder, label in folders:
            if not os.path.isdir(folder):
                print(f'[skip] no folder {folder}/')
                continue
            samples, rejected = extract_from_folder(folder, label, hands)
            print(f'[{label}] folder={folder} accepted={len(samples)} rejected={len(rejected)}')
            for r in rejected[:5]:
                print('   reject:', os.path.basename(r[0]), '-', r[1])
            all_samples.extend(samples)
            summary.append({
                'label': label,
                'folder': folder,
                'accepted': len(samples),
                'rejected': len(rejected),
            })

    out = {
        'dataset': [{'label': s['label'], 'landmark': s['landmark']} for s in all_samples],
        'sources': [{'label': s['label'], 'source': s['source']} for s in all_samples],
        'summary': summary,
    }
    with open('extracted.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    print(f'\nWrote extracted.json: {len(all_samples)} total samples')

if __name__ == '__main__':
    main()
