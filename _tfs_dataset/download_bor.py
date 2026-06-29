import urllib.request, os, concurrent.futures, time, sys

os.makedirs('bor', exist_ok=True)
with open('bor_urls.txt') as f:
    items = [line.strip().split('\t') for line in f if line.strip()]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
    'Referer': 'https://data.mendeley.com/datasets/rknd3wbz42/1',
}

def fetch(item):
    name, url = item
    path = os.path.join('bor', name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return name, 'cached'
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(path, 'wb') as f:
            f.write(data)
        return name, len(data)
    except Exception as e:
        return name, f'err:{e}'

# Serial downloads to avoid rate limiting
results = []
for i, item in enumerate(items):
    r = fetch(item)
    results.append(r)
    if isinstance(r[1], int):
        sys.stdout.write(f'\r[{i+1}/{len(items)}] {r[0]} ok ({r[1]}b)        ')
    elif r[1] == 'cached':
        sys.stdout.write(f'\r[{i+1}/{len(items)}] {r[0]} cached            ')
    else:
        sys.stdout.write(f'\r[{i+1}/{len(items)}] {r[0]} ERR              \n')
    sys.stdout.flush()
print()
ok = sum(1 for r in results if isinstance(r[1], int))
cached = sum(1 for r in results if r[1] == 'cached')
err = sum(1 for r in results if not isinstance(r[1], int) and r[1] != 'cached')
print(f'Done. fresh={ok} cached={cached} err={err}')
