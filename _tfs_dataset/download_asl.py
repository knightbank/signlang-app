import urllib.request, sys, time, os

URL = 'https://data.mendeley.com/public-files/datasets/j4y5w2c8w9/files/5a69be0c-1f83-45b2-b7d9-6710b585cacc/file_downloaded'
OUT = 'ASL_Processed_Images.zip'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://data.mendeley.com/datasets/j4y5w2c8w9/1',
}

if os.path.exists(OUT) and os.path.getsize(OUT) > 100_000_000:
    print(f'Already downloaded: {os.path.getsize(OUT)/1e6:.1f} MB')
    sys.exit(0)

req = urllib.request.Request(URL, headers=HEADERS)
start = time.time()
with urllib.request.urlopen(req, timeout=120) as r:
    total = int(r.headers.get('Content-Length', 0))
    print(f'Downloading {total/1e6:.1f} MB...')
    written = 0
    with open(OUT, 'wb') as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk: break
            f.write(chunk)
            written += len(chunk)
            if total:
                pct = written * 100 / total
                rate = written / (time.time() - start + 1e-6) / 1e6
                sys.stdout.write(f'\r{pct:5.1f}%  {written/1e6:6.1f}/{total/1e6:.1f} MB  {rate:5.1f} MB/s    ')
                sys.stdout.flush()
print(f'\nDone: {os.path.getsize(OUT)/1e6:.1f} MB in {time.time()-start:.1f}s')
