"""Download freely available papers."""
import urllib.request
import os
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parents[1] / 'docs' / 'papers'
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# Papers with known free PDF URLs
papers = {
    # arXiv preprints
    'Hidalgo_2007_Science_product_space.pdf': 
        'https://arxiv.org/pdf/0708.2090',
    'Gao_2018_PhysicaA_China_regional_complexity.pdf':
        'https://arxiv.org/pdf/1703.01292',
    # Working papers / institutional repos
    'Stojkoski_Hidalgo_2025_optimizing_complexity.pdf':
        'https://www.tse-fr.eu/sites/default/files/TSE/documents/doc/wp/2025/wp_tse_1623.pdf',
    'Stojkoski_Hidalgo_2025_optimizing_complexity_v2.pdf':
        'https://repository.ukim.mk/bitstreams/9bb02837-f8c8-43f2-9a65-8cd72368890b/download',
    # Open access
    'Ash_Hansen_2023_text_algorithms_economics.pdf':
        'https://www.annualreviews.org/content/journals/10.1146/annurev-economics-082222-074352',
    # EconStor open access
    'Grabner_Hornykewycz_2022_capability_accumulation.pdf':
        'https://www.econstor.eu/bitstream/10419/216851/1/1697675417.pdf',
}

results = {}
for filename, url in papers.items():
    filepath = PAPERS_DIR / filename
    if filepath.exists():
        print(f"  SKIP (exists): {filename}")
        results[filename] = 'exists'
        continue
    try:
        print(f"  Downloading: {filename} ...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"    FAILED: too small ({len(data)} bytes)")
                results[filename] = f'failed: too small'
                continue
            filepath.write_bytes(data)
            print(f"    OK: {len(data):,} bytes")
            results[filename] = f'ok ({len(data):,} bytes)'
    except Exception as e:
        print(f"    FAILED: {e}")
        results[filename] = f'failed: {e}'

print(f"\n{'='*60}")
print("Download Summary:")
for f, r in results.items():
    print(f"  {f}: {r}")
