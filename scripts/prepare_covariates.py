"""下载WDI基年GDP，记录来源；缺GDP的经济体使用出口规模代理，绝不伪造GDP。"""
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pipeline import ASIA_PACIFIC_ISO3, PROC, RAW, resource_product_mask

URL = 'https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?date=2000&format=json&per_page=400'


def main():
    target = ROOT / 'data/raw/world_bank/gdp_2000.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(URL, timeout=90) as response:
            body = response.read(5_000_000)
        obj = json.loads(body)
        if not isinstance(obj, list) or len(obj) != 2 or obj[0].get('pages') != 1:
            raise ValueError('WDI响应不完整')
        target.write_bytes(body)
    body = target.read_bytes()
    obj = json.loads(body)
    gdps = {r['countryiso3code']: r['value'] for r in obj[1]
            if r['date'] == '2000' and r['value'] is not None and r['value'] > 0}
    # 从管线输出读取全球国家列表（而非硬编码亚太子集）
    countries_meta = pd.read_parquet(PROC / 'country_metadata.parquet')
    countries = countries_meta['country_iso3'].tolist()
    available = {c: gdps[c] for c in countries if c in gdps}
    if len(available) < 50:
        raise ValueError('可用GDP不足，不采用手填数值')
    low, high = np.log(list(available.values())).min(), np.log(list(available.values())).max()
    cy = pd.read_csv(RAW / 'hs92_country_year.csv')
    exports = cy[cy.year == 2000].set_index('country_iso3_code').export_value
    exports = exports.reindex(countries)
    missing_export = exports.isna() | (exports <= 0)
    ex = np.log(exports.fillna(1.0))
    ex_norm = (ex - ex.min()) / (ex.max() - ex.min())
    rows = []
    for c in countries:
        value = available.get(c)
        rows.append(dict(country_iso3=c, year=2000, gdp_usd=value,
                         scale_normalized=(np.log(value) - low) / (high - low) if value else ex_norm[c],
                         source='WDI_GDP' if value else 'Atlas_export_proxy_NOT_GDP',
                         export_usd=exports.get(c, np.nan)))
    PROC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PROC / 'country_budget_inputs.csv', index=False)
    products = pd.read_parquet(PROC / 'product_metadata.parquet').product_hs92.tolist()
    pd.DataFrame({'product_hs92': products, 'resource_proxy': resource_product_mask(products)}).to_csv(
        PROC / 'resource_proxy.csv', index=False)
    manifest = dict(url=URL, indicator='NY.GDP.MKTP.CD', year=2000,
                    sha256=hashlib.sha256(body).hexdigest(), checked_at=datetime.now(timezone.utc).isoformat(),
                    n_gdp=len(available), fallback=[c for c in countries if c not in available],
                    fallback_rule='缺GDP时仅将基年出口对数规模映射为预算，不插补GDP',
                    resource_rule='HS25、26和2701至2709，仅为保守矿产代理')
    (ROOT / 'data/metadata/covariates_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
