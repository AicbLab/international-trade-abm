"""流式检查下载数据，输出覆盖范围；不清洗、不插补、不修改原始数据。"""

import csv
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from download_data import META, ROOT, MANIFEST, SELECTED_HS92, REFERENCE_FILES, checksums, utc_now

REQUIRED_YEARS = set(range(2000, 2024))
NUMERIC_FIELDS = {"export_value", "import_value", "eci", "pci", "export_rca"}


def profile_csv(path):
    counts = Counter()
    years = Counter()
    countries = set()
    products = set()
    missing = Counter()
    negative = Counter()
    invalid = Counter()
    coverage = Counter()
    examples = defaultdict(list)

    def record_example(issue, row):
        if len(examples[issue]) < 5:
            examples[issue].append(dict(row))

    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"无表头或重复表头：{path.name}")
        metrics = NUMERIC_FIELDS.intersection(fields)
        for row in reader:
            counts["rows"] += 1
            if None in row or any(value is None for value in row.values()):
                counts["malformed_rows"] += 1
                continue
            year = None
            if "year" in fields:
                try:
                    year = int(row["year"])
                    years[year] += 1
                except ValueError:
                    counts["invalid_year"] += 1
            country = row.get("country_iso3_code", "")
            if "country_iso3_code" in fields:
                if country:
                    countries.add(country)
                else:
                    counts["missing_country"] += 1
            product = row.get("product_hs92_code", "")
            if "product_hs92_code" in fields:
                if product:
                    products.add(product)
                    if not (len(product) == 4 and product.isdigit()):
                        counts["non_hs4_code"] += 1
                        record_example("non_hs4_code", row)
                        if year in REQUIRED_YEARS:
                            counts["non_hs4_code_2000_2023"] += 1
                else:
                    counts["missing_product"] += 1
            if year in REQUIRED_YEARS:
                counts["rows_2000_2023"] += 1
                if country:
                    coverage[(country, year)] += 1
            for field in metrics:
                value = row[field]
                if value.strip() == "":
                    missing[field] += 1
                    continue
                try:
                    number = float(value)
                    if not math.isfinite(number):
                        invalid[field] += 1
                    elif field in ("export_value", "import_value", "export_rca") and number < 0:
                        negative[field] += 1
                        record_example(f"negative_{field}", row)
                except ValueError:
                    invalid[field] += 1
            if counts["rows"] % 1_000_000 == 0:
                print(f"检查 {path.name}: {counts['rows']:,} 行", flush=True)
    result = {
        "columns": fields, **counts,
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "year_counts": dict(sorted(years.items())),
        "missing_requested_years": sorted(REQUIRED_YEARS.difference(years)) if "year" in fields else [],
        "countries": len(countries), "country_iso3_codes": sorted(countries),
        "products": len(products), "product_codes": sorted(products),
        "numeric_hs4_products": sum(len(code) == 4 and code.isdigit() for code in products),
        "missing_numeric": dict(missing), "invalid_numeric": dict(invalid),
        "negative_nonnegative_fields": dict(negative),
        "anomaly_examples": dict(examples),
        "coverage_scope": "文件存在的记录；未出现的记录不自动解释为零贸易",
        "duplicate_keys_checked": False,
    }
    if coverage:
        destination = META / f"{path.stem}_coverage_2000_2023.csv"
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["country_iso3_code", "year", "rows"])
            writer.writerows((country, year, n) for (country, year), n in sorted(coverage.items()))
        country_years = defaultdict(set)
        for country, year in coverage:
            country_years[country].add(year)
        result["countries_with_all_requested_years"] = sum(y == REQUIRED_YEARS for y in country_years.values())
        result["country_missing_years"] = {
            country: sorted(REQUIRED_YEARS - observed)
            for country, observed in sorted(country_years.items()) if observed != REQUIRED_YEARS
        }
    return result


def create_code_lookups():
    """用官方描述匹配 Atlas 已观察到的代码，未匹配项明确留空。"""
    reference = ROOT / "data/raw/un_comtrade_reference"
    atlas = ROOT / "data/raw/harvard_atlas"
    if not all((reference / name).exists() for name in ("H0.json", "Reporters.json")):
        return {"status": "reference_files_missing"}
    hs = json.loads((reference / "H0.json").read_text(encoding="utf-8-sig"))
    descriptions = {str(row["id"]): row["text"] for row in hs["results"] if row.get("aggrlevel") == 4}
    reporters = json.loads((reference / "Reporters.json").read_text(encoding="utf-8-sig"))
    names = {row["reporterCodeIsoAlpha3"]: row["reporterDesc"] for row in reporters["results"]}
    products = set()
    countries = set()
    with (atlas / "hs92_product_year_4.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            products.add((row["product_id"], row["product_hs92_code"]))
    with (atlas / "hs92_country_year.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            countries.add((row["country_id"], row["country_iso3_code"]))
    for filename, header, records, lookup in (
        ("hs92_hs4_product_lookup.csv", ["atlas_product_id", "product_hs92_code", "un_description"], products, descriptions),
        ("atlas_country_lookup.csv", ["atlas_country_id", "country_iso3_code", "un_reporter_name"], countries, names),
    ):
        with (META / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows((identifier, code, lookup.get(code, "")) for identifier, code in sorted(records, key=lambda pair: pair[1]))
    return {"products": len(products), "countries": len(countries),
            "unmatched_products": sorted(code for _, code in products if code not in descriptions),
            "unmatched_countries": sorted(code for _, code in countries if code not in names),
            "note": "本地派生映射表。Atlas 的扩展国家代码与 UN 代码不一定一一对应；只按原始字符串匹配。"}


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = {"checked_at": utc_now(), "files": [], "errors": []}
    required = {f"data/raw/harvard_atlas/{name}" for name in SELECTED_HS92}
    required.update(f"data/raw/un_comtrade_reference/{name}" for name, _ in REFERENCE_FILES)
    required.add("data/raw/figshare_integrated/universal_database.zip")
    for missing_path in sorted(required - {item["path"] for item in manifest["files"]}):
        report["errors"].append(f"下载清单缺少预期文件：{missing_path}")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        entry = {"path": item["path"]}
        if item["status"] not in ("verified", "verified_local_json") or not path.exists():
            report["errors"].append(f"未完成：{item['path']}")
            continue
        digest = checksums(path)
        if path.stat().st_size != item["bytes"] or digest["sha256"] != item["sha256"]:
            report["errors"].append(f"本地文件大小或 SHA256 不匹配：{item['path']}")
            continue
        if item.get("expected_md5") and digest["md5"] != item["expected_md5"]:
            report["errors"].append(f"官方 MD5 不匹配：{item['path']}")
            continue
        entry.update({"bytes": path.stat().st_size, "checksum_passed": True})
        if path.suffix == ".csv":
            entry["profile"] = profile_csv(path)
            profile = entry["profile"]
            if any(profile.get(key) for key in ("malformed_rows", "invalid_year", "missing_country", "missing_product", "non_hs4_code", "invalid_numeric", "negative_nonnegative_fields", "missing_requested_years")):
                report["errors"].append(f"字段或覆盖检查异常：{item['path']}")
            print(f"检查完成：{path.name}，{profile['rows']:,} 行，{profile['year_min']}—{profile['year_max']}", flush=True)
        elif path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                bad_file = archive.testzip()
                if bad_file:
                    report["errors"].append(f"压缩包 CRC 失败：{bad_file}")
                entry["zip_crc_passed"] = bad_file is None
                entry["members"] = [{"name": info.filename, "bytes": info.file_size} for info in archive.infolist()]
                print(f"压缩包校验：{path.name}，{len(entry['members'])} 个条目", flush=True)
        elif path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            entry["records"] = len(value.get("results", []))
        report["files"].append(entry)
    report["lookups"] = create_code_lookups()
    report["total_bytes"] = sum(entry["bytes"] for entry in report["files"])
    report["status"] = "passed" if not report["errors"] else "needs_attention"
    (META / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = []
    for entry in report["files"]:
        profile = entry.get("profile", {})
        fields = ("rows", "rows_2000_2023", "year_min", "year_max", "countries", "products", "numeric_hs4_products", "countries_with_all_requested_years", "country_missing_years", "missing_numeric", "negative_nonnegative_fields", "non_hs4_code", "non_hs4_code_2000_2023", "anomaly_examples")
        summary.append({"path": entry["path"], "bytes": entry["bytes"], **{key: profile[key] for key in fields if key in profile}})
    (META / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "total_bytes": report["total_bytes"], "lookups": report["lookups"], "errors": report["errors"]}, ensure_ascii=False), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
