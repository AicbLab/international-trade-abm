"""下载研究计划中的公开原始数据，校验官方 MD5 并保存来源清单。"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
MANIFEST = META / "download_manifest.json"
DATAVERSE = "https://dataverse.harvard.edu"
HS92_DOI = "doi:10.7910/DVN/T4CHWJ"
SELECTED_HS92 = (
    "hs92_data_dictionary.csv",
    "hs92_country_year.csv",
    "hs92_product_year_4.csv",
    "hs92_country_country_year.csv",
    "hs92_country_product_year_4.csv",
)
REFERENCE_FILES = (
    ("Reporters.json", "https://comtradeapi.un.org/files/v1/app/reference/Reporters.json"),
    ("H0.json", "https://comtradeapi.un.org/files/v1/app/reference/H0.json"),
)
MAX_FILE_BYTES = 600_000_000
HEADERS = {"User-Agent": "EconomicComplexityResearch/1.0 (public academic data download)"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fetch_json(url, destination):
    """只获取公开 JSON，不需要 API Key 或账户。"""
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read(10_000_001)
    if len(content) > 10_000_000:
        raise ValueError("元数据超出大小限制")
    data = json.loads(content)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def checksums(path):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def download(item):
    """有限重试；只有大小和校验和均匹配的文件才标为完成。"""
    path = ROOT / item["path"]
    expected_size = item["expected_bytes"]
    expected_md5 = item["expected_md5"]
    if expected_size > MAX_FILE_BYTES:
        raise ValueError(f"文件超过自动下载上限：{expected_size} 字节")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = checksums(path)
        if path.stat().st_size == expected_size and digest["md5"] == expected_md5:
            print(f"已校验，跳过：{path.name}", flush=True)
            return {**item, **digest, "bytes": expected_size, "status": "verified", "checked_at": utc_now()}
        raise ValueError(f"现有文件不匹配，保留且不覆盖：{path}")

    partial = path.with_name(path.name + ".part")
    last_error = None
    for attempt in range(1, 4):
        try:
            print(f"下载 {path.name}，{expected_size / 1e6:.2f} MB，第 {attempt} 次", flush=True)
            request = urllib.request.Request(item["url"], headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type.lower():
                    raise ValueError("收到网页而不是数据文件")
                md5 = hashlib.md5()
                sha256 = hashlib.sha256()
                size = 0
                last_progress = time.monotonic()
                with partial.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > expected_size:
                            raise ValueError("下载超过官方声明大小")
                        output.write(chunk)
                        md5.update(chunk)
                        sha256.update(chunk)
                        if time.monotonic() - last_progress >= 15:
                            print(f"  {path.name}: {size / 1e6:.1f}/{expected_size / 1e6:.1f} MB", flush=True)
                            last_progress = time.monotonic()
            if size != expected_size or md5.hexdigest() != expected_md5:
                raise ValueError(f"校验失败：大小 {size}/{expected_size}，MD5 {md5.hexdigest()}")
            partial.rename(path)
            print(f"校验通过：{path.name}", flush=True)
            return {**item, "bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest(),
                    "content_type": content_type, "status": "verified", "downloaded_at": utc_now()}
        except (OSError, ValueError) as error:
            last_error = error
            print(f"下载失败：{type(error).__name__}: {error}", flush=True)
            if attempt < 3:
                time.sleep(5 * attempt)
    raise RuntimeError(str(last_error))


def download_reference(name, url):
    """保存 UN 公开代码表原始字节；本地校验和不冒充官方校验和。"""
    path = ROOT / "data" / "raw" / "un_comtrade_reference" / name
    if path.exists():
        content = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read(10_000_001)
        if len(content) > 10_000_000:
            raise ValueError("代码表超出大小限制")
        data = json.loads(content)
        if not isinstance(data.get("results"), list) or not data["results"]:
            raise ValueError("代码表格式不符或为空")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    data = json.loads(content)
    print(f"代码表有效：{name}，{len(data['results'])} 条", flush=True)
    return {"source": "UN Comtrade public reference", "url": url,
            "path": path.relative_to(ROOT).as_posix(), "bytes": len(content),
            **checksums(path), "status": "verified_local_json", "checked_at": utc_now(),
            "official_checksum_available": False, "records": len(data["results"])}


def save_manifest(manifest):
    META.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("all", "atlas", "figshare", "references"), default="all")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {
        "requested_period": [2000, 2023],
        "primary_classification": "HS 1992, 4-digit",
        "created_at": utc_now(),
        "files": [],
    }
    jobs = []
    if args.source in ("all", "atlas"):
        url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={HS92_DOI}"
        meta = fetch_json(url, META / "harvard_hs92_dataset.json")
        version = meta["data"]["latestVersion"]
        available = {entry["dataFile"]["filename"]: entry for entry in version["files"]}
        for name in SELECTED_HS92:
            entry = available[name]
            if entry.get("restricted"):
                raise ValueError(f"文件需要授权，不自动下载：{name}")
            file = entry["dataFile"]
            if file["checksum"]["type"] != "MD5":
                raise ValueError("官方校验和类型已变更，请重新核对")
            jobs.append({
                "source": "Harvard Growth Lab / Harvard Dataverse",
                "dataset_doi": HS92_DOI,
                "dataset_version": f"{version['versionNumber']}.{version['versionMinorNumber']}",
                "license": version.get("license", {}),
                "path": f"data/raw/harvard_atlas/{name}",
                "url": f"{DATAVERSE}/api/access/datafile/{file['id']}",
                "expected_bytes": file["filesize"],
                "expected_md5": file["checksum"]["value"],
            })
    if args.source in ("all", "figshare"):
        meta = fetch_json("https://api.figshare.com/v2/articles/20167700", META / "figshare_integrated_dataset.json")
        if meta.get("download_disabled") or not meta.get("is_public"):
            raise ValueError("Figshare 未开放下载")
        for file in meta["files"]:
            name = Path(file["name"]).name
            jobs.append({
                "source": "Figshare / Patelli (2022)",
                "dataset_doi": meta["doi"],
                "license": meta["license"],
                "path": f"data/raw/figshare_integrated/{name}",
                "url": file["download_url"],
                "expected_bytes": file["size"],
                "expected_md5": file["computed_md5"],
                "role": "supplement_only; HS2 goods + services, 1996-2018; not HS4 ECI/PCI",
            })
    failures = []
    for item in jobs:
        try:
            result = download(item)
        except Exception as error:
            result = {**item, "status": "failed", "error": str(error), "checked_at": utc_now()}
            failures.append(item["path"])
        manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != item["path"]]
        manifest["files"].append(result)
        manifest["updated_at"] = utc_now()
        save_manifest(manifest)
    reference_count = 0
    if args.source in ("all", "references"):
        for name, url in REFERENCE_FILES:
            reference_count += 1
            try:
                result = download_reference(name, url)
            except Exception as error:
                result = {"path": f"data/raw/un_comtrade_reference/{name}", "url": url,
                          "status": "failed", "error": str(error), "checked_at": utc_now()}
                failures.append(result["path"])
            manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != result["path"]]
            manifest["files"].append(result)
            manifest["updated_at"] = utc_now()
            save_manifest(manifest)
    print(json.dumps({"completed": len(jobs) + reference_count - len(failures), "failed": failures}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
