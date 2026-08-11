import os
import re
import time
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from packaging.version import Version, InvalidVersion

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ota_server")

app = FastAPI(
    title="Development OTA NLU Server",
    description="A Backend-for-Frontend (BFF) proxy to serve NLU bundles from GitHub Releases securely.",
    version="1.0.0"
)

# Configuration
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "akashrwt5")
GITHUB_REPO = os.getenv("GITHUB_REPO", "IntentClassifier")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes cache
SUPPORTED_LANGUAGES = {"en", "fr", "de", "es", "it"}

# Regex for parsing asset names like "pack-en-v1.0.36-universal.nlu"
# Captures: lang, version, platform, extension
PACK_REGEX = re.compile(r"^pack-(.+)-v(.+?)-(.+)\.(zip|nlu)$")

# Global HTTPX Client
http_client: httpx.AsyncClient = None

# Thread-safe (async-safe) In-Memory Cache
_cache_lock = asyncio.Lock()
_release_cache = {
    "timestamp": 0,
    "packs": {},           # lang -> platform -> { version, size, download_url, sha256_hash, asset_id }
    "asset_lookup": {},    # asset_id -> download_url (for O(1) downloads)
    "release_notes": "",
    "published_at": ""
}

class NluUpdateResponse(BaseModel):
    update_available: bool
    version: str
    language: str
    published_at: str
    download_url: str
    size_bytes: int
    release_notes: str
    sha256_hash: str

async def _fetch_sha256(url: str, lang: str) -> tuple:
    """Downloads and parses the .sha256 file from GitHub concurrently."""
    try:
        response = await http_client.get(url)
        if response.status_code == 200:
            return lang, response.text.split()[0].strip()
    except Exception as e:
        logger.warning(f"Failed to fetch SHA256 from {url}: {e}")
    return lang, "hash-fetch-failed"

async def refresh_github_release_cache_locked():
    """Fetches the latest release from GitHub API, normalizes the data, and updates cache."""
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "OTA-NLU-Server/1.0"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        
    response = await http_client.get(url, headers=headers)
        
    if response.status_code in (403, 429):
        raise HTTPException(status_code=429, detail="GitHub API rate limit exceeded. Try again later.")
    elif response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to fetch upstream release: {response.status_code}")
        
    data = response.json()
    assets = data.get("assets", [])
    
    # Normalizing Cache Structure O(1)
    packs = {}
    asset_lookup = {}
    
    # First pass: map all assets by name for easy sha256 correlation
    asset_map_by_name = {a["name"]: a for a in assets}
    
    sha_fetch_tasks = []
    temp_pack_info = {}
    
    for a in assets:
        match = PACK_REGEX.match(a["name"])
        if match:
            lang, version, platform, ext = match.groups()
            if lang in SUPPORTED_LANGUAGES:
                if lang not in temp_pack_info:
                    temp_pack_info[lang] = {}
                    
                target_asset = a
                temp_pack_info[lang][platform] = {
                    "version": version,
                    "asset_id": str(target_asset["id"]),
                    "size": target_asset["size"],
                    "download_url": target_asset["browser_download_url"]
                }
                
                # Setup async task for fetching SHA256 if available
                sha_asset = asset_map_by_name.get(f"{target_asset['name']}.sha256")
                if sha_asset:
                    sha_fetch_tasks.append(_fetch_sha256(sha_asset["browser_download_url"], f"{lang}_{platform}"))

    # Fetch all SHAs in parallel
    sha_results = await asyncio.gather(*sha_fetch_tasks)
    sha_map = {key: hash_val for key, hash_val in sha_results}

    for lang, platforms in temp_pack_info.items():
        packs[lang] = {}
        for platform, info in platforms.items():
            info["sha256_hash"] = sha_map.get(f"{lang}_{platform}", "")
            packs[lang][platform] = info
            asset_lookup[info["asset_id"]] = info["download_url"]

    _release_cache["timestamp"] = time.time()
    _release_cache["packs"] = packs
    _release_cache["asset_lookup"] = asset_lookup
    _release_cache["release_notes"] = data.get("body", "")[:2048]
    _release_cache["published_at"] = data.get("published_at", datetime.utcnow().isoformat() + "Z")

async def get_cached_release():
    """Returns cached release or fetches it securely avoiding Cache Stampedes."""
    current_time = time.time()
    
    if (current_time - _release_cache["timestamp"]) <= CACHE_TTL_SECONDS and _release_cache["packs"]:
        return _release_cache

    async with _cache_lock:
        current_time = time.time()
        is_stale = (current_time - _release_cache["timestamp"]) > CACHE_TTL_SECONDS
        has_data = bool(_release_cache["packs"])

        if is_stale or not has_data:
            try:
                await refresh_github_release_cache_locked()
            except HTTPException as e:
                if has_data:
                    logger.warning(f"Cache refresh failed ({e.detail}), serving stale cache.")
                else:
                    raise e
                    
        return _release_cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=10.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
    )
    logger.info("Warming up cache on startup...")
    try:
        await get_cached_release()
        logger.info("Cache warmed successfully.")
    except Exception as e:
        logger.warning(f"Initial cache warm failed: {e}")
        
    yield
    await http_client.aclose()

app.router.lifespan_context = lifespan

@app.get("/api/v1/nlu/latest", response_model=NluUpdateResponse)
async def get_latest_nlu(request: Request, lang: str = "en", app_version: str = "1.0.0", platform: str = "universal"):
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Language '{lang}' not supported.")
        
    cache = await get_cached_release()
    lang_packs = cache["packs"].get(lang)
    
    if not lang_packs:
        raise HTTPException(status_code=404, detail=f"No language pack found for '{lang}'")
        
    pack = lang_packs.get(platform)
    if not pack:
        raise HTTPException(status_code=404, detail=f"No platform '{platform}' found for language '{lang}'")

    try:
        update_available = Version(pack["version"]) > Version(app_version)
    except InvalidVersion:
        update_available = True
        
    base_url = str(request.base_url).rstrip("/")
    proxy_download_url = f"{base_url}/api/v1/nlu/download?asset_id={pack['asset_id']}"

    return NluUpdateResponse(
        update_available=update_available,
        version=pack["version"],
        language=lang,
        published_at=cache["published_at"],
        download_url=proxy_download_url,
        size_bytes=pack["size"],
        release_notes=cache["release_notes"],
        sha256_hash=pack["sha256_hash"] or "hash-not-provided"
    )

@app.get("/api/v1/nlu/download")
async def download_nlu(asset_id: str):
    cache = await get_cached_release()
    
    download_url = cache["asset_lookup"].get(str(asset_id))
            
    if not download_url:
        raise HTTPException(status_code=404, detail="Asset not found")
        
    logger.info(f"Redirecting to secure asset: {download_url}")
    return RedirectResponse(url=download_url, status_code=302)

if __name__ == "__main__":
    logger.info("==================================================")
    logger.info("Starting Production-Grade Dev OTA Server")
    logger.info(f"Targeting Repo: {GITHUB_OWNER}/{GITHUB_REPO}")
    logger.info("==================================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)
