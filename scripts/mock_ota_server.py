import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from datetime import datetime

app = FastAPI(title="Mock OTA NLU Server")

# Hardcoded for local mobile testing
LATEST_VERSION = "1.0.35"
# Replace this with any actual GitHub release zip URL you want to test downloading
GITHUB_MOCK_ZIP_URL = "https://github.com/YOUR_ORG/IntentClassifier/releases/download/pack-en-v1.0.35/pack-en-v1.0.35.zip"

@app.get("/api/v1/nlu/latest")
def get_latest_nlu(lang: str = "en"):
    """
    Mock endpoint for the mobile app to check if an update exists.
    """
    return {
        "version": LATEST_VERSION,
        "language": lang,
        "published_at": datetime.utcnow().isoformat() + "Z",
        "download_url": f"http://127.0.0.1:8000/api/v1/nlu/download?version={LATEST_VERSION}&lang={lang}",
        "size_bytes": 9500000,
        "release_notes": "Mock release for mobile development."
    }

@app.get("/api/v1/nlu/download")
def download_nlu(version: str, lang: str = "en"):
    """
    Mock endpoint that redirects the mobile app to the actual zip file.
    """
    print(f"Mobile app requested download for v{version} ({lang})")
    # In a real server, this would dynamically generate a signed S3 URL or fetch the GitHub asset URL.
    # Here, we just 302 redirect to the hardcoded GitHub URL.
    return RedirectResponse(url=GITHUB_MOCK_ZIP_URL, status_code=302)

if __name__ == "__main__":
    print("Starting Mock OTA Server on http://127.0.0.1:8000")
    print("Mobile App should call: GET http://127.0.0.1:8000/api/v1/nlu/latest")
    uvicorn.run(app, host="0.0.0.0", port=8000)
