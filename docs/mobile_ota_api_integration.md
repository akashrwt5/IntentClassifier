# Mobile OTA NLU Update - API Contract & Integration Guide

This document outlines the API structure and expected integration flow for the Mobile App to download and apply Over-The-Air (OTA) Natural Language Understanding (NLU) model bundles.

## Overview
The OTA backend acts as a Backend-For-Frontend (BFF) proxy. It securely parses our GitHub Releases and provides a clean, stable API for mobile clients to query for updates and download `.nlu` packs.

---

## 1. Check for Latest Update

**Endpoint:** `GET /api/v1/nlu/latest`

This endpoint checks if a newer NLU pack is available for the given language and platform, relative to the app's current version.

### Request Query Parameters
| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `lang` | `string` | No | `"en"` | The target language code (e.g., `en`, `fr`, `de`). |
| `platform` | `string` | No | `"universal"` | The target platform (`android`, `ios`, `universal`). |
| `app_version` | `string` | No | `"1.0.0"` | The semantic version of the current NLU pack or app on the device. Used to calculate `update_available`. |

### Example Request
```http
GET /api/v1/nlu/latest?lang=en&platform=universal&app_version=1.0.35
```

### Success Response (HTTP 200)
```json
{
  "update_available": true,
  "version": "1.0.36",
  "language": "en",
  "published_at": "2026-08-08T12:10:37Z",
  "download_url": "https://<server_domain>/api/v1/nlu/download?asset_id=506349416",
  "size_bytes": 2829387,
  "release_notes": "Single-language `.nlu` (spec/bundle/3.0)...",
  "sha256_hash": "hash-not-provided"
}
```

> [!NOTE]
> **About `sha256_hash`**: This may return `"hash-not-provided"`. The downloaded `.nlu` pack itself contains internal cryptographic signatures and `manifest.json` hashes. The mobile app should rely on verifying the internal signature after downloading the zip file rather than this external hash.

### Error Responses
- **400 Bad Request:** Language not supported.
- **404 Not Found:** No release pack exists for the requested language or platform on the backend.

---

## 2. Download the NLU Pack

**Endpoint:** `GET /api/v1/nlu/download`

This endpoint facilitates the secure download of the binary pack without exposing direct underlying repository URLs initially.

### Request Query Parameters
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `asset_id` | `string` | Yes | The ID of the asset to download, provided by the `/latest` endpoint. |

### Example Request
```http
GET /api/v1/nlu/download?asset_id=506349416
```

### Response (HTTP 302 Found)
The backend will respond with an **HTTP 302 Redirect**. 
- The mobile HTTP client (e.g., OkHttp, URLSession) MUST be configured to **follow redirects**.
- The redirection targets a signed AWS S3 / GitHub Object Storage URL where the binary `.nlu` zip file resides.
- The actual download payload will have a MIME type of `application/octet-stream` or `application/zip`.

---

## Mobile Implementation Workflow

The mobile team should implement the following flow (preferably on a background thread):

1. **Boot / Daily Check:**
   - On app launch (or scheduled daily task), hit the `/api/v1/nlu/latest` endpoint.
   - Pass the `app_version` representing the NLU pack currently active on the device.

2. **Evaluate Response:**
   - If `update_available == false`, terminate the flow. The device is up to date.
   - If `update_available == true`, inspect `size_bytes` to determine if a download should proceed over the current network (e.g., Cellular vs WiFi).

3. **Execute Download:**
   - Execute an HTTP GET against the provided `download_url`.
   - Ensure the HTTP client is configured to automatically follow HTTP 302 Redirects.
   - Save the incoming byte stream to a temporary file on the device storage.

4. **Verify and Deploy:**
   - Unzip/Extract the `.nlu` payload into a staging directory.
   - Read the inner cryptographic signature (`manifest.json` etc.) and verify file integrity.
   - If valid, atomically swap the staging directory with the active NLU directory in the app's data folder.
   - Reload the on-device inference engine.
