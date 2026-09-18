# Deploying the Digit Detection API for free

The service is one container: FastAPI + ONNX Runtime + OpenCV, no TensorFlow,
about 400 MB image, under 300 MB RAM, cold start under a second. It serves the
web page, the JSON API and the Swagger docs from one origin, so one domain is
all you need:

| path | what |
| --- | --- |
| `/` | upload page (calls the API from the same origin, no CORS needed) |
| `/docs`, `/openapi.json` | interactive API documentation |
| `POST /api/v1/detect` | JSON result; multipart field `image` |
| `POST /api/v1/detect/annotated` | annotated PNG; text in header `X-Detected-Text` |
| `GET /api/v1/info` | model, limits, default parameters |
| `GET /healthz` | probe used by the platforms |

Files the deployment needs: `Dockerfile`, `requirements-server.txt`,
`server.py`, `preprocess.py`, `predictor.py`, `detect.py`, `static/`,
`models/mnist_cnn.onnx` (741 KB). Everything else (`venv`, training code,
samples, tests) is excluded by `.dockerignore`.

## 0. Run it locally first

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install fastapi "uvicorn[standard]" python-multipart
python export_onnx.py                      # only if models\mnist_cnn.onnx does not exist yet
uvicorn server:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860 and drop an image, or:

```powershell
curl.exe -X POST "http://localhost:7860/api/v1/detect" -F "image=@samples\clean_00.png"
curl.exe -X POST "http://localhost:7860/api/v1/detect/annotated" -F "image=@samples\clean_00.png" -o out.png
python -m pytest tests\test_api.py -q
```

With Docker Desktop installed the exact production image can be built and run
locally too:

```powershell
docker build -t digit-api .
docker run --rm -p 7860:7860 digit-api
```

## 0b. What the GitHub repository already does for free

Repository: https://github.com/andrewbaysidemp-hash/digit-detection-api

- **CI** (`.github/workflows/ci.yml`): every push runs the 26 tests on Ubuntu.
- **Container image** (`.github/workflows/docker-publish.yml`): every push to
  `main` builds the Dockerfile, starts the container, runs a real detection
  against it, and publishes it to GitHub Container Registry. Anyone with Docker
  can run the API with one command, no build needed:

  ```
  docker run --rm -p 7860:7860 ghcr.io/andrewbaysidemp-hash/digit-detection-api:latest
  ```

- **One-click Render deploy**: the "Deploy to Render" button in the README
  (Render account required, free plan, no card, sign in with GitHub). This is
  the recommended free path; see section 3.
- **Automatic Hugging Face deploy** (`.github/workflows/deploy-hf-space.yml`):
  add the secret `HF_TOKEN` (a Hugging Face write token) and the variable
  `HF_SPACE` (for example `your-hf-user/digit-api`) in the repo settings, and
  every push to `main` creates/updates the Space; the live URL is
  `https://<your-hf-user>-digit-api.hf.space`. The same script works locally:

  ```powershell
  $env:HF_TOKEN = "hf_..."; $env:HF_SPACE = "your-hf-user/digit-api"
  python -m pip install huggingface_hub
  python deploy\huggingface\deploy.py
  ```

  Note (checked 2026-09-17): creating a Docker Space on a free Hugging Face
  account now fails with `402 Payment Required: hosting Gradio and Docker
  Spaces on free cpu-basic requires a PRO subscription`. Only static Spaces
  are free. Use this path only with a PRO account.

GitHub itself cannot run the container (GitHub Pages serves static files only),
which is why one of the hosts below is needed for a live URL. Free hosts that
still work without a card, in order of preference: Render (section 3), Koyeb
(section 3b). Google Cloud Run (section 2) is free within quota but needs a
billing account. Hugging Face Spaces (section 1) needs a PRO subscription.

## 1. Hugging Face Spaces (PRO subscription required for Docker Spaces)

Hardware on the basic tier: 2 vCPU, 16 GB RAM, public HTTPS URL
`https://<user>-<space>.hf.space`. A Space goes to sleep after 48 h without
traffic and wakes on the next request (about 30 s). Only `git` is needed.
Free accounts can no longer create Docker Spaces (see the note above).

1. Create an account at https://huggingface.co and a token with **write**
   access (Settings -> Access Tokens).
2. New Space (https://huggingface.co/new-space): name it e.g. `digit-api`,
   SDK **Docker**, template **Blank**, hardware **CPU basic (free)**, public.
3. In PowerShell, from the project folder:

   ```powershell
   git clone https://huggingface.co/spaces/<user>/digit-api hf-space
   robocopy . hf-space Dockerfile requirements-server.txt server.py preprocess.py predictor.py detect.py .dockerignore
   robocopy static hf-space\static /E
   robocopy models hf-space\models mnist_cnn.onnx
   copy deploy\huggingface\README.md hf-space\README.md
   cd hf-space
   git add .
   git commit -m "Digit detection API"
   git push            # user name = your HF user name, password = the write token
   ```

   The Space README must start with the YAML block in
   `deploy/huggingface/README.md` (`sdk: docker`, `app_port: 7860`); that is
   how Spaces knows how to run the container.
4. Watch the build in the Space's **Logs** tab (3-6 minutes the first time).
   When it shows `Uvicorn running on http://0.0.0.0:7860`, open
   `https://<user>-digit-api.hf.space/` and `/docs`.
5. Optional settings (Space -> Settings): add variable `WORKERS=2` to use both
   vCPUs, `API_KEY=<secret>` to require the `X-API-Key` header,
   `MAX_UPLOAD_MB=4` to tighten uploads. Changing a variable restarts the Space.

Custom domains are not available on Spaces. If you need your own domain, use
Cloud Run or Render below, or put a free Cloudflare Worker in front that
proxies `api.yourdomain.com` to the Space URL.

## 2. Google Cloud Run (autoscaling; free tier, but a billing account must exist)

Cloud Run scales from zero to N instances per request load, which is the
"scalable" option. The always-free tier covers 2 million requests, 360,000
GiB-seconds of memory and 180,000 vCPU-seconds per month; this service uses
about 0.2 vCPU-second per image, so a few hundred thousand images a month stay
free. You need a Google Cloud project with billing enabled (a card on file),
and the `gcloud` CLI (https://cloud.google.com/sdk/docs/install).

```powershell
gcloud auth login
gcloud config set project <your-project-id>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy digit-api --source . --region europe-west1 --allow-unauthenticated `
  --cpu 1 --memory 512Mi --concurrency 8 --min-instances 0 --max-instances 3 `
  --set-env-vars WORKERS=1,MAX_UPLOAD_MB=8
```

`--source .` uploads the folder (respecting `.dockerignore`), builds the
Dockerfile with Cloud Build and deploys. The command prints the service URL
(`https://digit-api-xxxx-ew.a.run.app`). Notes:

- `--max-instances 3` caps the bill at zero surprises; raise it when needed.
- `--concurrency 8` is right for a CPU-bound service on 1 vCPU.
- Own domain: `gcloud beta run domain-mappings create --service digit-api --domain api.yourdomain.com --region europe-west1`, then add the DNS records it prints. Certificates are automatic and free.
- To require a key: `--set-env-vars API_KEY=<secret>`; or remove
  `--allow-unauthenticated` and use Google IAM instead.

## 3. Render (recommended free option: no card, custom domain included)

Free instances have 512 MB RAM and 0.1 vCPU, sleep after 15 minutes of
inactivity and take 30-60 s to wake. Fine for demos and light use.

Option A, from the repository (builds the Dockerfile on Render):

1. Sign in at https://dashboard.render.com with GitHub (no card) and click the
   "Deploy to Render" button in the README, or New -> **Blueprint** -> select
   the repository. `render.yaml` defines the service (Docker runtime, free
   plan, health check `/healthz`).
2. After the build (5-8 minutes), the URL is
   `https://digit-detection-api.onrender.com` (Render adds a suffix if the
   name is taken). Settings -> Custom Domains lets you attach your own domain
   for free.

Option B, from the prebuilt image (no build at all): New -> **Web Service**
-> "Existing image" -> `ghcr.io/andrewbaysidemp-hash/digit-detection-api:latest`,
instance type Free, health check path `/healthz`. Render pulls the public
image and starts it in about a minute.

Option C, fully scripted with a Render API key (Account Settings -> API Keys):

```powershell
$env:RENDER_API_KEY = "rnd_..."
$owner = (Invoke-RestMethod -Headers @{Authorization="Bearer $env:RENDER_API_KEY"} https://api.render.com/v1/owners)[0].owner.id
$body = @{ type="web_service"; name="digit-detection-api"; ownerId=$owner;
           image=@{ imagePath="ghcr.io/andrewbaysidemp-hash/digit-detection-api:latest" };
           serviceDetails=@{ plan="free"; region="oregon"; healthCheckPath="/healthz"; runtime="image";
                             envSpecificDetails=@{}; envVars=@(@{key="WORKERS"; value="1"}) } } | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post -Headers @{Authorization="Bearer $env:RENDER_API_KEY"} -ContentType application/json -Body $body https://api.render.com/v1/services
```

## 3b. Koyeb (free "Hobby" web service, no card)

https://app.koyeb.com -> Create Web Service -> Docker -> image
`ghcr.io/andrewbaysidemp-hash/digit-detection-api:latest`, instance **Free**,
port 7860, health check `/healthz`. The URL is
`https://<app>-<org>.koyeb.app`. The free instance is small (0.1 vCPU,
512 MB) and may be paused when idle.

The repository's `.gitignore` keeps `venv/` and `samples/` out and commits the
trained models (3.7 MB), so Render, Cloud Run and Spaces all build from the
repository as is.

## 4. Using the API

```bash
# JSON
curl -X POST "https://<host>/api/v1/detect?min_conf=0.5" -F "image=@digits.jpg"
# JSON + annotated image (base64 PNG in "annotated_png_base64")
curl -X POST "https://<host>/api/v1/detect?annotate=true" -F "image=@digits.jpg"
# annotated PNG directly
curl -X POST "https://<host>/api/v1/detect/annotated" -F "image=@digits.jpg" -o annotated.png
# with an API key
curl -H "X-API-Key: <secret>" -X POST "https://<host>/api/v1/detect" -F "image=@digits.jpg"
```

Python:

```python
import requests
with open("digits.jpg", "rb") as f:
    r = requests.post("https://<host>/api/v1/detect", files={"image": f}, params={"min_conf": 0.5})
r.raise_for_status()
print(r.json()["text"])
```

JavaScript (browser or Node 18+):

```js
const fd = new FormData();
fd.append("image", fileInput.files[0]);
const r = await fetch("https://<host>/api/v1/detect", { method: "POST", body: fd });
const { text, lines } = await r.json();
```

Query parameters on both detect endpoints: `min_conf` (0-1, default 0.5),
`annotate` (JSON endpoint only), `max_side` (200-4000), `otsu`, `keep_lines`,
`binary_crops`, `target_stroke` (0-0.4). Errors: 400 undecodable/empty image,
401 missing or wrong API key, 413 upload larger than `MAX_UPLOAD_MB`, 422
invalid parameter. An image with no digits is a normal 200 with
`digit_count: 0`.

## 5. Capacity and scaling notes

- Per request the work is ~20-200 ms of OpenCV segmentation (depends on image
  size, capped by `max_side`) plus ~0.3 ms per digit of ONNX inference.
  Measured on the development machine with one uvicorn worker: 28 ms round
  trip per 125 KB PNG of a 5-digit number, i.e. about 35 requests per second
  per worker; large phone photos take 100-200 ms each.
- The service is stateless: no files are written, nothing is kept between
  requests, so any number of instances can run behind one URL. On Cloud Run
  that is automatic; on a VM run `uvicorn --workers <vCPUs>` behind nginx or
  Caddy, or several containers behind a load balancer.
- `OMP_NUM_THREADS=1` in the Dockerfile stops ONNX Runtime and OpenCV from
  oversubscribing threads when several workers share the CPU.
- Memory per worker is about 250 MB; the free tiers above fit 1-2 workers.
- Cold starts: the container starts in under a second, so scale-to-zero
  platforms feel fine. Sleeping free tiers (Spaces after 48 h, Render after
  15 min) add their own wake-up delay.
- Protect a public endpoint: set `API_KEY`, keep `MAX_UPLOAD_MB` small, set a
  `--max-instances` cap on Cloud Run, and consider a Cloudflare free plan in
  front for rate limiting and your own domain.
