# MLOps Pipeline: DVC + MLflow + GitHub Actions on AWS (FastAPI)

> **Goal:** Automatically retrain and safely deploy a FastAPI ML service whenever **code** or **data** changes — with reproducible data/model versions, a proper model registry, and production observability.

---

## 🚀 Architecture (One-Page)
![Architecture](./docs/architecture.png)
> Put your diagram at `docs/architecture.png`. (You can export from draw.io as PNG.)

**Key flow**  
- **Code path:** Developer pushes → GitHub Actions runs CI (lint/tests) → builds Docker image → pushes to Docker Hub → deploys/refreshes FastAPI on EC2.  
- **Data path:** Data scientist updates CSV in S3 → S3 **PUT** triggers **EventBridge → Lambda** → dispatches **`retrain.yml`** workflow → GitHub runner executes **DVC** (`fetch → status → pull → repro → push`) → training code logs to **MLflow** → best run is **registered** in **Model Registry** → (optional) promote to **Production** → FastAPI reloads/uses production model.  
- **Observability:** FastAPI exposes `/metrics` → **Prometheus** scrapes → **Grafana** dashboards.

**Roles**  
- **DVC**: data/pipeline versioning + retraining orchestration.  
- **MLflow**: experiment tracking + model registry (not the trainer).  
- **CI/CD**: builds images, dispatches retrain, deploys app.  
- **AWS**: S3 (data + DVC remote), EventBridge/Lambda (triggers), EC2 (serving), Secrets Manager (config).

---

## 🧰 Tech Stack
- **Training/Serving:** Python, scikit-learn, FastAPI, Uvicorn
- **Tracking/Registry:** MLflow Tracking Server + Model Registry
- **Data/Pipeline:** DVC with **S3** remote
- **CI/CD:** GitHub Actions, Docker, Docker Hub
- **Infra:** AWS S3, EventBridge, Lambda, EC2 (in VPC), Secrets Manager
- **Observability:** Prometheus, Grafana

---

## 📁 Suggested Repo Layout
```
.
├─ app/                       # FastAPI app (serving)
│  └─ main.py
│  └─ metrics.py
│  └─ schemas.py

├─ src/                       # Training code
│  ├─ train.py
│  └─ data_preprocessing.py
│  └─ inference.py

├─ dvc.yaml                   # DVC pipeline stages
├─ requirements.txt
├─ Dockerfile
├─ Makefile                   # optional: handy commands
├─ .github/workflows/
│  ├─ ci-cd.yml
│  └─ retrain.yml
├─ logs/                      # for storing prediction logs
├─ monitoring/
│  └─ docker-compose.monitoring.yaml
│  └─ prometheus.yaml
├─ docs/
│  └─ architecture.png
└─ README.md
```

---

## 🔐 Secrets & Environment
Use **GitHub Actions → Settings → Secrets and variables → Actions**

**Required (typical):**
- `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`
- `AWS_REGION`
- If runners **don’t** have an AWS role attached: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `S3_DATA_BUCKET` (raw CSV location), `S3_DVC_BUCKET` (DVC remote)
- `MLFLOW_TRACKING_URI` (e.g., `http://mlflow.yourdomain:5000`)
- `MLFLOW_S3_BUCKET` (if MLflow artifacts in S3), optionally `MLFLOW_S3_ENDPOINT_URL`
- `GH_PAT` (if Lambda dispatches workflow via GitHub API; scope: `repo`, `workflow`)

On **EC2** (serving): place app secrets (DB URIs, registry/model name, etc.) in **AWS Secrets Manager** or instance profile/IAM role.

---

## 🗃️ DVC Setup (one-time)
```bash
# init (if not already)
dvc init

# track dataset
dvc add data/housing.csv
git add data/housing.csv.dvc .gitignore
git commit -m "Track dataset with DVC"

# add S3 remote
dvc remote add -d s3remote s3://$S3_DVC_BUCKET
dvc push                       # upload artifacts to S3

# commit DVC config so runners can pull
git add .dvc/config
git commit -m "Configure DVC S3 remote"
```

**Retrain steps run by CI:**  
```bash
dvc fetch && dvc status -r s3remote
dvc pull
dvc repro
dvc push
```

---

## 🧪 MLflow Setup
**Run server as systemd service (example):**
```bash
mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://$MLFLOW_S3_BUCKET/
```

**Training code (snippet):**
refer: src/train.py
Promotion to **Staging/Production** can be manual (UI) or automated (CI step).

---

## 🧱 DVC Pipeline (`dvc.yaml` example)
```yaml
stages:
  fetch_housing:
    cmd: aws s3 cp s3://dvc-data-src/housing.csv data/housing.csv
    deps:
      - s3://dvc-data-src/housing.csv
    outs:
      - data/housing.csv
  train:
    cmd: python3.11 src/train.py
    deps:
      - src/train.py
      - src/data_preprocessing.py
      - src/inference.py
      - s3://dvc-data-src/housing.csv
```
> Your `train.py` should log to MLflow and register the best model.

---

## 🧰 GitHub Actions

**`.github/workflows/ci-cd.yml` (code path)**
**`.github/workflows/retrain.yml` (data path)**
```yaml
name: Retrain on Data Update
on:
  workflow_dispatch:
  repository_dispatch:
    types: [s3-data-updated]

jobs:
  retrain:
    runs-on: ubuntu-latest
    env:
      AWS_REGION: ${{ secrets.AWS_REGION }}
      S3_DVC_BUCKET: ${{ secrets.S3_DVC_BUCKET }}
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -r requirements.txt
      - name: Configure AWS creds (if no OIDC/role)
        if: ${{ secrets.AWS_ACCESS_KEY_ID != '' }}
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}
      - name: Retrain with DVC
        run: |
          dvc pull
          dvc repro
          dvc push
```

**Lambda → GitHub dispatch (pseudo)**
```python
import json, os, urllib.request, urllib.error
import boto3

secrets = boto3.client("secretsmanager")

OWNER   = os.environ["GITHUB_OWNER"]
REPO    = os.environ["GITHUB_REPO"]
WF_ID   = os.environ["GITHUB_WORKFLOW"]   # e.g., 'retrain.yaml' or '1234567'
REF     = os.environ.get("GITHUB_REF","main")
PAT_ARN = os.environ["GITHUB_PAT_SECRET_ARN"]
S3_KEY_FILTER = "housing.csv"

def _github_pat():
    val = secrets.get_secret_value(SecretId=PAT_ARN)
    s = val.get("SecretString") or ""
    return s.strip()

def _trigger_github_dispatch(token, inputs):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{WF_ID}/dispatches"
    body = json.dumps({"ref": REF, "inputs": inputs}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(req) as resp:
        return resp.status

def handler(event, context):
    # S3 can batch multiple records
    token = _github_pat()
    triggered = 0
    for rec in event.get("Records", []):
        if rec.get("eventSource") != "aws:s3":
            continue
        s3 = rec.get("s3", {})
        bucket = s3.get("bucket", {}).get("name")
        key    = s3.get("object", {}).get("key")
        etag   = s3.get("object", {}).get("eTag")


        inputs = {
            "s3_bucket": bucket or "",
            "s3_key": key or "",
            "s3_etag": etag or ""
        }
        try:
            status = _trigger_github_dispatch(token, inputs)
            triggered += 1
            print(f"Dispatched workflow {WF_ID} for {key} (HTTP {status})")
        except urllib.error.HTTPError as e:
            print(f"GitHub API error {e.code}: {e.read().decode()}")
            raise
        except Exception as e:
            print(f"Error dispatching workflow: {e}")
            raise

    return {"dispatched": triggered}
```

---

## 🚢 Deploying FastAPI to EC2 via ci-cd.yaml
```bash
docker run -d --name fastapi \
  -p 80:8000 \
  -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
  -e MODEL_NAME=California_Housing_Regression \
  -e MODEL_STAGE=Production \
  $DOCKERHUB_USERNAME/fastapi-be:${GIT_SHA}
```
- App loads the **Production** model from **MLflow Registry** at startup.  
- Expose `/metrics` for Prometheus.

---

## 📊 Monitoring
- **Prometheus** scrapes `http://ec2-host:9090/metrics` (request count, latency, error rate; add model drift metrics if logged).
- **Grafana** dashboards visualize SLOs and model KPIs and its endpoint is as `http://ec2-host:3000/dashboards`.
---
---

