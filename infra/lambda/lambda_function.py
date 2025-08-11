import json
import os
import urllib.request
import urllib.error
import boto3

secrets = boto3.client("secretsmanager")

OWNER = os.environ["GITHUB_OWNER"]
REPO = os.environ["GITHUB_REPO"]
WF_ID = os.environ["GITHUB_WORKFLOW"]  # e.g., 'retrain.yaml' or '1234567'
REF = os.environ.get("GITHUB_REF", "main")
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
        key = s3.get("object", {}).get("key")
        etag = s3.get("object", {}).get("eTag")

        inputs = {"s3_bucket": bucket or "", "s3_key": key or "", "s3_etag": etag or ""}
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
