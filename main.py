from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
import boto3
import json

app = FastAPI()

# AWS Bedrock config
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
inference_profile_arn = "arn:aws:bedrock:ap-south-1:069717477936:inference-profile/apac.amazon.nova-micro-v1:0"

# ==== Input Schema ====
class Idea(BaseModel):
    title: str
    description: str
    tags: List[str]

class LensSelectorRequest(BaseModel):
    studyId: str
    idea: Idea
    stage: str  # "idea", "prototype", or "beta"

# ==== Prompt Builder ====
def build_prompt(idea: Idea, stage: str):
    return f"""
You are an AI research strategist helping a startup choose the best validation methods.

Given:
- Idea Title: {idea.title}
- Description: {idea.description}
- Tags: {', '.join(idea.tags)}
- Stage: {stage}

Available research lenses:
- SME (interviews with experts)
- Peer (calls with fellow founders)
- Survey (structured questions to users)
- Social (Reddit/Quora/Discord sentiment scraping)

Your job:
1. Rank the 4 lenses (1 = most useful).
2. For each lens, provide:
   - rank
   - confidence (0–1)
   - reason (why it's useful or not)
   - confidenceBasis (how you derived the score)
   - pros (1–2 bullets)
   - cons (1–2 bullets)

Format the output as a JSON array like this:
[
  {{
    "lens": "SME",
    "rank": 1,
    "reason": "...",
    "confidence": 0.85,
    "confidenceBasis": "...",
    "pros": ["...", "..."],
    "cons": ["...", "..."]
  }},
  ...
]

Return ONLY valid JSON with 4 entries.
"""

# ==== Nova Micro Call ====
def query_nova_micro(prompt_text):
    body = {
        "inferenceConfig": {
            "max_new_tokens": 1200
        },
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt_text}]
            }
        ]
    }

    response = bedrock.invoke_model_with_response_stream(
        modelId=inference_profile_arn,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    output_string = ""
    for event in response["body"]:
        if "chunk" in event:
            chunk = event["chunk"]["bytes"]
            if chunk:
                try:
                    payload = json.loads(chunk.decode("utf-8"))
                    delta = payload.get("contentBlockDelta", {}).get("delta", {}).get("text", "")
                    output_string += delta
                except Exception:
                    continue
    return output_string

# ==== Endpoint ====
@app.post("/api/ai/lens-selector")
async def lens_selector(payload: LensSelectorRequest):
    prompt = build_prompt(payload.idea, payload.stage)
    raw_output = query_nova_micro(prompt)

    try:
        parsed = json.loads(raw_output)
        return JSONResponse(content=parsed)
    except:
        return JSONResponse(content={"raw_output": raw_output, "error": "Could not parse JSON from model"}, status_code=200)
