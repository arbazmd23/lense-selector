import streamlit as st
import boto3
import json

# === AWS Bedrock Setup ===
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
inference_profile_arn = "arn:aws:bedrock:ap-south-1:069717477936:inference-profile/apac.amazon.nova-micro-v1:0"

# === Prompt Builder ===
def build_prompt(title, description, tags, stage):
    return f"""
You are an AI research strategist helping a startup choose the best validation methods.

Given:
- Idea Title: {title}
- Description: {description}
- Tags: {', '.join(tags)}
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

# === Nova Micro Call ===
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

# === Streamlit UI ===
st.title("🔍 Lens Selector AI (Nova Micro)")

title = st.text_input("📌 Idea Title")
description = st.text_area("📝 Idea Description")
tags_input = st.text_input("🏷️ Tags (comma separated)")
stage = st.selectbox("📊 Stage", options=["idea", "prototype", "beta"])

if st.button("🚀 Recommend Research Lenses") and title and description and tags_input and stage:
    tags = [tag.strip() for tag in tags_input.split(",")]
    prompt = build_prompt(title, description, tags, stage)

    st.info("⏳ Querying Nova Micro...")
    raw_output = query_nova_micro(prompt)

    try:
        parsed = json.loads(raw_output)
        st.success("✅ Ranked Research Lenses")
        for item in sorted(parsed, key=lambda x: x["rank"]):
            st.subheader(f"🔎 {item['lens']} (Rank {item['rank']})")
            st.markdown(f"**Confidence:** {item['confidence']} ({item['confidenceBasis']})")
            st.markdown(f"**Reason:** {item['reason']}")
            st.markdown("**Pros:**")
            st.markdown("\n".join(f"- {p}" for p in item["pros"]))
            st.markdown("**Cons:**")
            st.markdown("\n".join(f"- {c}" for c in item["cons"]))
            st.markdown("---")
    except Exception:
        st.error("❌ Failed to parse JSON output")
        st.text_area("Raw Output", raw_output, height=300)
