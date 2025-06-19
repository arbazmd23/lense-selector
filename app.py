import streamlit as st
import boto3
import json

# === AWS Bedrock Setup ===
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
inference_profile_arn = "arn:aws:bedrock:ap-south-1:069717477936:inference-profile/apac.amazon.nova-micro-v1:0"

# === Startup Stages Configuration ===
STARTUP_STAGES = {
    "IDEATION & PLANNING": {
        "key_questions": [
            "Why are you building this prototype?",
            "What are your key objectives?", 
            "What defines success for this prototype?",
            "What type of prototype do you need (low/high fidelity)?",
            "Who is your target audience and what do they need?",
            "How are they solving this problem today?",
            "What pain points exist in current solutions?"
        ],
        "focus_areas": ["Idea validation", "Market research", "User needs analysis", "Prototype planning"]
    },
    "PROTOTYPE DEVELOPMENT": {
        "key_questions": [
            "Why are you building this prototype?",
            "What are your key objectives?",
            "What defines success for this prototype?", 
            "What type of prototype is needed (low/high fidelity)?",
            "Who is your target audience and what do they need?",
            "How are users solving this problem currently?",
            "What pain points exist in current solutions?",
            "How easy is it for users to navigate and complete tasks?"
        ],
        "focus_areas": ["MVP development", "Team role clarity", "Task delegation", "User experience testing"]
    },
    "VALIDATION & ITERATION": {
        "key_questions": [
            "What problem does this prototype solve for you?",
            "What did you expect to see that was missing?",
            "How easy was it to navigate the prototype?",
            "Could you easily find what you were looking for?",
            "Did you encounter any difficulties while using the prototype?"
        ],
        "focus_areas": ["User feedback collection", "Usability testing", "Feature validation", "Market validation"]
    },
    "LAUNCH & SCALING": {
        "key_questions": [
            "What's the launch market?",
            "How will you market?",
            "What's the revenue goal?",
            "What's the scaling strategy?"
        ],
        "focus_areas": ["Go-to-market strategy", "Customer acquisition", "Revenue generation", "Market expansion"]
    },
    "GROWTH & OPTIMIZATION": {
        "key_questions": [
            "How will you reach your target market?",
            "What are your key distribution channels?",
            "What are your expected user/customer acquisition costs?",
            "How will you attract and retain customers?",
            "What are your key metrics for success?",
            "Who are your main competitors, and how do you differentiate yourself?",
            "How many people are on your team, and are they sufficient for your growth plans?"
        ],
        "focus_areas": ["Market penetration", "Customer retention", "Competitive analysis", "Team scaling", "Financial optimization"]
    }
}

# === Enhanced Prompt Builder ===
def build_prompt(title, description, tags, stage):
    stage_info = STARTUP_STAGES.get(stage, {})
    key_questions = stage_info.get("key_questions", [])
    focus_areas = stage_info.get("focus_areas", [])
    
    # Create stage-specific context for AI
    stage_context = ""
    if stage == "IDEATION & PLANNING":
        stage_context = "Early stage focusing on idea validation, market research, and planning. Prioritize methods that help validate core assumptions and understand market needs."
    elif stage == "PROTOTYPE DEVELOPMENT":
        stage_context = "Building MVP stage focusing on product development and team coordination. Prioritize methods that provide technical validation and user experience feedback."
    elif stage == "VALIDATION & ITERATION":
        stage_context = "Testing and refining stage focusing on user feedback and usability. Prioritize methods that provide direct user insights and iteration guidance."
    elif stage == "LAUNCH & SCALING":
        stage_context = "Go-to-market stage focusing on customer acquisition and scaling. Prioritize methods that provide market strategy and growth insights."
    elif stage == "GROWTH & OPTIMIZATION":
        stage_context = "Mature scaling stage focusing on optimization and expansion. Prioritize methods that provide competitive intelligence and growth optimization insights."
    
    # Industry/domain analysis
    domain_context = ""
    tag_string = ', '.join(tags).lower()
    if any(tech in tag_string for tech in ['ai', 'ml', 'tech', 'software', 'app']):
        domain_context = "Tech domain - SME and Peer insights often most valuable for technical validation."
    elif any(health in tag_string for health in ['health', 'medical', 'healthcare']):
        domain_context = "Healthcare domain - SME insights critical for regulatory and safety considerations."
    elif any(consumer in tag_string for consumer in ['b2c', 'consumer', 'retail']):
        domain_context = "Consumer domain - Survey and Social insights valuable for understanding user preferences."
    elif any(b2b in tag_string for b2b in ['b2b', 'enterprise', 'business']):
        domain_context = "B2B domain - SME and Peer insights crucial for understanding business needs."
    
    return f"""
Analyze this startup and rank 4 research methods. Return ONLY valid JSON.

Context:
- Title: {title}
- Description: {description}
- Tags: {', '.join(tags)}
- Stage: {stage}

{stage_context}
{domain_context}

Rank these 4 research lenses (1=best, 4=worst):
- SME: Expert interviews
- Peer: Founder conversations  
- Survey: User questionnaires
- Social: Social media analysis

Return exactly this JSON format:
[
{{"lens":"SME","rank":1,"reason":"Brief reason","confidence":0.8,"confidenceBasis":"Brief basis","pros":["Pro 1","Pro 2"],"cons":["Con 1","Con 2"],"stageRelevance":0.9}},
{{"lens":"Peer","rank":2,"reason":"Brief reason","confidence":0.7,"confidenceBasis":"Brief basis","pros":["Pro 1","Pro 2"],"cons":["Con 1","Con 2"],"stageRelevance":0.8}},
{{"lens":"Survey","rank":3,"reason":"Brief reason","confidence":0.6,"confidenceBasis":"Brief basis","pros":["Pro 1","Pro 2"],"cons":["Con 1","Con 2"],"stageRelevance":0.7}},
{{"lens":"Social","rank":4,"reason":"Brief reason","confidence":0.5,"confidenceBasis":"Brief basis","pros":["Pro 1","Pro 2"],"cons":["Con 1","Con 2"],"stageRelevance":0.6}}]
"""

# === Nova Micro Call with JSON Cleanup ===
def query_nova_micro(prompt_text):
    body = {
        "inferenceConfig": {
            "max_new_tokens": 800,
            "temperature": 0.3
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
    
    # Clean up the output to extract JSON
    output_string = output_string.strip()
    
    # Find JSON array in the response
    start_idx = output_string.find('[')
    end_idx = output_string.rfind(']') + 1
    
    if start_idx != -1 and end_idx != -1:
        json_str = output_string[start_idx:end_idx]
        # Fix common JSON issues
        json_str = json_str.replace("'", '"')  # Replace single quotes
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        return json_str
    
    return output_string

# === Streamlit UI ===
st.set_page_config(page_title="INLAW Lens Selector", page_icon="🔍", layout="wide")

st.title("🔍 INLAW Research Lens Selector")
st.markdown("*AI-powered research method recommendations for your startup stage*")

# === Input Form ===
col1, col2 = st.columns([2, 1])

with col1:
    title = st.text_input("📌 **Startup Idea Title**", placeholder="e.g., AI-powered fitness coach app")
    description = st.text_area("📝 **Idea Description**", 
                              placeholder="Describe your startup idea, target market, and core value proposition...",
                              height=100)
    tags_input = st.text_input("🏷️ **Tags** (comma separated)", 
                              placeholder="e.g., AI, fitness, mobile app, B2C")

with col2:
    stage = st.selectbox("📊 **Current Stage**", 
                        options=list(STARTUP_STAGES.keys()),
                        help="Select your current startup development stage")
    
    # Display stage info
    if stage:
        stage_info = STARTUP_STAGES[stage]
        st.markdown("**Stage Focus:**")
        for area in stage_info["focus_areas"]:
            st.markdown(f"• {area}")

# === Stage-specific guidance removed ===

# === Main Analysis ===
if st.button("🚀 **Get Research Lens Recommendations**", type="primary") and title and description and tags_input and stage:
    tags = [tag.strip() for tag in tags_input.split(",")]
    prompt = build_prompt(title, description, tags, stage)

    with st.spinner("⏳ Analyzing your startup context..."):
        raw_output = query_nova_micro(prompt)

    try:
        # Try to parse the cleaned JSON
        parsed = json.loads(raw_output)
        
        # Validate we have 4 lens entries
        if not isinstance(parsed, list) or len(parsed) != 4:
            raise ValueError("Invalid response format")
            
        # Ensure all required fields exist
        required_fields = ['lens', 'rank', 'reason', 'confidence', 'pros', 'cons']
        for item in parsed:
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing field: {field}")
        
        st.success("✅ **Personalized Research Lens Recommendations**")
        st.markdown(f"*Optimized for {stage} stage*")
        
        # === Results Display ===
        sorted_results = sorted(parsed, key=lambda x: x["rank"])
        
        for item in sorted_results:
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣"][item["rank"]-1]
            
            with st.container():
                st.markdown(f"### {rank_emoji} {item['lens']} Research Lens")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Recommendation:** {item['reason']}")
                
                with col2:
                    confidence_color = "green" if item['confidence'] > 0.7 else "orange" if item['confidence'] > 0.5 else "red"
                    st.markdown(f"**Confidence:** :{confidence_color}[{item['confidence']:.1%}]")
                    
                with col3:
                    stage_relevance = item.get('stageRelevance', 0.5)
                    relevance_color = "green" if stage_relevance > 0.7 else "orange" if stage_relevance > 0.5 else "red"
                    st.markdown(f"**Stage Fit:** :{relevance_color}[{stage_relevance:.1%}]")
                
                # Pros and Cons
                col_pros, col_cons = st.columns(2)
                
                with col_pros:
                    st.markdown("**✅ Advantages:**")
                    for pro in item["pros"]:
                        st.markdown(f"• {pro}")
                
                with col_cons:
                    st.markdown("**⚠️ Limitations:**")
                    for con in item["cons"]:
                        st.markdown(f"• {con}")
                
                st.markdown(f"*Confidence basis: {item['confidenceBasis']}*")
                st.markdown("---")
        
        # === Next Steps ===
        st.markdown("### 🎯 **Recommended Next Steps**")
        top_lens = sorted_results[0]['lens']
        st.info(f"**Start with {top_lens} research** - it's ranked #1 for your {stage} stage. "
                f"Consider combining it with the #2 ranked method for comprehensive insights.")
                
    except Exception as e:
        st.error("❌ **Failed to parse AI response**")
        st.markdown("**Raw AI Output:**")
        st.text_area("Debug Output", raw_output, height=300)
        st.markdown(f"**Error:** {str(e)}")

# === Sidebar Info ===
with st.sidebar:
    st.markdown("### 🔍 **About Research Lenses**")
    st.markdown("""
    **SME**: Expert interviews for deep technical insights
    
    **Peer**: Founder-to-founder conversations for startup-specific advice
    
    **Survey**: Structured questionnaires for quantitative validation
    
    **Social**: Social media sentiment analysis for organic feedback
    """)
    
    st.markdown("### 📊 **Startup Stages**")
    for stage_name in STARTUP_STAGES.keys():
        st.markdown(f"• **{stage_name}**")
    
    st.markdown("---")
    st.markdown("*Powered by AWS Bedrock Nova Micro*")