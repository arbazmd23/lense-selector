import streamlit as st
import anthropic
import json
import toml
from pathlib import Path
from typing import List, Optional
import traceback

# Configure page
st.set_page_config(
    page_title="INLAW Research Lens Selector",
    page_icon="🔍",
    layout="wide"
)

# === API Key Setup ===
@st.cache_resource
def get_api_key():
    try:
        # First try Streamlit secrets (for deployment)
        if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
            api_key = st.secrets['ANTHROPIC_API_KEY']
            if api_key and api_key.strip():
                return api_key.strip()
        
        # Fallback to .env file (for local development)
        env_path = Path(".env")
        if env_path.exists():
            env_content = toml.load(env_path)
            api_key = env_content.get("anthropic", {}).get("api_key")
            if api_key and api_key.strip():
                return api_key.strip()
    except Exception as e:
        st.error(f"Error reading API key: {e}")
    
    raise Exception("API key not found. Please add ANTHROPIC_API_KEY to Streamlit secrets or .env file")

# Initialize Anthropic client
@st.cache_resource
def get_anthropic_client():
    try:
        return anthropic.Client(api_key=get_api_key())
    except Exception as e:
        st.error(f"Failed to initialize Anthropic client: {e}")
        return None

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
def build_prompt(title: str, description: str, tags: List[str], stage: str):
    stage_info = STARTUP_STAGES.get(stage, {})
    key_questions = stage_info.get("key_questions", [])
    focus_areas = stage_info.get("focus_areas", [])
    
    # Create stage-specific context for AI
    stage_context = ""
    if stage == "IDEATION & PLANNING":
        stage_context = """Early stage focusing on idea validation, market research, and planning. 
        At this stage, you need to validate core assumptions and understand market needs. 
        SME insights help validate technical feasibility, Peer insights provide business model validation,
        Survey helps quantify market demand, Social reveals organic market conversations."""
    elif stage == "PROTOTYPE DEVELOPMENT":
        stage_context = """Building MVP stage focusing on product development and team coordination. 
        At this stage, you need technical validation and user experience feedback.
        SME insights crucial for technical decisions, Peer insights for development best practices,
        Survey for feature prioritization, Social for competitive analysis."""
    elif stage == "VALIDATION & ITERATION":
        stage_context = """Testing and refining stage focusing on user feedback and usability. 
        At this stage, direct user insights and iteration guidance are critical.
        Survey and Social become more valuable for user feedback, SME for technical optimization,
        Peer for scaling challenges."""
    elif stage == "LAUNCH & SCALING":
        stage_context = """Go-to-market stage focusing on customer acquisition and scaling. 
        At this stage, market strategy and growth insights are paramount.
        Peer insights for go-to-market strategies, Survey for pricing/positioning,
        Social for brand awareness, SME for operational scaling."""
    elif stage == "GROWTH & OPTIMIZATION":
        stage_context = """Mature scaling stage focusing on optimization and expansion. 
        At this stage, competitive intelligence and growth optimization are key.
        Survey for market expansion research, Social for competitive intelligence,
        Peer for scaling strategies, SME for advanced optimizations."""

    return f"""
You are an expert startup advisor with deep knowledge of research methodologies. Analyze this startup and determine which research method would provide the MOST actionable insights at this specific stage.

STARTUP CONTEXT:
- Title: {title}
- Description: {description}
- Tags: {', '.join(tags)}
- Current Stage: {stage}

STAGE CONTEXT: {stage_context}

RESEARCH LENSES TO ANALYZE:

SME (Subject Matter Expert Research):
- What: Direct interviews with domain experts, industry veterans, technical specialists, regulatory experts
- Provides: Deep technical validation, industry standards, regulatory requirements, feasibility assessment
- Best when: Complex technical challenges, regulated industries, specialized knowledge gaps, feasibility questions

Peer (Peer-to-Peer Research):
- What: Conversations with fellow entrepreneurs, startup founders, business leaders who've faced similar challenges
- Provides: Business model validation, go-to-market strategies, operational insights, scaling experiences
- Best when: Business strategy questions, operational challenges, fundraising, scaling decisions

Survey (Quantitative Research):
- What: Structured questionnaires to collect statistical data from target users/customers
- Provides: Market size validation, feature prioritization, pricing insights, user preference quantification
- Best when: Large addressable markets, consumer products, statistical validation needed, pricing decisions

Social (Social Media Analysis):
- What: Mining social platforms, forums, communities for organic conversations, sentiment, trends
- Provides: Brand perception, competitive intelligence, market trends, organic user feedback
- Best when: Consumer brands, trend-dependent products, competitive analysis, brand-sensitive markets

CRITICAL RANKING LOGIC:

For IDEATION & PLANNING stage:
- If highly technical/regulated → SME likely most valuable
- If business model unclear → Peer insights crucial
- If large consumer market → Survey for demand validation
- If trend/brand dependent → Social for market signals

For PROTOTYPE DEVELOPMENT stage:
- If technical complexity high → SME for development guidance
- If user experience critical → Survey for user testing
- If business model validation needed → Peer for strategy
- If competitive landscape active → Social for positioning

For VALIDATION & ITERATION stage:
- If user feedback critical → Survey typically #1
- If technical optimization needed → SME for advanced insights
- If business model pivoting → Peer for strategic guidance
- If market positioning unclear → Social for perception

For LAUNCH & SCALING stage:
- If go-to-market strategy unclear → Peer for execution insights
- If market sizing needed → Survey for demand quantification
- If brand building critical → Social for awareness strategies
- If operational scaling → SME for infrastructure

For GROWTH & OPTIMIZATION stage:
- If competitive intelligence needed → Social for market dynamics
- If expansion planning → Survey for new market validation
- If operational optimization → SME for advanced systems
- If strategic pivoting → Peer for scaling experiences

RANKING REQUIREMENTS:
1. Assign ranks 1-4 where 1 = most valuable, 4 = least valuable
2. Each lens must have a UNIQUE rank (no ties)
3. Rank based on which method provides the MOST ACTIONABLE insights for this specific startup at this stage
4. Consider: stage needs + domain complexity + target market + business model + title specifics + description details + tag implications
5. Rankings must vary significantly across different contexts - avoid defaulting to same patterns
6. CRITICAL: Analyze the COMPLETE startup context (title + description + tags + stage) to determine rankings

CONTEXT ANALYSIS REQUIREMENTS:
- Analyze the startup TITLE for business model clues
- Analyze the DESCRIPTION for technical complexity, target market, and value proposition
- Analyze the TAGS for domain, technology stack, and market type
- Analyze the STAGE for specific research needs at this phase
- Combine ALL these factors to determine optimal research lens ranking

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "lens": "SME",
    "rank": [1, 2, 3, or 4 - calculated based on value for this specific context],
    "reason": "Brief explanation why this rank for this specific startup/stage considering title, description, tags, and stage",
    "confidence": [0.1_TO_1.0],
    "confidenceBasis": "Why you're confident/uncertain about this ranking",
    "pros": ["Advantage 1 for this context", "Advantage 2 for this context"],
    "cons": ["Limitation 1 for this context", "Limitation 2 for this context"],
    "stageRelevance": [0.1_TO_1.0]
  }},
  {{
    "lens": "Peer",
    "rank": [1, 2, 3, or 4 - calculated based on value for this specific context],
    "reason": "Brief explanation why this rank for this specific startup/stage considering title, description, tags, and stage",
    "confidence": [0.1_TO_1.0],
    "confidenceBasis": "Why you're confident/uncertain about this ranking",
    "pros": ["Advantage 1 for this context", "Advantage 2 for this context"],
    "cons": ["Limitation 1 for this context", "Limitation 2 for this context"],
    "stageRelevance": [0.1_TO_1.0]
  }},
  {{
    "lens": "Survey",
    "rank": [1, 2, 3, or 4 - calculated based on value for this specific context],
    "reason": "Brief explanation why this rank for this specific startup/stage considering title, description, tags, and stage",
    "confidence": [0.1_TO_1.0],
    "confidenceBasis": "Why you're confident/uncertain about this ranking",
    "pros": ["Advantage 1 for this context", "Advantage 2 for this context"],
    "cons": ["Limitation 1 for this context", "Limitation 2 for this context"],
    "stageRelevance": [0.1_TO_1.0]
  }},
  {{
    "lens": "Social",
    "rank": [1, 2, 3, or 4 - calculated based on value for this specific context],
    "reason": "Brief explanation why this rank for this specific startup/stage considering title, description, tags, and stage",
    "confidence": [0.1_TO_1.0],
    "confidenceBasis": "Why you're confident/uncertain about this ranking",
    "pros": ["Advantage 1 for this context", "Advantage 2 for this context"],
    "cons": ["Limitation 1 for this context", "Limitation 2 for this context"],
    "stageRelevance": [0.1_TO_1.0]
  }}
]

IMPORTANT: Analyze the specific context deeply and rank based on maximum actionable value. Different startups with different titles, descriptions, tags, and stages should get significantly different rankings. Rankings must be context-sensitive and vary meaningfully.
"""

# === Claude API Call ===
def query_claude(prompt_text: str, client):
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            temperature=0.4,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
        )
        
        output_string = response.content[0].text.strip()
        
        # Find JSON array in the response
        start_idx = output_string.find('[')
        end_idx = output_string.rfind(']') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = output_string[start_idx:end_idx]
            return json_str
        
        return output_string
        
    except Exception as e:
        raise Exception(f"Claude API error: {str(e)}")

# === Streamlit UI ===
def main():
    st.title("🔍 INLAW Research Lens Selector")
    st.markdown("**Version 1.0.0** - Analyze your startup and get research lens recommendations")
    
    # Initialize client
    client = get_anthropic_client()
    
    if not client:
        st.error("❌ Failed to initialize Anthropic client. Please add ANTHROPIC_API_KEY to Streamlit secrets.")
        st.info("💡 In Streamlit Cloud: Go to App Settings → Secrets → Add your API key as ANTHROPIC_API_KEY")
        st.stop()
    
    st.success("✅ Anthropic client initialized successfully")
    
    # Sidebar with stage information
    st.sidebar.header("📊 Available Stages")
    for stage, info in STARTUP_STAGES.items():
        with st.sidebar.expander(stage):
            st.write("**Key Questions:**")
            for question in info["key_questions"][:3]:  # Show first 3 questions
                st.write(f"• {question}")
            st.write("**Focus Areas:**")
            st.write(f"• {', '.join(info['focus_areas'])}")
    
    # Main form
    st.header("🚀 Startup Analysis")
    
    with st.form("startup_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            title = st.text_input("Startup Title", placeholder="Enter your startup title")
            description = st.text_area(
                "Description", 
                placeholder="Describe your startup, its value proposition, and target market",
                height=100
            )
        
        with col2:
            stage = st.selectbox("Current Stage", list(STARTUP_STAGES.keys()))
            tags_input = st.text_area(
                "Tags (one per line)", 
                placeholder="e.g.\nAI\nSaaS\nHealthtech",
                height=100
            )
        
        submitted = st.form_submit_button("🔍 Analyze Startup", use_container_width=True)
    
    if submitted:
        # Validate inputs
        if not title.strip():
            st.error("❌ Title cannot be empty")
            return
        if not description.strip():
            st.error("❌ Description cannot be empty")
            return
        if not tags_input.strip():
            st.error("❌ Tags cannot be empty")
            return
        
        # Process tags
        tags = [tag.strip() for tag in tags_input.split('\n') if tag.strip()]
        
        if not tags:
            st.error("❌ Please provide at least one tag")
            return
        
        # Show analysis in progress
        with st.spinner("🤖 Analyzing your startup with Claude..."):
            try:
                # Build prompt
                prompt = build_prompt(title, description, tags, stage)
                
                # Query Claude
                raw_output = query_claude(prompt, client)
                
                # Parse JSON response
                try:
                    parsed = json.loads(raw_output)
                    
                    # Validate response format
                    if not isinstance(parsed, list) or len(parsed) != 4:
                        st.error("❌ Invalid response format - expected array of 4 items")
                        return
                        
                    # Validate required fields
                    required_fields = ['lens', 'rank', 'reason', 'confidence', 'pros', 'cons', 'stageRelevance']
                    for item in parsed:
                        for field in required_fields:
                            if field not in item:
                                st.error(f"❌ Missing field: {field}")
                                return
                    
                    # Validate rankings are unique and 1-4
                    ranks = [item['rank'] for item in parsed]
                    if sorted(ranks) != [1, 2, 3, 4]:
                        st.error("❌ Invalid rankings - must be unique values 1-4")
                        return
                    
                    # Calculate summary
                    avg_confidence = sum(item['confidence'] for item in parsed) / len(parsed)
                    high_confidence_lenses = [item['lens'] for item in parsed if item['confidence'] > 0.7]
                    top_lens = min(parsed, key=lambda x: x['rank'])['lens']
                    
                    summary = {
                        "average_confidence": round(avg_confidence, 3),
                        "high_confidence_lenses": high_confidence_lenses,
                        "top_recommendation": top_lens,
                        "stage": stage
                    }
                    
                    # Create final result
                    result = {
                        "results": parsed,
                        "summary": summary
                    }
                    
                    # Display results
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display JSON output
                    st.header("📋 Analysis Results (JSON)")
                    st.json(result)
                    
                    # Option to download JSON
                    st.download_button(
                        label="💾 Download JSON Results",
                        data=json.dumps(result, indent=2),
                        file_name=f"research_lens_analysis_{title.replace(' ', '_')}.json",
                        mime="application/json"
                    )
                    
                except json.JSONDecodeError as e:
                    st.error(f"❌ Failed to parse AI response as JSON: {str(e)}")
                    st.text("Raw response:")
                    st.text(raw_output)
                except ValueError as e:
                    st.error(f"❌ Invalid AI response format: {str(e)}")
                    st.text("Raw response:")
                    st.text(raw_output)
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.text("Full error details:")
                st.text(traceback.format_exc())

if __name__ == "__main__":
    main()