import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.agents import SequentialAgent

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

# Load env
load_dotenv()

model_name = os.getenv("MODEL") or "gemini-1.5-flash"

# =========================================================
# 🧠 STATE TOOLS
# =========================================================

def add_transcript_to_state(tool_context: ToolContext, transcript: str):
    tool_context.state["TRANSCRIPT"] = transcript
    logging.info("[STATE] Transcript stored")
    return {"status": "transcript_saved"}


def save_summary(tool_context: ToolContext, summary: str):
    tool_context.state["SUMMARY"] = summary
    logging.info("[STATE] Summary saved")
    return {"status": "summary_saved"}


def save_slides(tool_context: ToolContext, slides: str):
    tool_context.state["SLIDES"] = slides
    logging.info("[STATE] Slides saved")
    return {"status": "slides_saved"}


def save_actions(tool_context: ToolContext, actions: str):
    tool_context.state["ACTIONS"] = actions
    logging.info("[STATE] Actions saved")
    return {"status": "actions_saved"}


# =========================================================
# 📝 1. SUMMARIZER AGENT
# =========================================================

summarizer = Agent(
    name="summarizer",
    model=model_name,
    tools=[save_summary],
    description="Summarizes meeting transcript into concise insights.",
    instruction="""
You are a professional meeting summary assistant.

TASK:
- Read the transcript from {TRANSCRIPT}
- Identify the main topic
- Generate a clear and concise summary

**FORMAT :**

Here is your extensive summary for your meeting transcript:

• Point 1
• Point 2
• Point 3
• Point 4
• (Max 6 points)

=========

IMPORTANT RULES:
- ONLY generate summary (NO explanation)
- DO NOT say "Here are summary"
- DO NOT give any confirmation regarding summary creation
- DO NOT print summary
- DO NOT generate any confirmation message

CRITICAL:
- Pass ONLY the slide content inside 'save_summary'
- DO NOT pass any other text

FINAL RULE:
- Your final response MUST be a tool call to 'save_summary'
""",
    output_key="SUMMARY"
)


# =========================================================
# 📊 2. SLIDE GENERATOR AGENT
# =========================================================

slide_generator = Agent(
    name="slide_generator",
    model=model_name,
    tools=[save_slides],
    description="Converts summary into presentation slides.",
#     instruction="""
# You are a presentation assistant.

# TASK:
# - Read summary from {SUMMARY}
# - Create 4–5 slides

# **FORMAT :**

# Here are the contents for your slide

# Slide 1: Title
# - ...

# Slide 2:
# - ...

# Slide 3:
# - ...

# Slide 4:
# - ...

# Slide 5 (optional):
# - ...

# IMPORTANT:
# - DO NOT print slides
# - ONLY call 'save_slides'

# RULES:
# - Each slide: 3–5 bullet points
# - Keep content short
# - Short bullet points
# - No extra text
# - Final response MUST be a tool call

# """,
    instruction="""
You are a presentation assistant.

TASK:
- Read summary from {SUMMARY}
- Create 4–5 slides

FORMAT (STRICT):

Slide 1: Title
- ...

Slide 2:
- ...

Slide 3:
- ...

Slide 4:
- ...

Slide 5 (optional):
- ...

IMPORTANT RULES:
- ONLY generate slide content (NO explanation)
- DO NOT say "Here are slides"
- DO NOT say "Slides created"
- DO NOT generate any confirmation message

CRITICAL:
- Pass ONLY the slide content inside 'save_slides'
- DO NOT pass any other text

FINAL RULE:
- Your final response MUST be a tool call to 'save_slides'
""",
    output_key="SLIDES"
)


# =========================================================
# ✅ 3. ACTION ITEMS AGENT
# =========================================================

action_agent = Agent(
    name="action_agent",
    model=model_name,
    tools=[save_actions],
    description="Extracts structured action items from transcript.",
    instruction="""
You are an action item extraction assistant.

TASK:
- Read transcript from {TRANSCRIPT}
- Extract actionable tasks

**FORMAT :**

Here are your action items.

Action Items:

1. Task: ...
   Owner: ...
   Priority: High/Medium/Low
   Deadline: ...

(4–8 items)

IMPORTANT RULES:
- ONLY generate action Items (NO explanation)
- DO NOT say "Here are Action Items"
- DO NOT say "Action Items created"
- DO NOT generate any confirmation message
- DO NOT print action Items

CRITICAL:
- Pass ONLY the slide content inside 'save_actions'
- DO NOT pass any other text

FINAL RULE:
- Your final response MUST be a tool call to 'save_actions'
""",
    output_key="ACTIONS"
)
# =========================================================
# 📢 4. FINAL FORMATTER AGENT (IMPORTANT 🔥)
# =========================================================

final_formatter = Agent(
    name="final_formatter",
    model=model_name,
    description="Displays final output to user",
    instruction="""
You are the final response formatter.

Read:
- Summary from {SUMMARY}
- Slides from {SLIDES}
- Actions from {ACTIONS}

DISPLAY FORMAT:

========================
📌 SUMMARY
{SUMMARY}

========================
📊 SLIDES
{SLIDES}

========================
✅ ACTION ITEMS
{ACTIONS}

IMPORTANT:
- Only display content
- DO NOT call any tools
- DO NOT modify data
"""
)


# =========================================================
# 🔄 WORKFLOW (SEQUENTIAL)
# =========================================================

meetwise_workflow = SequentialAgent(
    name="meetwise_workflow",
    description="summary → slides → actions and final formatting",
    sub_agents=[
        summarizer,
        slide_generator,
        action_agent,
        final_formatter
    ]
)


# =========================================================
# 🌟 ROOT AGENT
# =========================================================

root_agent = Agent(
    name="root_agent",
    model=model_name,
    description="MeetWise AI main coordinator",
    tools=[add_transcript_to_state],
    sub_agents=[meetwise_workflow],
#     instruction="""
# You are MeetWise AI — a smart meeting assistant.

# FLOW:
# 1. Greet user in a friendly and catchy way
# 2. Ask for meeting transcript

# WHEN USER PROVIDES TRANSCRIPT:
# - Call 'add_transcript_to_state'
# - Then transfer to 'meetwise_workflow'

# IMPORTANT:
# - ALWAYS store transcript before processing
# - DO NOT skip workflow
# """
    instruction="""
You are MeetWise AI — an intelligent and efficient meeting assistant which helps users analyze meeting transcripts by generating summaries, slides, and action items.

FLOW:
1. Greet user in cathcy
2. Ask for transcript

WHEN transcript is provided:
- ALWAYS call 'add_transcript_to_state' (overwrite previous transcript)
- Then transfer to 'meetwise_workflow'

IMPORTANT:
- Always treat latest user input as new transcript
- Overwrite old data
- Trigger workflow only once per user input
- Do NOT call tools multiple times in a single turn
"""
)


# =========================================================
# 🚀 ENTRY POINT (OPTIONAL FOR LOCAL TEST)
# =========================================================

# if __name__ == "__main__":
#     print("MeetWise AI Agent system ready 🚀")
