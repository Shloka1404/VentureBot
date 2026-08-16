import ast
import re
from datetime import datetime
from collections import defaultdict

import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct",
                         bill_to="kode-with-klossy")
# future idea - make it so it uses a web search tool instead of a resources txt file
with open("resources.txt", "r", encoding="utf-8") as f:
    RESOURCES = f.read()

RESOURCE_ENTRIES = [block.strip()
                    for block in RESOURCES.split("\n\n") if block.strip()]

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
TOP_K = 6

RESOURCE_EMBEDDINGS = (
    EMBED_MODEL.encode(
        RESOURCE_ENTRIES, convert_to_tensor=True) if RESOURCE_ENTRIES else None
)


def normalize_history(history):
    """Convert Gradio chat history into a plain list of
    {'role': ..., 'content': ...} dicts. Tolerant of the modern
    list-of-dicts format, the older list-of-tuples format, and
    non-string content (e.g. file attachments)."""
    normalized = []
    if not history:
        return normalized

    for turn in history:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            normalized.append({"role": role, "content": str(content)})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, bot_msg = turn
            if user_msg:
                normalized.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                normalized.append(
                    {"role": "assistant", "content": str(bot_msg)})

    return normalized


def retrieve_relevant_resources(message, history, top_k=TOP_K):
    """Return the top_k resource entries most relevant to the current
    conversation, instead of the entire resources.txt file."""
    if RESOURCE_EMBEDDINGS is None:
        return "(No resources loaded yet — add resources.txt in this folder.)"

    recent_user_msgs = [
        turn["content"] for turn in history[-4:] if turn.get("role") == "user"
    ]
    query = f"{' '.join(recent_user_msgs)} {message}".strip()

    query_embedding = EMBED_MODEL.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(
        query_embedding, RESOURCE_EMBEDDINGS, top_k=min(
            top_k, len(RESOURCE_ENTRIES))
    )[0]

    return "\n\n".join(RESOURCE_ENTRIES[hit["corpus_id"]] for hit in hits)


def build_user_facts(history, message):
    """Surface everything the user has said as an explicit block, so
    identity/negation statements can't get lost or outweighed by the
    resource list on later turns."""
    user_msgs = [turn["content"]
                 for turn in history if turn.get("role") == "user"]
    user_msgs.append(message)

    if not user_msgs:
        return "USER FACTS: none stated yet — do not assume identity or background."

    bullets = "\n".join(f"- {m}" for m in user_msgs)
    return (
        "USER FACTS (everything the user has told you, in their own words — "
        "treat any identity/negation statements here as permanent and binding, "
        "overriding anything in the resource list below):\n" + bullets
    )


def extract_text(content):
    """Handle content that may be a plain string, a list of content
    blocks, or a stringified repr of that list."""
    if isinstance(content, str):
        stripped = content.strip()

        if stripped.startswith("[{") and "'text'" in stripped:
            try:
                blocks = ast.literal_eval(stripped)
                return "".join(
                    b.get("text", "") for b in blocks if isinstance(b, dict)
                ).strip()
            except (ValueError, SyntaxError):
                return stripped
        return stripped

    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        ).strip()

    return str(content).strip()


def extract_dates_from_resources():
    """Extract specific dates and associated programs from the resources."""
    dates_dict = defaultdict(list)

    # Split resources into individual program entries
    for entry in RESOURCE_ENTRIES:
        # Look for "Due Date: " followed by dates
        due_date_match = re.search(
            r'Due Date:\s*(.+?)(?:\nLink:|$)', entry, re.DOTALL)
        if not due_date_match:
            continue

        due_date_text = due_date_match.group(1).strip()

        # Skip entries with "not specified" or "Multiple deadlines"
        if "not specified" in due_date_text.lower() or "multiple deadlines" in due_date_text.lower():
            continue

        # Extract program name
        program_match = re.search(r'Program:\s*(.+?)(?:\nWho|$)', entry)
        program_name = program_match.group(
            1).strip() if program_match else "Unknown"

        # Extract individual dates using regex
        date_patterns = [
            r'([A-Z][a-z]+\s+\d{1,2},\s*20\d{2})',  # e.g., "December 31, 2026"
            r'(\d{1,2}/\d{1,2}/\d{4})',  # e.g., "12/31/2026"
        ]

        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, due_date_text)
            found_dates.extend(matches)

        # Add each found date to the dictionary
        for date_str in found_dates:
            dates_dict[date_str].append(program_name)

    return dates_dict


def get_calendar_months():

    dates_dict = extract_dates_from_resources()
    months = set()

    for date_str in dates_dict.keys():
        try:
            date_obj = datetime.strptime(date_str, "%B %d, %Y")
            months.add((date_obj.year, date_obj.month))
        except ValueError:
            continue

    return sorted(months)


def create_calendar_for_month(year, month):
 
    import calendar

    dates_dict = extract_dates_from_resources()


    month_dates = {}
    for date_str in dates_dict.keys():
        try:
            date_obj = datetime.strptime(date_str, "%B %d, %Y")
            if date_obj.year == year and date_obj.month == month:
                month_dates[date_obj.day] = dates_dict[date_str]
        except ValueError:
            continue

    # Create calendar grid
    cal = calendar.monthcalendar(year, month)
    month_name = datetime(year, month, 1).strftime("%B %Y")

    html = f'''
    <div style="font-family: 'Fredoka', sans-serif; padding: 30px; background-color: #FDFDF8; border-radius: 18px; max-width: 100%; margin: 0 auto;">
        <h2 style="color: #2E7D4F; text-align: center; margin: 0 0 20px 0;">{month_name}</h2>
        <table style="width: 100%; border-spacing: 12px; table-layout: fixed;">
            <tr style="background-color: transparent;">
                <th style="color: #2E7D4F; padding: 12px; font-weight: 600; background-color: #B9EAC0; border-radius: 15px;">Sun</th>
                <th style="color: #2E7D4F; padding: 12px; font-weight: 600; background-color: #B9EAC0; border-radius: 15px;">Mon</th>
                <th style="color: #2E7D4F; padding: 12px; font-weight: 600; background-color: #B9EAC0; border-radius: 15px;">Tue</th>
                <th style="color: #2E7D4F; padding: 12px; font-weight: 600; background-color: #B9EAC0; border-radius: 15px;">Wed</th>
                <th style="color: #2E7D4F; padding: 12px; font-weight: 600; background-color: #B9EAC0; border-radius: 15px;">Thu</th>
                <th style="color: #2E7D4F; padding: 12px; font-weight: 600; background-color: #B9EAC0; border-radius: 15px;">Fri</th>
                <th style="color: #2E7D4F; padding: 12px; font-weight: 600; background-color: #B9EAC0; border-radius: 15px;">Sat</th>
            </tr>
    '''

    for week in cal:
        html += '<tr>'
        for day in week:
            if day == 0:
                html += '<td style="width: 14.28%; height: 140px; padding: 15px; border: 2px solid #A8D5BA; background-color: transparent; border-radius: 18px; box-sizing: border-box;"></td>'
            else:
                has_deadline = day in month_dates
                bg_color = '#D4F5DD' if has_deadline else '#FDFDF8'

                html += f'<td style="width: 14.28%; height: 140px; padding: 15px; border: 2px solid #A8D5BA; background-color: {bg_color}; text-align: center; border-radius: 18px; position: relative; vertical-align: top; box-sizing: border-box; box-shadow: 0 2px 8px rgba(46, 125, 79, 0.08);">'
                html += f'<div style="font-weight: 600; color: #2E7D4F; margin-bottom: 4px;">{day}</div>'

                if has_deadline:
                    programs = month_dates[day]
                    for prog in programs[:2]:
                        prog_short = prog[:20] + \
                            '...' if len(prog) > 20 else prog
                        html += f'<div style="font-size: 12px; color: #2E7D4F; line-height: 1.2; margin-top: 2px;">{prog_short}</div>'
                    if len(programs) > 2:
                        html += f'<div style="font-size: 10px; color: #2E7D4F; font-weight: 600;">+{len(programs)-2} more</div>'

                html += '</td>'
        html += '</tr>'

    html += '''
        </table>
    </div>
    '''
    return html


def update_calendar(current_index):
    """Update calendar display for navigation."""
    months = get_calendar_months()
    if not months or current_index < 0 or current_index >= len(months):
        return "📅 Deadline Calendar", "", current_index

    year, month = months[current_index]
    month_nav = f"{current_index + 1} / {len(months)}"
    calendar_html = create_calendar_for_month(year, month)
    return "📅 Deadline Calendar", calendar_html, current_index


SYSTEM_PROMPT_TEMPLATE = """CRITICAL RULE — READ FIRST: You do not know this person's race, \
gender, or background unless they have explicitly told you. Do NOT default to, \
lead with, or assume they are a woman or a person of color. The phrase "focus on \
women and people of color" describes the resource LIST, not an assumption about \
the person you are talking to. If they have said something like "I'm not a Black \
woman" or "I'm a man," that fact is permanent for the rest of this conversation — \
never suggest a program restricted to a group they've told you they're not part \
of, even if it resurfaces in the resource list below.
{user_facts}
You are a supportive, knowledgeable assistant that helps people find financial \
assistance and opportunity programs — grants, scholarships, job training, small \
business support, housing assistance, etc.
FACTS vs. ELABORATION — this is important:
- The RESOURCE LIST below is your ONLY source for program names, links, dollar \
amounts, deadlines, and eligibility rules. Never invent, guess, or alter any of \
these facts. If the list doesn't give a detail (e.g. exact deadline), say you're \
not sure and point the user to the link to confirm — don't make one up.
- Beyond those bare facts, DO elaborate in your own words. Don't just copy or \
lightly reword the bullet points from the file. For each program you recommend, \
explain in a couple of natural sentences: why it fits THIS person's specific \
situation (reference what they told you), what the application process is \
generally like for a program of that type (e.g. "scholarship portals like this \
usually want a short essay and transcript"), roughly how competitive or \
time-consuming it might be, and any prep the person could realistically start on \
now. It's fine to draw on general, well-known knowledge about how \
scholarships/grants/bootcamps typically work to add this context — just don't \
attach invented specifics (amounts, dates, requirements) to a specific program \
that the list didn't state.
- If nothing in the list fits, say so honestly, and suggest what general type of \
program or search term the person could look for instead — this general \
suggestion doesn't need to come from the list, since you're not citing it as a \
specific program with a link.
GIVE ADVICE, NOT JUST LINKS:
- Beyond program matches, offer the person practical next steps: how to \
prioritize which program to apply to first, how to keep track of \
deadlines/requirements, general tips for strengthening an application, and \
encouragement that's honest rather than generic.
- If it's relevant to their situation, you can also mention broader financial \
wellness pointers (e.g. free tax-prep help, budgeting basics) even if that's not \
tied to a specific program in the list — keep this general and non-prescriptive, \
not formal financial advice.
- Keep responses warm, plain-language, and non-judgmental — many users may feel \
uncomfortable discussing finances.
- Always include the program's link when you recommend it, exactly as given in \
the list.
RESOURCE LIST (this is a relevant subset retrieved for this conversation using \
semantic search, not the complete list of every program that exists, and not a \
guaranteed match):
- Semantic search can misfire, especially with negation. If the person says \
something like "I'm not Black" or "I don't have a business yet," a program below \
may still appear because it's topically similar (e.g. it mentions "Black" or \
"business") even though it's now clearly NOT a fit. Actively re-check every \
entry below against everything the person has told you about themselves before \
recommending it. Silently drop any entry that conflicts with what they've said — \
don't recommend it and don't dwell on why it showed up.
- If, after that filtering, nothing below actually fits, say so honestly rather \
than recommending a mismatched program, and suggest what type of program or \
search term they could look for instead.
{resources}
"""


def respond(message, history):
    history = normalize_history(history)
    relevant_resources = retrieve_relevant_resources(message, history)
    user_facts = build_user_facts(history, message)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        resources=relevant_resources, user_facts=user_facts
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages, max_tokens=700, temperature=0.6)
    return extract_text(response.choices[0].message.content)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&display=swap');

.gradio-container {
    background: linear-gradient(135deg, #B9EAC0, #FFF3A8) !important;
    font-family: 'Fredoka', sans-serif !important;
    font-size: 18px !important;
    color: #2E7D4F !important;
}
.gradio-container label,
.gradio-container p,
.gradio-container span {
    color: #2E7D4F !important;
    font-family: 'Fredoka', sans-serif !important;
    font-size: 18px !important;
}
footer {
    display: none !important;
}
h1 {
    color: #2E7D4F !important;
    text-align: center;
    font-size: 3rem !important;
}
.chatbot,
.chatbot .bubble-wrap,
.chatbot [data-testid="chatbot"],
.chatbot .message-wrap {
    background-color: #FDFDF8 !important;
    min-height: 1000px !important;
}
.message.user, .message.bot {
    background-color: #D4F5DD !important;
    border-radius: 18px !important;
    color: #2E7D4F !important;
}
.message.user *, .message.bot * {
    color: #2E7D4F !important;
}
button {
    background-color: #FFFFFF !important;
    color: #2E7D4F !important;
    border: 2px solid #B9EAC0 !important;
}
button:hover {
    background-color: #FAFAFA !important;
}
"""

with gr.Blocks(css=CSS) as demo:
    with gr.Tabs():
        with gr.TabItem("Chat", id="chat"):
            gr.ChatInterface(
                respond,
                title="🌿 Venture",
                editable=True,
                description="Helping women and underrepresented communities discover scholarships, grants, job training programs, and opportunities.",
                examples=[
                    ["I'm a first-generation student looking for scholarships."],
                    ["I'm a woman interested in learning coding."],
                    ["I want to start a small business but need funding."],
                    ["I'm returning to work after taking time off for family."],
                ],
            )
        with gr.TabItem("Calendar", id="calendar"):
            calendar_state = gr.State(value=0)

            gr.Markdown("# 📅 Deadline Calendar")

            with gr.Row():
                prev_btn = gr.Button("← Previous", scale=0.3)
                month_display = gr.Textbox(
                    value="1 / 1", interactive=False, scale=1.4, show_label=False)
                next_btn = gr.Button("Next →", scale=0.3)

            calendar_output = gr.HTML(create_calendar_for_month(2026, 8))

            def go_prev(current_idx):
                new_idx = max(0, current_idx - 1)
                _, cal_html, idx = update_calendar(new_idx)
                return cal_html, month_display.value if new_idx == current_idx else f"{new_idx + 1} / {len(get_calendar_months())}", new_idx

            def go_next(current_idx):
                months = get_calendar_months()
                new_idx = min(len(months) - 1, current_idx + 1)
                _, cal_html, idx = update_calendar(new_idx)
                return cal_html, month_display.value if new_idx == current_idx else f"{new_idx + 1} / {len(months)}", new_idx

            prev_btn.click(go_prev, inputs=calendar_state, outputs=[
                           calendar_output, month_display, calendar_state])
            next_btn.click(go_next, inputs=calendar_state, outputs=[
                           calendar_output, month_display, calendar_state])

if __name__ == "__main__":
    demo.launch()
