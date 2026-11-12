"""Conversation mode: role-play scenarios. Every learner turn is also pronunciation-scored."""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

SCENARIOS: dict[str, dict[str, Any]] = {
    "job_interview": {
        "title": "Job interview",
        "role": "a friendly hiring manager at a US biotech company interviewing the learner for a lab scientist role",
        "script": [
            "Thanks for coming in today. Could you tell me a little about yourself?",
            "What's a project you're especially proud of?",
            "How do you handle tight deadlines?",
            "Why do you want to work here?",
            "Do you have any questions for me?",
        ],
    },
    "lab_meeting": {
        "title": "Lab meeting",
        "role": "a principal investigator running a weekly chemistry lab meeting",
        "script": [
            "Let's start with your update. What did you work on this week?",
            "Interesting. What were the main results?",
            "Did anything go wrong with the experiment?",
            "What will you try next?",
            "Great, thanks. Anything you need from the team?",
        ],
    },
    "small_talk": {
        "title": "Small talk",
        "role": "a friendly American coworker chatting in the break room",
        "script": [
            "Hey! How was your weekend?",
            "Nice. Did you do anything fun?",
            "Have you tried any good restaurants lately?",
            "Are you planning any trips this year?",
            "Well, it was great talking to you!",
        ],
    },
    "doctor_visit": {
        "title": "Doctor's appointment",
        "role": "a primary care doctor in the US seeing the learner for a check-up",
        "script": [
            "Hi, what brings you in today?",
            "How long has that been going on?",
            "Are you taking any medication?",
            "Any allergies I should know about?",
            "Okay, here's what I'd suggest. Any questions?",
        ],
    },
}


def _system(sc: dict[str, Any]) -> str:
    return (
        f"You are {sc['role']}. Stay in character and keep the conversation going naturally. "
        "Reply in 1–3 short sentences of natural spoken American English, then ask one follow-up question. "
        "Do not comment on the learner's pronunciation or grammar; a separate system handles feedback. "
        "No markdown, no emojis."
    )


def reply(scenario: str, history: list[dict[str, str]], user_text: str) -> dict[str, str]:
    sc = SCENARIOS.get(scenario, SCENARIOS["small_talk"])
    n_user = sum(1 for h in history if h.get("role") == "user")
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        script = sc["script"]
        return {"text": script[min(n_user + 1, len(script) - 1)], "source": "script"}
    import anthropic

    client = anthropic.Anthropic()
    from anthropic.types.beta import BetaAssistantMessageParam, BetaMessageParam, BetaUserMessageParam

    msgs: list[BetaMessageParam] = [
        BetaUserMessageParam(role="user", content="(The learner walks in.)"),
        BetaAssistantMessageParam(role="assistant", content=sc["script"][0]),
    ]
    for h in history[-20:]:
        if h.get("role") == "user" and h.get("text"):
            msgs.append(BetaUserMessageParam(role="user", content=h["text"]))
        elif h.get("role") == "assistant" and h.get("text"):
            msgs.append(BetaAssistantMessageParam(role="assistant", content=h["text"]))
    msgs.append(BetaUserMessageParam(role="user", content=user_text))
    try:
        resp = client.beta.messages.create(
            model=os.environ.get("AT_COACH_MODEL", "claude-opus-5-5"),
            max_tokens=2000,
            system=_system(sc),
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            messages=msgs,
        )
    except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as e:
        log.warning("conversation LLM failed: %s", e)
        return {"text": sc["script"][min(n_user + 1, len(sc["script"]) - 1)], "source": "script"}
    if resp.stop_reason == "refusal":
        return {"text": "Let's talk about something else. " + sc["script"][-1], "source": "script"}
    text = " ".join(b.text for b in resp.content if b.type == "text").strip()
    return {"text": text, "source": "llm"}
