"""Quick routing test for the unified skills.py."""
import sys
from pathlib import Path

# Ensure src is on sys.path
_src = str(Path(__file__).parent.parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import ticket_to_code.llm_utils
if not hasattr(ticket_to_code.llm_utils, "expand_ticket_keywords"):
    ticket_to_code.llm_utils.expand_ticket_keywords = lambda t: {"current_state_literals": [], "desired_state_literals": []}

from ticket_to_code.reasoning.skills import select_skill


def test_skills_routing():
    tests = [
        ("", "Add a new REST API endpoint with swagger docs", "rest_endpoint"),
        ("", "Add a Flyway migration to add a column", "db_migration"),
        ("", "Create a new Angular component in xchange-ui", "ui_component"),
        ("", "Fix NPE regression in saga compensate", "bugfix"),
        ("version_bump", "Bump version from 26.3 to 26.4", "version_change"),
        ("", "Something totally unrelated", "cc4e_default"),
        ("access_control", "Restrict coordinator role for organization", "access_control"),
        ("", "Add RabbitMQ consumer for event stream", "messaging_event"),
        ("", "Add a ConstraintValidator for field length", "validator"),
        ("", "Update application.yaml config for timeout", "configuration"),
    ]

    for ticket_type, text, expected in tests:
        skill = select_skill(ticket_type, text)
        assert skill.name == expected, f"Failed for '{text}': got {skill.name}, expected {expected}"

