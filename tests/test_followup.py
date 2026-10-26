"""Run: python3 tests/test_followup.py — a new topic must not be searched with the last one."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.answer import is_followup, search_query

NEW = ["What is figma", "What are AI models And how many types are there",
       "What is figma and how do PMs use it", "What is the first step of design thinking",
       "What is a story point and what is velocity?", "who teaches class 5"]
FOLLOW = ["explain the second one", "how is it different from kanban", "why?", "explain more",
          "give an example", "and what about scrum", "what are its types", "who taught it",
          "Explain this with an example", "what is the difference between these two"]

for q in NEW:
    assert not is_followup(q), q
for q in FOLLOW:
    assert is_followup(q), q

chat = [{"role": "user", "content": "What is a story point and what is velocity?"},
        {"role": "assistant", "content": "…"},
        {"role": "user", "content": "Please summarize this"},
        {"role": "assistant", "content": "…"}]
assert search_query("What is figma", chat) == ("What is figma", False)
assert "story point" in search_query("explain the second one", chat)[0]
print("ok")
