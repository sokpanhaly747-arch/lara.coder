# browser/

Gives the agent eyes on the product it built — problems like broken
layouts, overlapping elements, or dead buttons are not visible from
source code alone.

- `controller.py` — navigate/interact with the running preview (Playwright-based).
- `screenshot.py` — capture + diff screenshots for visual review/testing.
- `dom_inspector.py` — structural checks (missing alt text, empty containers, etc).
