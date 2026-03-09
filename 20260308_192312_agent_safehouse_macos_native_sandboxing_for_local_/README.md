# Agent Safehouse

Based on: https://agent-safehouse.dev/

## What This Does

macOS-native sandboxing for local AI agents.

1. Generates macOS Seatbelt (.sb) sandbox profiles from policy rules
2. Trains an ML policy classifier to predict action safety (allow/deny)
3. Verifies sandbox enforcement via real `sandbox-exec` subprocess calls
4. Runs an agentic loop with learned policy enforcement

Before a local agent performs a file/network/exec action, the policy classifier (trained on action features) decides allow/deny — critical actions are double-checked via a real macOS `sandbox-exec` call.

## Run It

```bash
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.10+
- macOS (uses `sandbox-exec`)
- See `requirements.txt` for dependencies
