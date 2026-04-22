import sys
import os
import time
workspace_dir = '/Users/jatinankushnimje/Documents/Coding/driveiq_practice'
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
from backend.routes.coach import _generate_flan_tip, warmup_flan_async, get_coach_status
print("Starting Flan warmup...")
warmup_flan_async()
# Wait for it to become ready
while True:
    status = get_coach_status()
    print(f"Current Flan status: {status['status']}")
    if status['status'] in ('ready', 'failed', 'disabled'):
        break
    time.sleep(2)
if status['status'] == 'ready':
    score = 45.0
    features = {
        "braking_flag": 1,
        "speed": 115,
        "proximity_score": 0.2
    }
    history_summary = "Has a history of harsh braking."
    
    print("\n--- Generating Tip ---")
    tip, debug_reason = _generate_flan_tip(score, features, history_summary)
    
    print(f"\nResult Tip: {tip}")
    print(f"Debug Reason: {debug_reason}")
else:
    print("\nFlan model failed to load or is disabled.")
    print(status)