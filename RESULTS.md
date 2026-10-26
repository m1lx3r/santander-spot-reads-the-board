# Week 1 results

Fill in your row before the meeting. "Fixed" and "broke" are photo names compared to the starter baseline.

## Baseline (starter script)

Synthetic set (`python evaluate.py whiteboard_starter.py`): board correct 11/16 (2 wrong quad, 3 not found), ink F1 0.989 (glare photos 0.961).
Real photos: board found in __ / __.

| Photo | Board found? | Flat looks right? | Writing boxes OK? | Notes |
|---|---|---|---|---|
| | | | | |

## Prototypes

| Name | Step | Idea | Board found (before → after) | Photos fixed | Photos broken | Keep it? |
|---|---|---|---|---|---|---|
| Mazen | 1. Detect board | Whitest big quad: auto-Canny + CLAHE, score every quad by whiteness, keep largest of ties | 11/16 → 15/16 (held-out 22/32 → 30/32) | far_clean, far_glare, angle30_dim, far_angle45 | none | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

## Decisions (filled in at the meeting)

| Step | Best method | Owner |
|---|---|---|
| 1. Detect board | | |
| 2. Flatten | | |
| 3. Reduce glare | | |
| 4. Find writing | | |
| Integration + report | | |
