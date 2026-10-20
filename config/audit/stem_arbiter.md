Each item has: id, image, question and disputes. In each dispute, independent auditors disagree on one field. You see their anonymous judgments: value and note, and for description statements the facts they listed with kind, label and reason.

Fields:
- "legibility": "ok" | "partial" | "illegible".
- "coherence": "ok" | "incoherent" | "corrupt".
- "cNN:vis" covers the visible facts of the statement shown; "cNN:ded" covers its deductions.
  - "false": at least one fact of that kind in the statement is false given the image.
  - "not_false": otherwise (true or unverifiable).
- In description statements, the text in backticks after a component name is the label written in the image; (`None`) means no label is written, not a label reading "None".
- "gold": the reference_answer shown. Values "correct" | "wrong" (with corrected_answer) | "uncertain".

For each dispute, look at the image yourself and decide which value the image supports.
- Do not count votes: a judgment is worth only what its note shows to be true in the image.
- If all judgments are wrong, give the right value and set shared_error to true.
- If the image does not let you decide, give the verdict "unclear".

Output: a JSON list, one object per item in batch order:
{"id": "p001", "disputes": [{"field": "c03:vis", "verdict": "false", "shared_error": false, "note": "..."}]}
Add "corrected_answer" for a gold dispute judged wrong. Include every dispute of every item. Then reply with one line: the number of items written.
