Each item has: id, image, question, solution and disputes. Each dispute is on one field: visual_ok, frame_ok, reasoning_valid or relies_on_unseen. It shows the anonymous judgments of independent auditors (value true or false, and note).

The solution was written by a model that did not see the image. The fields mean:
- visual_ok: no statement about what the figure shows is false or unverifiable (coordinate systems or orientations the solution assumes in order to compute are judged by frame_ok, not here);
- frame_ok: no coordinate system, placement or orientation assumed by the solution contradicts the figure;
- reasoning_valid: the reasoning is valid given what the image and question show;
- relies_on_unseen: the solution uses information that is neither in the image nor in the question.

Decide each dispute from the image and the solution: "true" | "false" | "unclear".
- Do not count votes: a judgment is worth only what its note shows to be true.
- If all judgments are wrong, give the right value and set shared_error to true.
- When you decide "false" for visual_ok or frame_ok, put in the note the exact short text of the solution you judge false, copied character by character.

Output: a JSON list, one object per item in batch order:
{"id": "f001", "disputes": [{"field": "visual_ok", "verdict": "false", "shared_error": false, "note": "..."}]}
Then reply with one line: the number of items written.
