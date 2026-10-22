Each item has: id, image, question and solution. The solution was written by a model that did NOT see the image; it read a text description of it. A vision model will be trained to produce this solution while looking at the image, so any statement about the figure that is false or not visible teaches it to invent.

For each item:
1. Look at the image. figure_claims: count the distinct statements in the solution about what the figure shows (shapes, labels, lengths, angles, positions, orientations, which side something is attached to), true or false. Do not count facts only restated from the question text.
2. List every statement about the figure that is:
   - false: it contradicts the image or names elements that do not exist;
   - unverifiable: it is presented as read from the figure, but it is not shown in the figure and not given in the question.
   For each one: quote, copied character by character from the solution (a short exact span, no paraphrase); type ("false" or "unverifiable"); kind; image_shows (what the image shows). kind is one of:
   - "measure": a length, angle, area or other number;
   - "layout": a position, orientation or arrangement stated as what the figure shows (left/right, above/below, horizontal/vertical, inside/outside, which side something is attached to);
   - "frame": a coordinate system, placement or orientation that the solution chooses or assumes in order to compute ("place D at the origin", "let AB be horizontal") and that contradicts the figure. An assumption that agrees with the figure, or that the figure cannot contradict, is not an issue;
   - "existence": an element that is not in the figure;
   - "other".
3. visual_ok: true if no listed issue has a kind other than "frame".
4. frame_ok: true if no listed issue has kind "frame".
5. reasoning_valid: true if the reasoning is mathematically valid given what the image and question actually show, and reaches the final answer without unjustified steps. reasoning_note: why, briefly.
6. relies_on_unseen: true if the solution uses information that is neither in the image nor in the question, for example "the figure says ..." followed by data that is not shown.

Output: a JSON list, one object per item in batch order:
{"id": "f001", "figure_claims": 12, "visual_ok": false, "frame_ok": true, "issues": [{"quote": "...", "type": "false", "kind": "layout", "image_shows": "..."}], "reasoning_valid": true, "reasoning_note": "...", "relies_on_unseen": false}
Then reply with one line: the number of items written.
