Each item has: id, image, question and solution. The solution was written by a model that did NOT see the image; it read a text description of it. A vision model will be trained to produce this solution while looking at the image, so any statement about the figure that is false or not visible teaches it to invent.

For each item:
1. Look at the image. List every statement in the solution about what the figure shows that is:
   - false: it contradicts the image or names elements that do not exist;
   - unverifiable: it is presented as read from the figure, but it is not shown in the figure and not given in the question.
   Quote the exact short text and say what the image shows.
2. visual_ok: true if the list is empty.
3. reasoning_valid: true if the reasoning is mathematically valid given what the image and question actually show, and reaches the final answer without unjustified steps. reasoning_note: why, briefly.
4. relies_on_unseen: true if the solution uses information that is neither in the image nor in the question, for example "the figure says ..." followed by data that is not shown.

Output: a JSON list, one object per item in batch order:
{"id": "f001", "visual_ok": false, "issues": [{"quote": "...", "type": "false", "image_shows": "..."}], "reasoning_valid": true, "reasoning_note": "...", "relies_on_unseen": false}
Then reply with one line: the number of items written.
