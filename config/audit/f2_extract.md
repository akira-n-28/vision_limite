This task has no images: you work on the text only (the image rule of the general rules does not apply here).

Each item has: id, question and solution. The solution was written by a model that did NOT see the figure. Your job is only to list what the solution says about the figure. Do not judge whether anything is true.

For each item, list every distinct statement the solution makes about the figure:
- lengths, angles and other values of figure elements, stated as given, read or derived;
- which shapes there are and where they are attached or share a side;
- positions and orientations (left/right, above/below, horizontal/vertical, inside/outside, top/bottom);
- elements named as present;
- coordinate systems, origins or orientations the solution chooses in order to compute.
Include statements restated from the question or quoted from a text.

Rules:
- One claim per fact. If the same fact is repeated, list it once, at its first occurrence. If the solution elsewhere gives a different value, side or position for the same element, that is a separate claim with its own quote.
- quote: the exact text of the claim, copied character by character from the solution (keep LaTeX as written), under 120 characters, enough to identify the fact.
- statement: the fact in plain self-contained words, e.g. "AB has length 16", "square CBEF is attached to side CB of ABCD", "DF is vertical".
- kind, one of:
  - "measure": a length, angle or other value stated as given or read from the figure;
  - "construction": which shapes there are and where they are attached;
  - "position": position or orientation of elements;
  - "existence": an element named as present;
  - "frame": a coordinate system, origin or orientation chosen to compute;
  - "deduction": a value or property the solution computes from other facts.
- Do not skip a claim because it looks obvious or because it is repeated from the question. At most 60 claims per item; if there are more, prefer distinct elements.

Output: a JSON list, one object per item in batch order:
{"id": "p001", "claims": [{"quote": "...", "statement": "...", "kind": "measure"}]}
Then reply with one line: the number of items written.
