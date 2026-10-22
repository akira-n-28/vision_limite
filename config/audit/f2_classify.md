This task has no images: you work on the text only (the image rule of the general rules does not apply here).

Each item has: id, question, solution and list. In the solution every sentence starts with a marker such as ⟦s012⟧; the same marker appears again where the same sentence is repeated. The solution was written by a model that did NOT see the figure. You do not judge whether anything is true: you report what the sentences in list say about the figure.

For every marker in list, write each fact about the figure that this sentence states or relies on:
- values of lengths, angles and other quantities of figure elements, given, read or computed;
- which shapes there are and how they are built: attached to which side, sharing a side, drawn inside or outside, removed or replaced;
- relations without values: adjacent, opposite, parallel, perpendicular, on which side of a line, inside or outside, where a right angle is, which point lies on which line or circle, tangency;
- directions on the page: horizontal, vertical, left, right, above, below, top, bottom, clockwise;
- elements named as present ("the semicircle", "point G"), also when only implied: "the rectangle ABCD" implies that ABCD is a rectangle;
- coordinates, origins or orientations the solution chooses in order to compute.
A sentence with no fact about the figure (only arithmetic on letters with no figure meaning, plans, checks of the answer letter) gets an empty facts list.

For each fact:
- quote: the exact words of this sentence that state it, copied character by character (keep LaTeX as written), under 100 characters;
- statement: the fact in plain self-contained words, with the values and names exactly as this sentence gives them ("AB has length 16", "square CBEF is attached to side CB of ABCD", "DF is vertical"). Never correct a value or a name, even when another sentence says something different;
- kind: "measure", "construction", "relation", "orientation", "existence" or "coordinates";
- basis, how the solution presents it:
  - "given": restated from the question or from the figure text;
  - "figure": presented as what the figure shows (drawn, labelled, placed on the page);
  - "chosen": a coordinate system, origin or orientation picked to compute, or a position that follows only from that choice;
  - "computed": derived by calculation or reasoning;
- stance:
  - "asserted": stated as true;
  - "hypothesis": supposed, considered or asked, and not settled here;
  - "rejected": considered and then explicitly discarded by the solution, in this sentence or later;
- used_in_answer: only for hypothesis and rejected facts: true if the final answer or the final computation relies on it, otherwise false.

Rules:
- Write every marker of list once, in the order of list, also when its facts list is empty. Do not write markers that are not in list.
- Treat each sentence on its own: repeat a fact for every listed sentence that states it (repetitions are merged later), never skip a sentence because its fact came earlier, never merge two sentences.
- At most 8 facts per sentence.

Output: a JSON list, one object per item in batch order:
{"id": "p001-1", "sentences": [{"s": "s012", "facts": [{"quote": "...", "statement": "...", "kind": "measure", "basis": "given", "stance": "asserted"}]}, {"s": "s013", "facts": []}]}
Then reply with one line: the number of sentences written.
