Each item has: id, image and claims. Each claim has: cid, statement, quote and context (the sentence of a solution where the claim appears). The solution was written by a model that did not see the image. Open the image first, then judge every claim of that item ONLY against the image. Judge each claim on its own: do not assume the solution is right or wrong.

verdict, one of:
- "true": the image shows it. For a coordinate system or orientation chosen to compute: it agrees with the image, or nothing in the image contradicts it.
- "false": the image contradicts it: a different value, another side, another position or orientation, an element that does not exist, or a chosen coordinate system that contradicts the image.
- "not_visible": it is presented as given or read from the figure, but the image does not show it. Use this only for statements presented as given or read, never for computed values.
- "deduction": a computed value or property that the image does not show directly and that does not contradict what is shown. If it contradicts labelled values, say "false".
- "unclear": you cannot tell from the image.

image_shows: for "false" and "not_visible", what the image shows instead, briefly; otherwise "".

Output: a JSON list, one object per item in batch order:
{"id": "p001", "claims": [{"cid": "c01", "verdict": "true", "image_shows": ""}]}
Every claim of the batch must appear once. Then reply with one line: the number of claims written.
