Each item has: id, image and claims. Each claim has: cid, statement, quote, context (the sentence where it appears) and basis (how the text presents it: "given" restated from the problem, "figure" presented as drawn, "chosen" a coordinate choice made to compute, "computed" derived by reasoning). The text was written by a model that did not see the image. Open the image first, then judge every claim of that item ONLY against the image. Judge each claim on its own: do not assume the text is right or wrong.

The drawing is not to scale and can be stretched. Never use pixel proportions to decide a shape, a value, equal lengths, tangency or which segment is longer. Use the labels, the marks (right-angle marks, ticks), which points are joined, the order of the vertices, on which side of a line a shape lies, what is inside or outside.

Give two judgments for every claim.

config: the geometry, as if the whole figure could be turned, moved or mirrored. Directions on the page do not count here.
- "true": the image shows it. For coordinates: some turn, move or mirror image of the drawn figure fits them (the labelled values, the order of the vertices, on which side of each line each shape lies, where the right angles are).
- "false": the image contradicts it: another value, another side or vertex, a shape on the other side of a line, inside instead of outside, an element that does not exist, coordinates that no turn or mirror image of the figure can fit.
- "not_visible": presented as given or drawn (basis "given" or "figure") but the image does not show it.
- "deduction": a computed value or property that the image does not show directly and that does not contradict it. A claim with basis "chosen" or "computed" never gets "not_visible".
- "unclear": you cannot tell from the image.

page: directions on the page as drawn (horizontal, vertical, left, right, above, below, top, bottom, clockwise).
- "matches": the claim states a direction on the page and the image agrees.
- "contradicts": the claim states a direction on the page and the image shows another one. For coordinates, x grows to the right and y grows upward.
- "none": the claim states no direction on the page.

image_shows: when config is "false" or "not_visible", or page is "contradicts", what the image shows instead, briefly; otherwise "".

Output: a JSON list, one object per item in batch order:
{"id": "p001-1", "claims": [{"cid": "c001", "config": "true", "page": "none", "image_shows": ""}]}
Every claim of the batch must appear once. Then reply with one line: the number of claims written.
