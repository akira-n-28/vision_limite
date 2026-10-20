Each item has: id, image (PNG at the exact resolution a model sees in training) and units. The units are the statements of a description of the image written by another model: cid, section and text. Sections summary, components and relations are meant to describe what is drawn. Sections inferred, ambiguities and context hold that model's deductions.
In component names, the text in backticks is the label written in the image for that component; (`None`) means the description says no label is written for it, not that the label reads "None".

For each unit:
1. Split it into its facts. One fact is one checkable property: an element exists, a label, a value, a count, a position, a relation or a color; or one deduction, computation or interpretation.
2. For each fact give:
   - kind: "visible" (a property you should be able to see in the image) or "deduction" (an inference, a computed value, an interpretation, something taken from the problem text, or a claim about what the figure is for);
   - label:
     - "verified": you can see in the image that it is true; for a deduction, it follows correctly from what is visible;
     - "false": the image contradicts it, the element does not exist, or the deduction is wrong given what is visible;
     - "unverifiable": the image does not let you decide (not shown, too small to read, or a deduction that needs information not in the image);
   - reason: a short justification citing what the image shows, at most 15 words for "verified" and 30 otherwise.
Every unit must have at least one fact. Do not merge or skip units.

Output: a JSON list, one object per item in batch order:
{"id": "p001", "units": [{"cid": "c01", "facts": [{"fact": "...", "kind": "visible", "label": "verified", "reason": "..."}]}]}
Include every cid of every item exactly once. Then reply with one line: the number of items written.
