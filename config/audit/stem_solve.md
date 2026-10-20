Each item has: id, image (PNG at the exact resolution a model sees in training), question, check (the fields to fill for this item).

For each item fill only the fields listed in its check:
- solve: solve the problem from the image and the question only.
  - own_answer: your final answer in its shortest form (number, expression, option letter, word); null if you cannot reach an answer you trust.
  - solve_note: 1-2 sentences with the values you read from the image.
  - program: if the answer comes from a calculation, a short Python program that recomputes it, otherwise null. Format: {"inputs": {"name": {"value": "3", "source": "question"}}, "code": "result = ..."}. Every number the calculation uses must be an input: named, value as a string, source "question" if it is written in the question text or "image" if you read it in the figure. The code may use math, fractions and sympy, must use the inputs as variables, and must assign the final answer to result. No other imports, no files, no printing.
- legibility: can the information needed to answer be read at this resolution?
  - "ok";
  - "partial": some labels are hard to read, but what is needed can be read;
  - "illegible": needed labels, numbers or shapes cannot be read.
  - legibility_note: what is or is not readable.
- coherence: do question and image belong together and make sense?
  - "ok";
  - "incoherent": the question refers to things that are not in the image or to a different figure, or needs data that is neither in the image nor in the question;
  - "corrupt": the question text is broken, truncated or garbled, or its options are missing.
  - coherence_note.

Output: a JSON list with one object per item, in batch order, with "id" and the fields of its check, e.g.
{"id": "p001", "own_answer": "12", "solve_note": "...", "program": null, "legibility": "ok", "legibility_note": "...", "coherence": "ok", "coherence_note": "..."}
Then reply with one line: the number of items written.
