Each item has: id, image, question, reference_answer (the dataset's answer) and other_answers. other_answers are answers that independent solvers or programs reached; they may be wrong, and the list may be empty.

Decide whether reference_answer is the correct answer to this question for this image.
1. Look at the image and solve the problem carefully yourself, checking each value against the image.
2. verdict:
   - "correct": right, possibly in an equivalent form;
   - "wrong": give corrected_answer;
   - "uncertain": you cannot establish it (too hard, figure unreadable, question ambiguous or incomplete).
3. answer_coherent: "yes" if reference_answer is the kind of answer the question asks for (an option letter that exists among the options, a value of the right type), otherwise "no".
4. note: your reasoning in 1-4 sentences, citing what the image shows.

Output: a JSON list, one object per item in batch order:
{"id": "p001", "verdict": "correct", "corrected_answer": null, "answer_coherent": "yes", "note": "..."}
Then reply with one line: the number of items written.
