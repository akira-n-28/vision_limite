# Sonda del contratto di risposta (D3): sonda-1

Job 30910, codice `1addc90cea28eb318f09bf2692139a6b1b88198b` con modifiche, schema `contract-probe/1`. Generato da `scripts/contract_probe.py`: diagnostica CPU col tokenizer dello snapshot, senza pesi. Dimostra proprietà meccaniche del formato, non quale alternativa il modello seguirà meglio. Le alternative e il template della sonda esistono solo nella sonda; nessun contratto è attivato.

## Identità

| file dello snapshot | sha256 |
| --- | --- |
| `tokenizer.json` | `296e081e2f5ecf9d87814aa9b0f4b12d670ed2b2e2be6c84e01a9466c953afb7` |
| `tokenizer_config.json` | `7c9b02a0378ecebbf99549aea37259bc8183df58a70c3d3f564f8ea7d4144681` |
| `special_tokens_map.json` | `6676f091c8bc4d1b50146427cfde92073402866b87b6e39223227931b70083e9` |
| `chat_template.jinja` | `f6bd2b6bf08be63507e9da57900f5b4572be94605074eff65423da5374c48011` |
| `generation_config.json` | `f512b06d847fbceacedadb3414fa9b99f7568c98ecbe3039841ff6cdb32d5bc5` |

- Tokenizer `Qwen2Tokenizer`, 151667 voci, transformers 5.6.2, tokenizers 0.22.2; il tokenizer usa il `chat_template.jinja` dello snapshot: sì.
- Template della sonda `limite-vl-sonda/1`, sha256 `02da63de40f82869c087661810da9c8b0c973344f70e160d6d65be008a4ec1ca`: quello dello snapshot con un system iniziale canonico o VL; testo in `report.json`.
- `generation_config.json`: `eos_token_id` = 151643.

## Intestazioni e system

- Canonica: `<|endoftext|><|im_start|>system\nYou are a helpful assistant.\nPlease reason step by step, and put your final answer within \boxed{}.<|im_end|>\n`, 27 token.
- VL: `<|endoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n`, 12 token.
- Prefisso comune dei due system: 5 token; poi ['.\n', 'Please'] nel canonico e ['.'] nel VL.

## Alternative

| id | fase | regola |
| --- | --- | --- |
| A | F1 storico (decoder congelato); il ragionato è il formato del replay e di F2 storici | template dello snapshot (system canonico); diretta: prefill <think></think> senza label, poi risposta ed EOS con label; ragionata: <think>…</think> e risposta con label |
| B | uso finale, decoder addestrabile | system VL; diretta: <think></think> generato e con label, poi risposta ed EOS; ragionata: <think>…</think> e risposta con label |
| B-F1 | F1, decoder congelato | token di B; il <think></think> della diretta è un prefill dichiarato senza label; un turno, nessun target ragionato |
| C | uso finale, decoder addestrabile | system VL; diretta: risposta ed EOS senza tag; ragionata: <think>…</think> e risposta |
| M | baseline matematica e replay (modalità canonica) | template dello snapshot; istruzione di MATH-500 dopo il problema; sempre <think>…</think> e \boxed{} nella risposta finale |

## Esempi

Testi decodificati con `<|image_pad|>` contratto (×N token immagine); `⟨canonica⟩` e `⟨VL⟩` sono le intestazioni sopra. `prompt` è ciò che riceve il modello in inferenza (prefill compreso), `target` le label dell'ultimo turno; con la cronologia, anche i turni precedenti hanno label.

### latex: trascrizione LaTeX, con immagine

- utente: `Transcribe the formula in the image into LaTeX.` (con immagine)

```
A     prompt  ⟨canonica⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Transcribe the formula in the image into LaTeX.<|im_end|>\n<|im_start|>assistant\n<think></think>
      target  \frac{a+b}{2}\geq\sqrt{ab}<|endoftext|>
      token   intestazione 27, resto del prompt 20 + immagine 64, prefill 2 (senza label), target 16 (con label), cronologia con label 0, totale 129
B     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Transcribe the formula in the image into LaTeX.<|im_end|>\n<|im_start|>assistant\n
      target  <think></think>\frac{a+b}{2}\geq\sqrt{ab}<|endoftext|>
      token   intestazione 12, resto del prompt 20 + immagine 64, prefill 0 (senza label), target 18 (con label), cronologia con label 0, totale 114
B-F1  prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Transcribe the formula in the image into LaTeX.<|im_end|>\n<|im_start|>assistant\n<think></think>
      target  \frac{a+b}{2}\geq\sqrt{ab}<|endoftext|>
      token   intestazione 12, resto del prompt 20 + immagine 64, prefill 2 (senza label), target 16 (con label), cronologia con label 0, totale 114
C     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Transcribe the formula in the image into LaTeX.<|im_end|>\n<|im_start|>assistant\n
      target  \frac{a+b}{2}\geq\sqrt{ab}<|endoftext|>
      token   intestazione 12, resto del prompt 20 + immagine 64, prefill 0 (senza label), target 16 (con label), cronologia con label 0, totale 112
```

### didascalia: didascalia breve, con immagine

- utente: `Describe this image briefly.` (con immagine)

```
A     prompt  ⟨canonica⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Describe this image briefly.<|im_end|>\n<|im_start|>assistant\n<think></think>
      target  A red bicycle leaning against a white brick wall.<|endoftext|>
      token   intestazione 27, resto del prompt 15 + immagine 64, prefill 2 (senza label), target 11 (con label), cronologia con label 0, totale 119
B     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Describe this image briefly.<|im_end|>\n<|im_start|>assistant\n
      target  <think></think>A red bicycle leaning against a white brick wall.<|endoftext|>
      token   intestazione 12, resto del prompt 15 + immagine 64, prefill 0 (senza label), target 13 (con label), cronologia con label 0, totale 104
B-F1  prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Describe this image briefly.<|im_end|>\n<|im_start|>assistant\n<think></think>
      target  A red bicycle leaning against a white brick wall.<|endoftext|>
      token   intestazione 12, resto del prompt 15 + immagine 64, prefill 2 (senza label), target 11 (con label), cronologia con label 0, totale 104
C     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Describe this image briefly.<|im_end|>\n<|im_start|>assistant\n
      target  A red bicycle leaning against a white brick wall.<|endoftext|>
      token   intestazione 12, resto del prompt 15 + immagine 64, prefill 0 (senza label), target 11 (con label), cronologia con label 0, totale 102
```

### numerica: risposta numerica, con immagine

- utente: `Answer the question with a short answer.\nWhat is the length of side AB?` (con immagine)

```
A     prompt  ⟨canonica⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Answer the question with a short answer.\nWhat is the length of side AB?<|im_end|>\n<|im_start|>assistant\n<think></think>
      target  12<|endoftext|>
      token   intestazione 27, resto del prompt 26 + immagine 64, prefill 2 (senza label), target 3 (con label), cronologia con label 0, totale 122
B     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Answer the question with a short answer.\nWhat is the length of side AB?<|im_end|>\n<|im_start|>assistant\n
      target  <think></think>12<|endoftext|>
      token   intestazione 12, resto del prompt 26 + immagine 64, prefill 0 (senza label), target 5 (con label), cronologia con label 0, totale 107
B-F1  prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Answer the question with a short answer.\nWhat is the length of side AB?<|im_end|>\n<|im_start|>assistant\n<think></think>
      target  12<|endoftext|>
      token   intestazione 12, resto del prompt 26 + immagine 64, prefill 2 (senza label), target 3 (con label), cronologia con label 0, totale 107
C     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Answer the question with a short answer.\nWhat is the length of side AB?<|im_end|>\n<|im_start|>assistant\n
      target  12<|endoftext|>
      token   intestazione 12, resto del prompt 26 + immagine 64, prefill 0 (senza label), target 3 (con label), cronologia con label 0, totale 105
```

### scelta_multipla: scelta multipla, con immagine

- utente: `Which shape has the largest area?\nA. the square\nB. the circle\nC. the triangle\nAnswer with the letter of the correct option.` (con immagine)

```
A     prompt  ⟨canonica⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Which shape has the largest area?\nA. the square\nB. the circle\nC. the triangle\nAnswer with the letter of the correct option.<|im_end|>\n<|im_start|>assistant\n<think></think>
      target  B<|endoftext|>
      token   intestazione 27, resto del prompt 41 + immagine 64, prefill 2 (senza label), target 2 (con label), cronologia con label 0, totale 136
B     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Which shape has the largest area?\nA. the square\nB. the circle\nC. the triangle\nAnswer with the letter of the correct option.<|im_end|>\n<|im_start|>assistant\n
      target  <think></think>B<|endoftext|>
      token   intestazione 12, resto del prompt 41 + immagine 64, prefill 0 (senza label), target 4 (con label), cronologia con label 0, totale 121
B-F1  prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Which shape has the largest area?\nA. the square\nB. the circle\nC. the triangle\nAnswer with the letter of the correct option.<|im_end|>\n<|im_start|>assistant\n<think></think>
      target  B<|endoftext|>
      token   intestazione 12, resto del prompt 41 + immagine 64, prefill 2 (senza label), target 2 (con label), cronologia con label 0, totale 121
C     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>Which shape has the largest area?\nA. the square\nB. the circle\nC. the triangle\nAnswer with the letter of the correct option.<|im_end|>\n<|im_start|>assistant\n
      target  B<|endoftext|>
      token   intestazione 12, resto del prompt 41 + immagine 64, prefill 0 (senza label), target 2 (con label), cronologia con label 0, totale 119
```

### spiegazione: problema con spiegazione e risposta finale, solo testo

- utente: `A rectangle has a perimeter of 20 cm and one side of length 3 cm. What is its area?`

```
A     prompt  ⟨canonica⟩<|im_start|>user\nA rectangle has a perimeter of 20 cm and one side of length 3 cm. What is its area?<|im_end|>\n<|im_start|>assistant\n
      target  <think>Half the perimeter is 10 cm, so the other side is 10 - 3 = 7 cm. The area is 3 \times 7 = 21.</think>The other side is 7 cm, so the area is \boxed{21} square centimetres.<|endoftext|>
      token   intestazione 27, resto del prompt 32 + immagine 0, prefill 0 (senza label), target 65 (con label), cronologia con label 0, totale 124
B     prompt  ⟨VL⟩<|im_start|>user\nA rectangle has a perimeter of 20 cm and one side of length 3 cm. What is its area?<|im_end|>\n<|im_start|>assistant\n
      target  <think>Half the perimeter is 10 cm, so the other side is 10 - 3 = 7 cm. The area is 3 \times 7 = 21.</think>The other side is 7 cm, so the area is \boxed{21} square centimetres.<|endoftext|>
      token   intestazione 12, resto del prompt 32 + immagine 0, prefill 0 (senza label), target 65 (con label), cronologia con label 0, totale 109
C     prompt  ⟨VL⟩<|im_start|>user\nA rectangle has a perimeter of 20 cm and one side of length 3 cm. What is its area?<|im_end|>\n<|im_start|>assistant\n
      target  <think>Half the perimeter is 10 cm, so the other side is 10 - 3 = 7 cm. The area is 3 \times 7 = 21.</think>The other side is 7 cm, so the area is \boxed{21} square centimetres.<|endoftext|>
      token   intestazione 12, resto del prompt 32 + immagine 0, prefill 0 (senza label), target 65 (con label), cronologia con label 0, totale 109
M     prompt  ⟨canonica⟩<|im_start|>user\nA rectangle has a perimeter of 20 cm and one side of length 3 cm. What is its area?\nPlease reason step by step, and put your final answer within \boxed{}.<|im_end|>\n<|im_start|>assistant\n
      target  <think>Half the perimeter is 10 cm, so the other side is 10 - 3 = 7 cm. The area is 3 \times 7 = 21.</think>The other side is 7 cm, so the area is \boxed{21} square centimetres.<|endoftext|>
      token   intestazione 27, resto del prompt 47 + immagine 0, prefill 0 (senza label), target 65 (con label), cronologia con label 0, totale 139
```

### dialogo: conversazione con un turno precedente dell'assistente (diretto), poi un turno ragionato

- utente: `What shape is shown in the image?` (con immagine)
- utente: `Its legs are 3 and 4. What is its area?`

```
A     prompt  ⟨canonica⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>What shape is shown in the image?<|im_end|>\n<|im_start|>assistant\n<think></think>A right triangle.<|im_end|>\n<|im_start|>user\nIts legs are 3 and 4. What is its area?<|im_end|>\n<|im_start|>assistant\n
      cronologia con label  <think></think>A right triangle.
      target  <think>The area of a right triangle is half the product of its legs: 3 \times 4 / 2 = 6.</think>The area is \boxed{6}.<|endoftext|>
      token   intestazione 27, resto del prompt 48 + immagine 64, prefill 0 (senza label), target 38 (con label), cronologia con label 6, totale 177
B     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>What shape is shown in the image?<|im_end|>\n<|im_start|>assistant\n<think></think>A right triangle.<|im_end|>\n<|im_start|>user\nIts legs are 3 and 4. What is its area?<|im_end|>\n<|im_start|>assistant\n
      cronologia con label  <think></think>A right triangle.
      target  <think>The area of a right triangle is half the product of its legs: 3 \times 4 / 2 = 6.</think>The area is \boxed{6}.<|endoftext|>
      token   intestazione 12, resto del prompt 48 + immagine 64, prefill 0 (senza label), target 38 (con label), cronologia con label 6, totale 162
C     prompt  ⟨VL⟩<|im_start|>user\n<|vision_start|><|image_pad|>×64<|vision_end|>What shape is shown in the image?<|im_end|>\n<|im_start|>assistant\nA right triangle.<|im_end|>\n<|im_start|>user\nIts legs are 3 and 4. What is its area?<|im_end|>\n<|im_start|>assistant\n
      cronologia con label  A right triangle.
      target  <think>The area of a right triangle is half the product of its legs: 3 \times 4 / 2 = 6.</think>The area is \boxed{6}.<|endoftext|>
      token   intestazione 12, resto del prompt 46 + immagine 64, prefill 0 (senza label), target 38 (con label), cronologia con label 4, totale 160
```

## Token per token

Dall'apertura dell'ultimo turno dell'assistente; per il dialogo dal primo turno dell'assistente. `label` = il token è supervisionato.

### latex, A

```
pos    id      label  token
108    151644  no     '<|im_start|>'
109    77091   no     'assistant'
110    198     no     '\n'
111    151665  no     '<think>'
112    151666  no     '</think>'
113    59      sì     '\\'
114    37018   sì     'frac'
115    90      sì     '{'
116    64      sì     'a'
117    35093   sì     '+b'
118    15170   sì     '}{'
119    17      sì     '2'
120    11035   sì     '}\\'
121    709     sì     'ge'
122    80      sì     'q'
123    59      sì     '\\'
124    26888   sì     'sqrt'
125    90      sì     '{'
126    370     sì     'ab'
127    92      sì     '}'
128    151643  sì     '<|endoftext|>'
```

### latex, B

```
pos    id      label  token
93     151644  no     '<|im_start|>'
94     77091   no     'assistant'
95     198     no     '\n'
96     151665  sì     '<think>'
97     151666  sì     '</think>'
98     59      sì     '\\'
99     37018   sì     'frac'
100    90      sì     '{'
101    64      sì     'a'
102    35093   sì     '+b'
103    15170   sì     '}{'
104    17      sì     '2'
105    11035   sì     '}\\'
106    709     sì     'ge'
107    80      sì     'q'
108    59      sì     '\\'
109    26888   sì     'sqrt'
110    90      sì     '{'
111    370     sì     'ab'
112    92      sì     '}'
113    151643  sì     '<|endoftext|>'
```

### latex, B-F1

```
pos    id      label  token
93     151644  no     '<|im_start|>'
94     77091   no     'assistant'
95     198     no     '\n'
96     151665  no     '<think>'
97     151666  no     '</think>'
98     59      sì     '\\'
99     37018   sì     'frac'
100    90      sì     '{'
101    64      sì     'a'
102    35093   sì     '+b'
103    15170   sì     '}{'
104    17      sì     '2'
105    11035   sì     '}\\'
106    709     sì     'ge'
107    80      sì     'q'
108    59      sì     '\\'
109    26888   sì     'sqrt'
110    90      sì     '{'
111    370     sì     'ab'
112    92      sì     '}'
113    151643  sì     '<|endoftext|>'
```

### latex, C

```
pos    id      label  token
93     151644  no     '<|im_start|>'
94     77091   no     'assistant'
95     198     no     '\n'
96     59      sì     '\\'
97     37018   sì     'frac'
98     90      sì     '{'
99     64      sì     'a'
100    35093   sì     '+b'
101    15170   sì     '}{'
102    17      sì     '2'
103    11035   sì     '}\\'
104    709     sì     'ge'
105    80      sì     'q'
106    59      sì     '\\'
107    26888   sì     'sqrt'
108    90      sì     '{'
109    370     sì     'ab'
110    92      sì     '}'
111    151643  sì     '<|endoftext|>'
```

### dialogo, B

```
pos    id      label  token
91     151644  no     '<|im_start|>'
92     77091   no     'assistant'
93     198     no     '\n'
94     151665  sì     '<think>'
95     151666  sì     '</think>'
96     32      sì     'A'
97     1290    sì     ' right'
98     21495   sì     ' triangle'
99     13      sì     '.'
100    151645  no     '<|im_end|>'
101    198     no     '\n'
102    151644  no     '<|im_start|>'
103    872     no     'user'
104    198     no     '\n'
105    36120   no     'Its'
106    14201   no     ' legs'
107    525     no     ' are'
108    220     no     ' '
109    18      no     '3'
110    323     no     ' and'
111    220     no     ' '
112    19      no     '4'
113    13      no     '.'
114    3555    no     ' What'
115    374     no     ' is'
116    1181    no     ' its'
117    3082    no     ' area'
118    30      no     '?'
119    151645  no     '<|im_end|>'
120    198     no     '\n'
121    151644  no     '<|im_start|>'
122    77091   no     'assistant'
123    198     no     '\n'
124    151665  sì     '<think>'
125    785     sì     'The'
126    3082    sì     ' area'
127    315     sì     ' of'
128    264     sì     ' a'
129    1290    sì     ' right'
130    21495   sì     ' triangle'
131    374     sì     ' is'
132    4279    sì     ' half'
133    279     sì     ' the'
134    1985    sì     ' product'
135    315     sì     ' of'
136    1181    sì     ' its'
137    14201   sì     ' legs'
138    25      sì     ':'
139    220     sì     ' '
140    18      sì     '3'
141    1124    sì     ' \\'
142    15136   sì     'times'
143    220     sì     ' '
144    19      sì     '4'
145    608     sì     ' /'
146    220     sì     ' '
147    17      sì     '2'
148    284     sì     ' ='
149    220     sì     ' '
150    21      sì     '6'
151    13      sì     '.'
152    151666  sì     '</think>'
153    785     sì     'The'
154    3082    sì     ' area'
155    374     sì     ' is'
156    1124    sì     ' \\'
157    79075   sì     'boxed'
158    90      sì     '{'
159    21      sì     '6'
160    7810    sì     '}.'
161    151643  sì     '<|endoftext|>'
```

### dialogo, C

```
pos    id      label  token
91     151644  no     '<|im_start|>'
92     77091   no     'assistant'
93     198     no     '\n'
94     32      sì     'A'
95     1290    sì     ' right'
96     21495   sì     ' triangle'
97     13      sì     '.'
98     151645  no     '<|im_end|>'
99     198     no     '\n'
100    151644  no     '<|im_start|>'
101    872     no     'user'
102    198     no     '\n'
103    36120   no     'Its'
104    14201   no     ' legs'
105    525     no     ' are'
106    220     no     ' '
107    18      no     '3'
108    323     no     ' and'
109    220     no     ' '
110    19      no     '4'
111    13      no     '.'
112    3555    no     ' What'
113    374     no     ' is'
114    1181    no     ' its'
115    3082    no     ' area'
116    30      no     '?'
117    151645  no     '<|im_end|>'
118    198     no     '\n'
119    151644  no     '<|im_start|>'
120    77091   no     'assistant'
121    198     no     '\n'
122    151665  sì     '<think>'
123    785     sì     'The'
124    3082    sì     ' area'
125    315     sì     ' of'
126    264     sì     ' a'
127    1290    sì     ' right'
128    21495   sì     ' triangle'
129    374     sì     ' is'
130    4279    sì     ' half'
131    279     sì     ' the'
132    1985    sì     ' product'
133    315     sì     ' of'
134    1181    sì     ' its'
135    14201   sì     ' legs'
136    25      sì     ':'
137    220     sì     ' '
138    18      sì     '3'
139    1124    sì     ' \\'
140    15136   sì     'times'
141    220     sì     ' '
142    19      sì     '4'
143    608     sì     ' /'
144    220     sì     ' '
145    17      sì     '2'
146    284     sì     ' ='
147    220     sì     ' '
148    21      sì     '6'
149    13      sì     '.'
150    151666  sì     '</think>'
151    785     sì     'The'
152    3082    sì     ' area'
153    374     sì     ' is'
154    1124    sì     ' \\'
155    79075   sì     'boxed'
156    90      sì     '{'
157    21      sì     '6'
158    7810    sì     '}.'
159    151643  sì     '<|endoftext|>'
```

## Controlli

| gruppo | superati | falliti |
| --- | --- | --- |
| fatti | 5 | 0 |
| latex | 47 | 0 |
| didascalia | 47 | 0 |
| numerica | 47 | 0 |
| scelta_multipla | 47 | 0 |
| spiegazione | 46 | 0 |
| dialogo | 39 | 0 |
| rifiuti | 42 | 0 |
| processor operativo | 4 | 0 |
| confine della cronologia | 3 | 0 |
| controlli dei controlli | 2 | 0 |

In tutto 329 su 329; falliti: nessuno. Elenco completo in `report.json` e nel riepilogo JSON.

## Rifiuti delle combinazioni incoerenti

| combinazione | atteso | esito |
| --- | --- | --- |
| sonda: system non ammesso | rifiuto (template) | template |
| snapshot: system VL | rifiuto (template) | template |
| sonda: system non in prima posizione | rifiuto (template) | template |
| A: prefill <think></think> e tag nel contenuto | rifiuto (contratto) | contratto |
| reasoning_content e <think> nel contenuto | rifiuto (contratto) | contratto |
| B: prefill nell'ultimo turno | rifiuto (contratto) | contratto |
| C: prefill nell'ultimo turno | rifiuto (contratto) | contratto |
| B: prefill nel prompt d'inferenza | rifiuto (contratto) | contratto |
| C: prefill nel prompt d'inferenza | rifiuto (contratto) | contratto |
| C: <think></think> vuoto | rifiuto (contratto) | contratto |
| B: risposta diretta senza <think> | rifiuto (contratto) | contratto |
| A: <think></think> vuoto con label nell'ultimo turno | rifiuto (contratto) | contratto |
| M: risposta diretta con <think></think> | rifiuto (contratto) | contratto |
| M: risposta senza tag | rifiuto (contratto) | contratto |
| M: risposta finale senza \boxed{} | rifiuto (contratto) | contratto |
| M: system VL | rifiuto (contratto) | contratto |
| B: system canonico | rifiuto (contratto) | contratto |
| B-F1: target ragionato | rifiuto (contratto) | contratto |
| B-F1: più turni | rifiuto (contratto) | contratto |
| tag nel testo dato al contratto | rifiuto (contratto) | contratto |
| A: tag malformati '</think>12' | rifiuto (contratto) | contratto |
| B: tag malformati '</think>12' | rifiuto (contratto) | contratto |
| B-F1: tag malformati '</think>12' | rifiuto (contratto) | contratto |
| C: tag malformati '</think>12' | rifiuto (contratto) | contratto |
| M: tag malformati '</think>12' | rifiuto (contratto) | contratto |
| A: tag malformati '<think>a</think>b<think>c</think>d' | rifiuto (contratto) | contratto |
| B: tag malformati '<think>a</think>b<think>c</think>d' | rifiuto (contratto) | contratto |
| B-F1: tag malformati '<think>a</think>b<think>c</think>d' | rifiuto (contratto) | contratto |
| C: tag malformati '<think>a</think>b<think>c</think>d' | rifiuto (contratto) | contratto |
| M: tag malformati '<think>a</think>b<think>c</think>d' | rifiuto (contratto) | contratto |
| A: tag malformati 'x<think>a</think>b' | rifiuto (contratto) | contratto |
| B: tag malformati 'x<think>a</think>b' | rifiuto (contratto) | contratto |
| B-F1: tag malformati 'x<think>a</think>b' | rifiuto (contratto) | contratto |
| C: tag malformati 'x<think>a</think>b' | rifiuto (contratto) | contratto |
| M: tag malformati 'x<think>a</think>b' | rifiuto (contratto) | contratto |
| A: tag malformati '<think>a' | rifiuto (contratto) | contratto |
| B: tag malformati '<think>a' | rifiuto (contratto) | contratto |
| B-F1: tag malformati '<think>a' | rifiuto (contratto) | contratto |
| C: tag malformati '<think>a' | rifiuto (contratto) | contratto |
| M: tag malformati '<think>a' | rifiuto (contratto) | contratto |
| processor: token speciale nel testo dell'assistente | rifiuto (processor) | processor |
| processor: prefill in un turno intermedio | rifiuto (processor) | processor |

## Processor operativo: combinazioni accettate oggi

| combinazione | processor operativo | contratto della sonda | coda dell'esempio |
| --- | --- | --- | --- |
| prefill <think></think> e <think></think> nel contenuto | accettato | rifiutato | `<\|im_end\|>\n<\|im_start\|>assistant\n<think></think><think></think>12<\|endoftext\|>` |
| reasoning_content e <think> nel contenuto | accettato | rifiutato | `<\|im_start\|>assistant\n<think>r</think><think>x</think>12<\|endoftext\|>` |
| </think> senza <think> | accettato | rifiutato | `user\nQ<\|im_end\|>\n<\|im_start\|>assistant\n</think>12<\|endoftext\|>` |
| due blocchi <think> | accettato | rifiutato | `<\|im_start\|>assistant\n<think>a</think><think>b</think>12<\|endoftext\|>` |

## Confine fra template e cronologia

| alternativa | turno precedente | atteso | esito |
| --- | --- | --- | --- |
| C | `\n12` | rifiuto del processor: il prefisso non si conserva (\n\n diventa un token) | rifiuto del processor: turno 2: la tokenizzazione del prefisso non è un prefisso di quella completa |
| C | `12` | uguali | uguali |
| B | `<think></think>\n12` | uguali | uguali |

## Costi

- 13.3 s, RSS massima 813.7 MiB; nessun peso, nessun dataset, solo i file del tokenizer e del template.

## Limiti

- proprietà meccaniche del formato (testo, token, label, prefisso, terminatori, rifiuti), non il comportamento del modello: quale alternativa Violetto o il modello VL seguiranno meglio richiede un esperimento GPU autorizzato
- pochi esempi scritti a mano; il segnaposto visivo ha 64 token, il numero reale dipende dalla risoluzione
- il template della sonda e le regole del contratto esistono solo qui: il processor, i loader e le configurazioni operative restano quelli storici (A)
- la baseline canonica è verificata sul dominio condiviso: senza system o col system canonico in prima posizione; un system canonico in un'altra posizione, che lo snapshot accetta e non scrive, la sonda lo rifiuta
- la ritokenizzazione dei turni generati è provata sul testo; i token generati davvero da un modello possono avere una segmentazione diversa da quella canonica del tokenizer
