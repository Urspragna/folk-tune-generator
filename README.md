# Folk Tune Generator — GPT-2 → TunesFormer

**M.Sc. Advanced Neural Networks · Universität Rostock**

Generative model for folk tunes in ABC notation. Progressed from fine-tuning a general-purpose GPT-2 (v1) to using TunesFormer — a Transformer pre-trained specifically on ABC-notation folk music (v2).

## Notebooks

| File | Description |
|---|---|
| `v1_gpt2_baseline.ipynb` | Fine-tune GPT-2 (124M) on the Nottingham corpus; custom constrained decoding |
| `v2_tunesformer.ipynb` | TunesFormer with control codes; musical evaluation; comparison vs v1 |

## v1 — GPT-2 fine-tuning

- Downloads the Nottingham folk tune dataset (1,200 tunes, ABC notation)
- Fine-tunes `gpt2` (124M params) with HuggingFace `Trainer` for 6 epochs
- **Held-out perplexity: 2.96**
- Custom `ABCOnlyLogitsProcessor` constrains decoding to 5,459 valid ABC tokens (blocks 89% of GPT-2 vocabulary)
- Compares unconstrained vs constrained generation qualitatively and quantitatively

## v2 — TunesFormer (upgraded)

- Uses [`sander-wood/tunesformer`](https://huggingface.co/sander-wood/tunesformer) — a Transformer pre-trained on 10,000+ folk tunes
- **No additional fine-tuning needed** — model already knows folk music
- Structured control codes specify tune type, meter, and key:
  ```
  <|R:reel|><|M:4/4|><|K:G|><|NBARS:8|>
  ```
- Generates reels, jigs, waltzes, and hornpipes in different keys on demand
- Same constrained decoding applied; now blocks far fewer tokens (domain-specific vocab)
- Musical evaluation: structural validity, melodic diversity score, pitch distribution
- Optional `music21` integration for deeper pitch/interval analysis

## Why TunesFormer beats GPT-2 for this task

| | GPT-2 (v1) | TunesFormer (v2) |
|---|---|---|
| Base training | General English text | Folk tunes in ABC notation |
| Tokenizer | BPE — breaks `M:4/4` into odd subwords | Character-level ABC vocab |
| Training data | Fine-tuned on 1,200 tunes | Pre-trained on 10,000+ tunes |
| Style control | None | Tune type · Meter · Key |
| Constraint overhead | 89% of vocab blocked | Much lower |

## Tech stack

`PyTorch` · `HuggingFace Transformers` · `TunesFormer` · `music21` · `Plotly` · `Python 3`

## How to run

Upload either notebook to [Google Colab](https://colab.research.google.com), select **Runtime → T4 GPU**, then **Run all**.
