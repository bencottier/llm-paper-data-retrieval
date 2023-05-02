# 2023-Apr-27

Notes on the initial result:

- Most of the answers are "N/A"
- Some of the answers are correctly formatted but wrong (e.g. TPU v3 for GPT-4)
- Some of the answers seem like hallucinations (e.g. "1.45e+13 FLOP/s" for https://arxiv.org/pdf/2110.08554.pdf - I can't find such a number anywhere in the paper using Ctrl+F)
- Some of the answers are incorrectly formatted (e.g. "1. N/A" for Switch Transformers)
- Some values are NaN which suggests the answer was the empty string, or something else went wrong

# 2023-Apr-28

Fixing formatting issues that result from my post-processing

- Output can start with a new line so I added `strip()`
- Added clearer instructions about using "N/A" for each answer in the prompt template

Retrying the first 10 entries

- All N/A except for PaLM hardware model and FLOP/s
  - Hardware model is correct
  - Hmm, I would have expected it to retrieve the number of chips. It's in the Abstract. Maybe it struggles with remembering that far back in the context?
    - Maybe I should mention "chips" in the prompt. But I worry this could cause more confusion overall.
  - But the FLOP/s is interesting. 127e12 FLOP/s.
    - Ctrl+F for "127" comes up with nothing.
    - Ctrl+F for "FLOP" comes up with no relevant numbers right next to it.
    - But 127e12 / 275e12 is 46.2%. Which is the reported model FLOPs utilization of PaLM 540B.
    - Did GPT figure this out from the reported utilization!? That would be impressive.

Prompt changes

- Include "or chips" in question 1.
- Say that the example answers are just to show the format.

Result for PaLM:

- Now they're all N/A! Damn.
- Rerun
  - "N/A	TPUv4	2% in model FLOPs utilization"
  - Huh. So it ain't deterministic. That's annoying.
- Rerun
  - All N/A
- Rerun
  - All N/A
- Remove the "example answers are just to show the format" part
  - "N/A	N/A	127e12 FLOPS"
  - Rerun
    - Same

Now running on all post-2017 papers while I work on other things.

- Didn't work because I ran out of my free quota.
