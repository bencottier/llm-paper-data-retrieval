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

# 2023-May-05

Experimenting with ChatGPT (GPT-4) as proof-of-concept for GPT-4

- Input: prompt + excerpt of paper
  - The excerpt of the paper contains the relevant info, and is roughly the maximum length that I can input in ChatGPT

## Experiment with GPT-4 and GPT-3.5 chat mode

### Prompt

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

{paper_text}

====

```

### PaLM

Input:

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

PaLM: Scaling Language Modeling with Pathways

Aakanksha Chowdhery∗ Sharan Narang∗ Jacob Devlin∗
Maarten Bosma Gaurav Mishra Adam Roberts Paul Barham

Hyung Won Chung Charles Sutton

Sebastian Gehrmann Parker Schuh Kensen Shi

Sasha Tsvyashchenko

Joshua Maynez Abhishek Rao† Parker Barnes Yi Tay

Noam Shazeer‡ Vinodkumar Prabhakaran Emily Reif Nan Du Ben Hutchinson

Reiner Pope

James Bradbury

Jacob Austin Michael Isard Guy Gur-Ari

Pengcheng Yin Toju Duke Anselm Levskaya

Sanjay Ghemawat

Sunipa Dev

Henryk Michalewski Xavier Garcia Vedant Misra Kevin Robinson Liam Fedus
Denny Zhou Daphne Ippolito David Luan‡ Hyeontaek Lim Barret Zoph

Alexander Spiridonov Ryan Sepassi David Dohan

Shivani Agrawal Mark Omernick

Andrew M. Dai Thanumalayan Sankaranarayana Pillai Marie Pellat Aitor Lewkowycz
Erica Moreira Rewon Child Oleksandr Polozov† Katherine Lee Zongwei Zhou
Xuezhi Wang Brennan Saeta Mark Diaz Orhan Firat Michele Catasta† Jason Wei

Kathy Meier-Hellstern Douglas Eck

Jeﬀ Dean

Slav Petrov Noah Fiedel

Google Research

Abstract

Large language models have been shown to achieve remarkable performance across a variety of natural
language tasks using few-shot learning, which drastically reduces the number of task-speciﬁc training
examples needed to adapt the model to a particular application. To further our understanding of the
impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer
language model, which we call Pathways Language Model (PaLM).

We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly eﬃcient
training across multiple TPU Pods. We demonstrate continued beneﬁts of scaling by achieving state-of-
the-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a
number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the ﬁnetuned state-
of-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the
recently released BIG-bench benchmark. A signiﬁcant number of BIG-bench tasks showed discontinuous
improvements from model scale, meaning that performance steeply increased as we scaled to our largest
model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we
demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias
and toxicity, and study the extent of training data memorization with respect to model scale. Finally,
we discuss the ethical considerations related to large language models and discuss potential mitigation
strategies.

2
2
0
2

t
c
O
5

]
L
C
.
s
c
[

5
v
1
1
3
2
0
.
4
0
2
2
:
v
i
X
r
a

∗Equal Contribution. Author contributions and ordering details are listed in Appendix A.

Correspondence authors: chowdhery@google.com, sharannarang@google.com
In addition to other contributions, the last ﬁve authors advised the overall project.

†Alphabet, X, the Moonshot Factory
‡Work done while at Google

1

 
 
 
 
 
 
Contents

1 Introduction

2 Model Architecture

2.1 Model Scale Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.2 Model Card . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

3 Training Dataset

4 Training Infrastructure

4.1 Training Eﬃciency . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

5 Training Setup

5.1 Training Instability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

6 Evaluation

6.1 English NLP tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6.2 BIG-bench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6.3 Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6.4 Code Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6.5 Translation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6.6 Multilingual Natural Language Generation . . . . . . . . . . . . . . . . . . . . . . . .
6.7 Multilingual Question Answering . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6.8 Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

7 Memorization

8 Dataset Contamination

9 Exploring Explanations

10 Representational Bias Analysis

10.1 Distributional bias in social groups . . . . . . . . . . . . . . . . . . . . . . . . . . . .
10.2 Toxicity in open-ended generation . . . . . . . . . . . . . . . . . . . . . . . . . . . .
10.3 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

11 Ethical Considerations

12 Related Work

13 Open Questions in Scaling

14 Conclusion

15 Acknowledgments

A Contributions

B Compute Usage and Environmental Impact

C Dataset Analysis

D Datasheet

E Model Card

F Training for longer

G Sample Model Outputs

G.1 Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
G.2 Bias and Toxicity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
G.3 Memorization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

H Additional Results

H.1 English NLP tasks on smaller models . . . . . . . . . . . . . . . . . . . . . . . . . . .
H.2 Additional BIG-bench results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
H.3 Additional Multilingual NLG results . . . . . . . . . . . . . . . . . . . . . . . . . . .

2

3

5
6
6

6

7
9

10
11

11
11
14
20
22
28
30
33
33

35

36

38

40
40
43
45

45

47

48

50

51

64

66

67

69

74

76

78
78
79
82

83
83
84
86

1

Introduction

Over the last few years, very large neural networks trained for language understanding and generation have
achieved astonishing results across a wide array of tasks. Many of these models, such as BERT (Devlin
et al., 2019) and T5 (Raﬀel et al., 2020), are trained with encoder-only or encoder-decoder architectures
using an inﬁlling (“masked LM” or “span corruption”) pre-training objective on a large corpus of text, and
are typically then adapted to one or more speciﬁc tasks through ﬁnetuning. Although these models have
achieved near universal state of the art across thousands of natural language tasks, the downside is that they
require a signiﬁcant number of task-speciﬁc training examples to ﬁnetune the model. Additionally, at least a
portion of the model parameters must be updated to ﬁt the task, adding complexity from model ﬁnetuning
and deployment.

GPT-3 (Brown et al., 2020) demonstrated that extremely large autoregressive language models (LMs) can
be used for few-shot predictions, where the model is only given a natural language task description and
(optionally) a handful of exemplars demonstrating how the task should be completed. This class of models
is trained with a decoder-only architecture and a standard left-to-right language modeling objective on a
large text corpus, where the objective is to predict the next token given the preceding tokens in the example.
Few-shot evaluation has been shown to achieve very strong results without the need for large-scale task-speciﬁc
data collection or model parameter updating.

Since GPT-3, a number of other large autoregressive language models have been developed which have
continued to push the state of the art forward. The most powerful of these post-GPT-3 models are GLaM (Du
et al., 2021), Gopher (Rae et al., 2021), Chinchilla (Hoﬀmann et al., 2022), Megatron–Turing NLG (Smith
et al., 2022), and LaMDA (Thoppilan et al., 2022), all of which achieved few-shot state-of-the-art results on a
signiﬁcant number of tasks at the time of their release.1 Like GPT-3, these models are all variants of the
Transformer architecture (Vaswani et al., 2017). The improvements in these models have primarily come
from one or more of the following approaches: (1) scaling the size of the models in both depth and width;
(2) increasing the number of tokens that the model was trained on; (3) training on cleaner datasets from
more diverse sources; and (4) increasing model capacity without increasing the computational cost through
sparsely activated modules.

In this work, we continue the scaling line of language modeling improvements and train a 540 billion parameter,
densely activated, autoregressive Transformer on 780 billion tokens of high-quality text. This was achieved
through the use of Pathways (Barham et al., 2022), a new ML system which enables highly eﬃcient training
of very large neural networks across thousands of accelerator chips, including those spanning multiple Tensor
Processing Units (TPU) v4 Pods. This new model, called Pathways Language Model (PaLM), achieves
state-of-the-art few-shot results across hundreds of natural language, code, and mathematical reasoning tasks.
We achieve breakthrough performance on a number of these tasks.

The key takeaways from this work are as follows:

• Eﬃcient scaling – We demonstrate the ﬁrst large-scale use of Pathways (Barham et al., 2022) – a new
ML system which enables training a single model across thousands or tens of thousands of accelerator
chips in a highly eﬃcient manner. With Pathways, we trained a 540B parameter language model on
6144 TPU v4 chips at eﬃciency levels that could not be reached before for models of this scale. Most
previous large language models were either trained on a single TPU system (Du et al., 2021; Thoppilan
et al., 2022) or used pipeline parallelism (Huang et al., 2019) to scale across GPU clusters (Smith et al.,
2022) or multiple TPU v3 pods (Rae et al., 2021), with a maximum scale of 4096 TPU v3 chips. In
Section 4, we describe how we were able to scale pipeline-free training of PaLM 540B to 6144 chips
across two TPU v4 Pods while achieving very high eﬃciency of 46.2% in model FLOPs utilization
(observed throughput relative to theoretical max throughput) and 57.8% in hardware FLOPs utilization.

====

```

GPT-3.5 output:

```
1. 6144
2. TPU v4
```

Correct.

GPT-4 output:

```
1. 6144
2. TPU v4
```

Correct.

### AlexaTM

Input:

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

2 Model Architecture

For AlexaTM 20B, we used the standard Transformer model architecture (Vaswani et al., 2017) with learnable
positional embeddings with the small modiﬁcation of moving the layernorms (both in the encoder and the
decoder) to be located exactly at the beginning of each layer (right after the skip connection) instead of at
the end of each layer (i.e., Pre-LN). This modiﬁcation has been demonstrated to improve the stability of the
training, especially for large models (Shoeybi et al., 2019; Xiong et al., 2020). Table 1 shows the detailed
hyper-parameters of the AlexaTM 20B’s architecture.

Model

Encoder Layers Decoder Layers # of Heads

dmodel

# of Parameters
(in billions)

AlexaTM 20B

46

32

32

4096

19.75

Table 1: Model architecture details. The feed-forward size dﬀ is 4 × dmodel.

3 Training Data Preparation

3.1 Datasets

The pre-training data consists of Wikipedia and mC4 (Xue et al., 2021) datasets. We use the data in 12
languages, namely, Arabic, English, French, German, Hindi, Italian, Japanese, Marathi, Portuguese, Spanish,
Tamil, and Telugu. We pack sequences of tokens to produce sequences of approximately 1024 subword units.
We allow unrelated content to be packed together in the same sequence but we separate them with a special
symbol ([DOC]). Maintaining a relatively constant number of subword sequences reduces padding and results
in eﬃcient compute.

Dataset

Sampling

Final Tokens Train Set Percent

Wikipedia

mC4

α = 0.5 lang sampling, then
7 × spoken form plus
3 × written form
α = 0.5 lang sampling, then
0.7 × spoken form plus
0.3 × written form

119B

1.2T

9%

91%

Table 2: Source datasets and sampling methods for the training set.

Since our objective is to allow AlexaTM 20B to act on both spoken queries and written text, we use a written
to spoken formatter for all languages to format the data into spoken format (remove capitalization, remove
punctuations, etc.). We include more spoken than written text to satisfy our internal use cases. The ﬁnal
training set, described in Table 2, is developed by combining data from various sources using three types of
mixing:

• Frequency based upsampling which helps increase the representation of under-represented languages
In particular, we sample sentences according to a multinomial

following Conneau et al. (2020).
distribution with probabilities (q1, q2, . . . , qN ), where:

qi =

pα
i
j=1 pα
j

(cid:80)N

, pi =

ni
j=1 nj

(cid:80)N

3

(1)

Figure 1: AlexaTM 20B pre-training objectives. During pre-training the model is trained on the denoising
task 80% and on the Causal Language Modeling 20% of the time. pre-training data consists of Wikipedia
and mC4 datasets in 12 languages as speciﬁed in Section 3.

in which N is the total number of languages and ni is the total number of sentences in language i (we
set α = 0.5).

• Upsampling Wikipedia data (which has a higher quality) by 10 to be represented more in all data.

• Scaling to favor spoken format over written 7 to 3.

We followed Brown et al. (2020b) and ﬁltered out known benchmarks from the training data by checking for
13-gram overlaps between each sentence and the benchmarks (for the list of ﬁltered datasets see Appendix C).

3.2 Subword Tokenizer

We use SentencePiece (SP) (Kudo & Richardson, 2018) to tokenize the input. We trained a unigram-based
model of SP to determine a set of sub-words that best represents the training data. Our ﬁnal choice was
to train a 150K unigram sentencepiece model from a 7 to 3 mixture ratio of spoken to written data. We
reserve 1K vocabulary entries to be used for tags in downstream tasks (e.g., parse nodes in semantic parsing
task). We also manually augment the vocabulary with a few entries to guarantee better character and word
coverage.

4 Training Setup

AlexaTM 20B model class is derived from BART (Lewis et al., 2020) class implementation in Huggingface (Wolf
et al., 2019) allowing us to beneﬁt from the generate function built in the parent class for inference. To
train the model, we used a denoising objective in which we drop 15% of the tokens in the input (in spans
of length determined by a Poisson distribution with mean equal to 3) and expect the model to reconstruct
the input. We do not add any mask tokens in the input during training: 1) to have the most consistency
during pre-training, inference, and ﬁne-tuning (i.e., no mask tokens appear in any setting), 2) to require
the decoder to play a more active role during pre-training, and 3) to leverage the 10B encoder that we had
trained previously (FitzGerald et al., 2022) to initialize AlexaTM 20B’s encoder (adding [MASK] would have
made decoder job easy given encoder’s ability in “unmasking”).

To make our model more eﬃcient for in-context learning, we added an extra Causal Language Modeling
(CLM) task to the 20B training for 20% of the time.1 In this task, the model is required to continue the
input instead of denoising the input. The model will know to do CLM based on a special token that we add
to the beginning of the sentence ([CLM]). For the CLM task, we only feed a single document (instead of
concatenation of multiple documents) and give 20% to 80% of the document randomly (uniformly) as the
input to the model (so the model learns to continue both from long inputs and short inputs).

1The CLM using seq2seq models has also been called Preﬁx Language Modeling (PLM) (Raﬀel et al., 2020). However, since
the term “preﬁx” has been excessively used recently in other context (Li & Liang, 2021), we prefer to use CLM to avoid any
confusions.

4

To speed up the model training, we initialized the encoder by an internal 10B pre-trained encoder FitzGerald
et al. (2022) (we also initialize the decoder embeddings and LM head embeddings with the embedding from
encoder but we do not tie any pair of embeddings). During training, we initially keep the encoder frozen (for
around 100k updates) but unfreeze the encoder to train the model end to end.

We trained AlexaTM 20B for 120 days on 128 A100 GPUs for the total of 500k updates with the accumulated
batch size of 2 million tokens (total of 1 trillion token updates). We used Adam optimizer (Kingma & Ba,
2015) with lr = 1e−4 with linear decay to lr = 5e−6 over 500k updates. We used weight decay of 0.1 on all
parameters except biases and layernorms. Finally, we trained the model in BFloat16 which helped with the
stability of training (Raﬀel et al., 2020).

We used DeepSpeed’s ZeRO Stage 3 (Rasley et al., 2020) to partition model weights, optimizer states, and
gradients across all GPU workers, allowing us to train the model with high throughput. We relied on an
internal and optimized version of DeepSpeed that we have since open-sourced (Chiu & Zheng, 2022) to obtain
training throughput of up to 154 TFLOPS/GPU on 16 AWS p4d.24xlarge compute instances.

5 Evaluation Setups

We evaluate AlexaTM 20B both using zero/few-shot in-context learning as well as by ﬁnetuning the model
on selected generation tasks. In all few-shot learning settings, we use greedy search.

5.1 Few-shot Learning

Since AlexaTM 20B is trained both on denoising and CLM tasks, both of these modes can be used for
in-context learning. In this subsection, we describe some of the techniques that we used for in-context learning
using the model.

For some tasks, instead of asking the model to generate answers, we present multiple inputs to the model
(corresponding to task labels/choices) and calculate the model’s score for each input. We refer to this as
scoring. Speciﬁcally, we provide the encoder and decoder inputs to AlexaTM 20B (e.g. see Table A3) and
compute the cross-entropy loss in the decoder with teacher-forcing for each set of inputs. The model predicts
the task label or choice with the best score.

====

```

GPT-3.5 output:

```
1. 128
2. A100
```

Correct.

GPT-4 output:

```
1. 128
2. A100
```

Correct.

### High-resolution image synthesis

Input

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

mainly coincide, except for the ImageNet and LSUN-Bedrooms datasets, where we notice slightly varying scores of 7.76
(torch-fidelity) vs. 7.77 (Nichol and Dhariwal) and 2.95 vs 3.0. For the future we emphasize the importance of a
uniﬁed procedure for sample quality assessment. Precision and Recall are also computed by using the script provided by
Nichol and Dhariwal.

E.3.2 Text-to-Image Synthesis

Following the evaluation protocol of [66] we compute FID and Inception Score for the Text-to-Image models from Tab. 2 by
comparing generated samples with 30000 samples from the validation set of the MS-COCO dataset [51]. FID and Inception
Scores are computed with torch-fidelity.

E.3.3 Layout-to-Image Synthesis

For assessing the sample quality of our Layout-to-Image models from Tab. 9 on the COCO dataset, we follow common
practice [37, 87, 89] and compute FID scores the 2048 unaugmented examples of the COCO Segmentation Challenge split.
To obtain better comparability, we use the exact same samples as in [37]. For the OpenImages dataset we similarly follow
their protocol and use 2048 center-cropped test images from the validation set.

E.3.4 Super Resolution

We evaluate the super-resolution models on ImageNet following the pipeline suggested in [72], i.e. images with a shorter
size less than 256 px are removed (both for training and evaluation). On ImageNet, the low-resolution images are produced
using bicubic interpolation with anti-aliasing. FIDs are evaluated using torch-fidelity [60], and we produce samples
on the validation split. For FID scores, we additionally compare to reference features computed on the train split, see Tab. 5
and Tab. 11.

E.3.5 Efﬁciency Analysis

For efﬁciency reasons we compute the sample quality metrics plotted in Fig. 6, 17 and 7 based on 5k samples. Therefore,
the results might vary from those shown in Tab. 1 and 10. All models have a comparable number of parameters as provided
in Tab. 13 and 14. We maximize the learning rates of the individual models such that they still train stably. Therefore, the
learning rates slightly vary between different runs cf . Tab. 13 and 14.

E.3.6 User Study

For the results of the user study presented in Tab. 4 we followed the protocoll of [72] and and use the 2-alternative force-choice
paradigm to assess human preference scores for two distinct tasks. In Task-1 subjects were shown a low resolution/masked
image between the corresponding ground truth high resolution/unmasked version and a synthesized image, which was gen-
erated by using the middle image as conditioning. For SuperResolution subjects were asked: ’Which of the two images is a
better high quality version of the low resolution image in the middle?’. For Inpainting we asked ’Which of the two images
contains more realistic inpainted regions of the image in the middle?’. In Task-2, humans were similarly shown the low-
res/masked version and asked for preference between two corresponding images generated by the two competing methods.
As in [72] humans viewed the images for 3 seconds before responding.

27

F. Computational Requirements

Method

LSUN Churches 2562

StyleGAN2 [42]†
LDM-8 (ours, 100 steps, 410K)

LSUN Bedrooms 2562

ADM [15]† (1000 steps)
LDM-4 (ours, 200 steps, 1.9M)

CelebA-HQ 2562

LDM-4 (ours, 500 steps, 410K)

FFHQ 2562

StyleGAN2 [42]
LDM-4 (ours, 200 steps, 635K)

ImageNet 2562

VQGAN-f-4 (ours, ﬁrst stage)
VQGAN-f-8 (ours, ﬁrst stage)
BigGAN-deep [3]†
ADM [15] (250 steps) †
ADM-G [15] (25 steps) †
ADM-G [15] (250 steps)†
ADM-G,ADM-U [15] (250 steps)†
LDM-8-G (ours, 100, 2.9M)
LDM-8 (ours, 200 ddim steps 2.9M, batch size 64)
LDM-4 (ours, 250 ddim steps 178K, batch size 1200)
LDM-4-G (ours, 250 ddim steps 178K, batch size 1200, classiﬁer-free guidance [32] scale 1.25)
LDM-4-G (ours, 250 ddim steps 178K, batch size 1200, classiﬁer-free guidance [32] scale 1.5)

Generator Classiﬁer
Compute

Compute Compute Throughput∗

Overall

Inference

Nparams

FID↓

IS↑

Precision↑ Recall↑

64
18

232
60

14.4

32.13‡
26

29
66

128-256
916
916
916
329
79
79
271
271
271

-
-

-
-

-

-
-

-
-

-
46
46
30
12
-
-
-
-

64
18

232
55

-
6.80

0.03
1.07

59M
256M

3.86
4.02

552M
274M

1.9
2.95

14.4

0.43

274M

5.11

32.13†
26

29
66

128-256
916
962
962
349
91
79
271
271
271

-
1.07

-
-

-
0.12
0.7
0.07
n/a
1.93
1.9
0.7
0.4
0.4

59M
274M

3.8
4.98

55M
68M

340M
554M
608M
608M
n/a
506M
395M
400M
400M
400M

0.58††
1.14††

6.95
10.94
5.58
4.59
3.85
8.11
17.41
10.56
3.95
3.60

-
-

-
-

-

-
-

-
-

203.6±2.6
100.98
-
186.7
221.72
190.4±2.6
72.92
103.49±1.24
178.22±2.43
247.67±5.59

-
0.64

0.66
0.66

-
0.52

0.51
0.48

0.72

0.49

-
0.73

-
-

0.87
0.69
0.81
0.82
0.84
0.83
0.65
0.71
0.81
0.87

-
0.50

-
-

0.28
0.63
0.49
0.52
0.53
0.36
0.62
0.62
0.55
0.48

Table 18. Comparing compute requirements during training and inference throughput with state-of-the-art generative models. Compute
during training in V100-days, numbers of competing methods taken from [15] unless stated differently;∗: Throughput measured in sam-
ples/sec on a single NVIDIA A100;†: Numbers taken from [15] ;‡: Assumed to be trained on 25M train examples; ††: R-FID vs. ImageNet
validation set

In Tab 18 we provide a more detailed analysis on our used compute ressources and compare our best performing models
on the CelebA-HQ, FFHQ, LSUN and ImageNet datasets with the recent state of the art models by using their provided
numbers, cf . [15]. As they report their used compute in V100 days and we train all our models on a single NVIDIA A100
GPU, we convert the A100 days to V100 days by assuming a ×2.2 speedup of A100 vs V100 [74]4. To assess sample quality,
we additionally report FID scores on the reported datasets. We closely reach the performance of state of the art methods as
StyleGAN2 [42] and ADM [15] while signiﬁcantly reducing the required compute resources.

4This factor corresponds to the speedup of the A100 over the V100 for a U-Net, as deﬁned in Fig. 1 in [74]

28

G. Details on Autoencoder Models

We train all our autoencoder models in an adversarial manner following [23], such that a patch-based discriminator Dψ
is optimized to differentiate original images from reconstructions D(E(x)). To avoid arbitrarily scaled latent spaces, we
regularize the latent z to be zero centered and obtain small variance by introducing an regularizing loss term Lreg.
(i) a low-weighted Kullback-Leibler-term between qE (z|x) =
We investigate two different regularization methods:
N (z; Eµ, Eσ2) and a standard normal distribution N (z; 0, 1) as in a standard variational autoencoder [46, 69], and, (ii) regu-
larizing the latent space with a vector quantization layer by learning a codebook of |Z| different exemplars [96].
To obtain high-ﬁdelity reconstructions we only use a very small regularization for both scenarios, i.e. we either weight the
KL term by a factor ∼ 10−6 or choose a high codebook dimensionality |Z|.

====
```

GPT-3.5 output:

```
1. N/A
2. A100, V100, BigGAN-deep, ADM, ADM-G, ADM-U
```

Incorrect.

GPT-4 output:

```
1. N/A
2. A100, V100, BigGAN-deep
```

Incorrect.

Correct answer:

```
1. 1
2. A100
```

### GPT-4 technical report

Input:


```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

3
2
0
2

r
a

M
7
2

]
L
C
.
s
c
[

3
v
4
7
7
8
0
.
3
0
3
2
:
v
i
X
r
a

GPT-4 Technical Report

OpenAI∗

Abstract

We report the development of GPT-4, a large-scale, multimodal model which can
accept image and text inputs and produce text outputs. While less capable than
humans in many real-world scenarios, GPT-4 exhibits human-level performance
on various professional and academic benchmarks, including passing a simulated
bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-
based model pre-trained to predict the next token in a document. The post-training
alignment process results in improved performance on measures of factuality and
adherence to desired behavior. A core component of this project was developing
infrastructure and optimization methods that behave predictably across a wide
range of scales. This allowed us to accurately predict some aspects of GPT-4’s
performance based on models trained with no more than 1/1,000th the compute of
GPT-4.

1

Introduction

This technical report presents GPT-4, a large multimodal model capable of processing image and
text inputs and producing text outputs. Such models are an important area of study as they have the
potential to be used in a wide range of applications, such as dialogue systems, text summarization,
and machine translation. As such, they have been the subject of substantial interest and progress in
recent years [1–34].

One of the main goals of developing such models is to improve their ability to understand and generate
natural language text, particularly in more complex and nuanced scenarios. To test its capabilities
in such scenarios, GPT-4 was evaluated on a variety of exams originally designed for humans. In
these evaluations it performs quite well and often outscores the vast majority of human test takers.
For example, on a simulated bar exam, GPT-4 achieves a score that falls in the top 10% of test takers.
This contrasts with GPT-3.5, which scores in the bottom 10%.

On a suite of traditional NLP benchmarks, GPT-4 outperforms both previous large language models
and most state-of-the-art systems (which often have benchmark-speciﬁc training or hand-engineering).
On the MMLU benchmark [35, 36], an English-language suite of multiple-choice questions covering
57 subjects, GPT-4 not only outperforms existing models by a considerable margin in English, but
also demonstrates strong performance in other languages. On translated variants of MMLU, GPT-4
surpasses the English-language state-of-the-art in 24 of 26 languages considered. We discuss these
model capability results, as well as model safety improvements and results, in more detail in later
sections.

This report also discusses a key challenge of the project, developing deep learning infrastructure and
optimization methods that behave predictably across a wide range of scales. This allowed us to make
predictions about the expected performance of GPT-4 (based on small runs trained in similar ways)
that were tested against the ﬁnal run to increase conﬁdence in our training.

Despite its capabilities, GPT-4 has similar limitations to earlier GPT models [1, 37, 38]: it is not fully
reliable (e.g. can suffer from “hallucinations”), has a limited context window, and does not learn

∗Please cite this work as “OpenAI (2023)". Full authorship contribution statements appear at the end of the

document. Correspondence regarding this technical report can be sent to gpt4-report@openai.com

 
 
 
 
 
 
from experience. Care should be taken when using the outputs of GPT-4, particularly in contexts
where reliability is important.

GPT-4’s capabilities and limitations create signiﬁcant and novel safety challenges, and we believe
careful study of these challenges is an important area of research given the potential societal impact.
This report includes an extensive system card (after the Appendix) describing some of the risks we
foresee around bias, disinformation, over-reliance, privacy, cybersecurity, proliferation, and more.
It also describes interventions we made to mitigate potential harms from the deployment of GPT-4,
including adversarial testing with domain experts, and a model-assisted safety pipeline.

2 Scope and Limitations of this Technical Report

This report focuses on the capabilities, limitations, and safety properties of GPT-4. GPT-4 is a
Transformer-style model [39] pre-trained to predict the next token in a document, using both publicly
available data (such as internet data) and data licensed from third-party providers. The model was
then ﬁne-tuned using Reinforcement Learning from Human Feedback (RLHF) [40]. Given both
the competitive landscape and the safety implications of large-scale models like GPT-4, this report
contains no further details about the architecture (including model size), hardware, training compute,
dataset construction, training method, or similar.

We are committed to independent auditing of our technologies, and shared some initial steps and
ideas in this area in the system card accompanying this release.2 We plan to make further technical
details available to additional third parties who can advise us on how to weigh the competitive and
safety considerations above against the scientiﬁc value of further transparency.

3 Predictable Scaling

A large focus of the GPT-4 project was building a deep learning stack that scales predictably. The
primary reason is that for very large training runs like GPT-4, it is not feasible to do extensive
model-speciﬁc tuning. To address this, we developed infrastructure and optimization methods that
have very predictable behavior across multiple scales. These improvements allowed us to reliably
predict some aspects of the performance of GPT-4 from smaller models trained using 1, 000
–
10, 000

less compute.

×

×

3.1 Loss Prediction

The ﬁnal loss of properly-trained large language models is thought to be well approximated by power
laws in the amount of compute used to train the model [41, 42, 2, 14, 15].

To verify the scalability of our optimization infrastructure, we predicted GPT-4’s ﬁnal loss on our
internal codebase (not part of the training set) by ﬁtting a scaling law with an irreducible loss term
(as in Henighan et al. [15]): L(C) = aC b + c, from models trained using the same methodology
but using at most 10,000x less compute than GPT-4. This prediction was made shortly after the run
started, without use of any partial results. The ﬁtted scaling law predicted GPT-4’s ﬁnal loss with
high accuracy (Figure 1).

3.2 Scaling of Capabilities on HumanEval

Having a sense of the capabilities of a model before training can improve decisions around alignment,
safety, and deployment. In addition to predicting ﬁnal loss, we developed methodology to predict
more interpretable metrics of capability. One such metric is pass rate on the HumanEval dataset [43],
which measures the ability to synthesize Python functions of varying complexity. We successfully
predicted the pass rate on a subset of the HumanEval dataset by extrapolating from models trained
with at most 1, 000

less compute (Figure 2).

For an individual problem in HumanEval, performance may occasionally worsen with scale. Despite
C−k
these challenges, we ﬁnd an approximate power law relationship

EP [log(pass_rate(C))] = α

−

∗

2In addition to the accompanying system card, OpenAI will soon publish additional thoughts on the social

and economic implications of AI systems, including the need for effective regulation.

2

×

Figure 1. Performance of GPT-4 and smaller models. The metric is ﬁnal loss on a dataset derived
from our internal codebase. This is a convenient, large dataset of code tokens which is not contained in
the training set. We chose to look at loss because it tends to be less noisy than other measures across
different amounts of training compute. A power law ﬁt to the smaller models (excluding GPT-4) is
shown as the dotted line; this ﬁt accurately predicts GPT-4’s ﬁnal loss. The x-axis is training compute
normalized so that GPT-4 is 1.

Figure 2. Performance of GPT-4 and smaller models. The metric is mean log pass rate on a subset of
the HumanEval dataset. A power law ﬁt to the smaller models (excluding GPT-4) is shown as the dotted
line; this ﬁt accurately predicts GPT-4’s performance. The x-axis is training compute normalized so that
GPT-4 is 1.

3

ObservedPredictiongpt-4100p10n1µ100µ0.011Compute1.02.03.04.05.06.0Bits per wordOpenAI codebase next word predictionObservedPredictiongpt-41µ10µ100µ0.0010.010.11Compute012345– Mean Log Pass RateCapability prediction on 23 coding problemswhere k and α are positive constants, and P is a subset of problems in the dataset. We hypothesize
that this relationship holds for all problems in this dataset. In practice, very low pass rates are difﬁcult
or impossible to estimate, so we restrict to problems P and models M such that given some large
sample budget, every problem is solved at least once by every model.

We registered predictions for GPT-4’s performance on HumanEval before training completed, using
only information available prior to training. All but the 15 hardest HumanEval problems were split
into 6 difﬁculty buckets based on the performance of smaller models. The results on the 3rd easiest
bucket are shown in Figure 2, showing that the resulting predictions were very accurate for this
subset of HumanEval problems where we can accurately estimate log(pass_rate) for several smaller
models. Predictions on the other ﬁve buckets performed almost as well, the main exception being
GPT-4 underperforming our predictions on the easiest bucket.

Certain capabilities remain hard to predict. For example, the Inverse Scaling Prize [44] proposed
several tasks for which model performance decreases as a function of scale. Similarly to a recent
result by Wei et al. [45], we ﬁnd that GPT-4 reverses this trend, as shown on one of the tasks called
Hindsight Neglect [46] in Figure 3.

Figure 3. Performance of GPT-4 and smaller models on the Hindsight Neglect task. Accuracy is shown
on the y-axis, higher is better. ada, babbage, and curie refer to models available via the OpenAI API [47].

We believe that accurately predicting future capabilities is important for safety. Going forward we
plan to reﬁne these methods and register performance predictions across various capabilities before
large model training begins, and we hope this becomes a common goal in the ﬁeld.

4 Capabilities

We tested GPT-4 on a diverse set of benchmarks, including simulating exams that were originally
designed for humans.4 We did no speciﬁc training for these exams. A minority of the problems in the
exams were seen by the model during training; for each exam we run a variant with these questions
removed and report the lower score of the two. We believe the results to be representative. For further
details on contamination (methodology and per-exam statistics), see Appendix C.

Exams were sourced from publicly-available materials. Exam questions included both multiple-
choice and free-response questions; we designed separate prompts for each format, and images were
included in the input for questions which required it. The evaluation setup was designed based
on performance on a validation set of exams, and we report ﬁnal results on held-out test exams.
Overall scores were determined by combining multiple-choice and free-response question scores
using publicly available methodologies for each exam. We estimate and report the percentile each
overall score corresponds to. See Appendix A for further details on the exam evaluation methodology.

3For AMC 10 and AMC 12 2022 exams, the human percentiles are not yet published, so the reported numbers

are extrapolated and likely have wide uncertainty. See Appendix A.5.

4We used the post-trained RLHF model for these exams.

====
```

GPT-3.5 output:

```
1. N/A
2. N/A
```

Correct.

GPT-4 output:

```
1. N/A
2. N/A
```

Correct.

### LaMDA

Input:

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====
2 Related work

Language models and dialog models: Language models have attracted much attention recently thanks to their
successes in NLP applications (e.g., [19, 20, 21, 2, 1, 22, 23, 5, 12, 24]). Our study of scaling laws with respect to
model sizes is inspired by recent work on the scaling laws of neural language models [12, 13]. Similar to their ﬁndings,
our results show that model scaling improves our quality (sensibleness, speciﬁcity, and interestingness), safety and
groundedness metrics to some extent. However, ﬁne-tuning combined with scaling signiﬁcantly improves performance
on all metrics.

Our work is also closely related to recent successes in applying language models to dialog modeling (e.g., [25, 26,
17, 18]), which built on earlier research in neural dialog modeling (e.g., [14, 15, 16, 27, 28]). One of our ﬁne-tuning
stages requires training on dialog-only data, which is related to Wolf et al. [29], Dinan et al. [25] and Zhang et al. [30].
Our use of ﬁne-tuning on crowdworker-annotated data to improve interestingness is comparable to Roller et al. [18].
However, we aim to maximize the interestingness of the model’s output distinctly from its ability to engage the user in
further interaction.

Our ﬁnding that pure scaling has a limited effect on key measures of open-domain dialog model performance echoes
that of Shuster et al. [31], who also focus on the problem of groundedness. Recent studies on scaling have found that
performance on question-answering tasks improves with model size [32, 33], similar to our ﬁndings on pre-trained
LaMDA prior to ﬁne-tuning.

Our approach to improving model groundedness is broadly consistent with a growing literature on augmenting neural
language models with retrieval systems. Most of the existing literature focuses on the problem of open-domain
question-answering rather than dialog generation, and the models themselves are used to index and rank knowledge
sources, rather than trained to use an intermediate tool. Given these differences, we note that the range of existing
approaches to this problem include the RNNLM [34], RAG [35], REALM [36], and FiD [37] architectures. Zhu et
al. [38] provide a survey of further recent work. See Karpukhin et al. [39] for details on the ‘dense passage retriever’
used in RAG. Recent work in this direction has expanded and elaborated on neural models’ ability to retrieve and rank
passages [40]. The RETRO architecture demonstrates that language models can be primed with results retrieved from
a database as large as two trillion tokens [41]. At a broad level, our approach is also comparable to that of Byrne et
al. [42], which ﬁne-tunes the model to use external APIs for movie ticketing dialog.

Parts of our ﬁndings are similar to recent studies on dialog groundedness. Granting access to external knowledge
bases has been shown to reduce the rate at which models hallucinate unsourced statements in dialog across a variety of
retrieval systems and model architectures [31]. Another study ﬁnds that a question-answering system’s accuracy is
improved by separating it into a reasoning unit and a response generator, analogous to our separation of ‘Base’ and
‘Research’ models in our study [43]. Meanwhile, the WebGPT framework includes a language system that can interact
with the open web via a text-only interface, and learns to imitate humans in answering questions by citing external
sources [44]. Komeili et al. [45] compare different types of pre-trained models and retrieval methods, and reach a
similar conclusion that augmenting language models with a search engine provides more factually grounded responses.
They encode the input context with grounded information from search to generate the next response, while we augment
the generated responses with information from known sources in our method. This allows us to ﬁne-tune the model for
groundedness without sacriﬁcing gains in safety or quality from other ﬁne-tuning treatments.

Dialog metrics: Deﬁning effective metrics for dialog models remains an open research topic. Our approach is
inspired by Adiwardana et al. [17], who argued for human-like metrics, such as sensibleness and speciﬁcity. Many
automated metrics for dialog models have been studied, including perplexity [16, 17], F1, Hits@1/N [25], USR [46],
or BLEU/ROUGE [47, 15, 27]. However, such automated metrics may not correlate well with human judgment [48].
More reliable metrics for dialog modeling require human evaluation [49, 50, 18, 25, 17, 51], as used in this paper.

Earlier research attempted to combine multifaceted evaluations of dialog quality into a single headline metric [52]. We
follow the pattern established in Adiwardana et al. [17] and Roller et al. [18] by considering the different components
of our evaluations separately. In addition to sensibleness and speciﬁcity per Adiwardana et al. [17], we add new metrics:
interestingness, safety, and groundedness. An advantage of using several different metrics is their debuggability: by
exploring responses with low safety or groundedness scores, we have been able to develop targeted methods to improve
them.

Safety and safety of dialog models:
Inappropriate and unsafe risks and behaviors of language models have been
extensively discussed and studied in previous works (e.g., [53, 54]). Issues encountered include toxicity (e.g., [55, 56,
57]), bias (e.g., [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]), and inappropriately revealing personally
identifying information (PII) from training data [73]. Weidinger et al. [54] identify 21 risks associated with large-scale

3

language models and discuss the points of origin for these risks. While many mitigation strategies have also been
suggested (e.g., [74, 75, 76, 77, 78, 79, 80, 81, 82]), meaningfully addressing these issues remains an active research
area.

Similar issues have also been discussed speciﬁcally for dialog models [53]. For instance, examples of bias, offensiveness,
and hate speech have been found both in training data drawn from social media, and consequently in the output of dialog
models trained on such data [83]. Dialog models [84] can learn, and even amplify, biases in the training data. Echoing
Gehman et al. [85], we ﬁnd ﬁne-tuning effective to augment language models for safety. The method we use in this
paper follows previous attempts to tackle these issues by training separate layers to detect unsafe output [17, 86, 18, 79].
Our strategy is similar to recent work that also uses ﬁne-tuning [87]. While their safety guidelines were derived from
human rights principles, they similarly ﬁnd that increasing scale has no impact on toxicity metrics, while ﬁne-tuning on
safety evaluations does.

Groundedness metrics: Similar to other recent research into groundedness cited above, we assess groundedness
by asking crowdworkers to judge whether the model’s output is in accordance with authoritative external sources.
The recently-proposed Attributable to Identiﬁed Sources (AIS) framework [88] articulates a more precise approach
to assess output of language models that pertains to the external world. It splits evaluation into two stages, where
crowdworkers are asked: (1) if they can understand and identify the information shared in a dialog turn, and (2) if all
of this information can be attributed to a source. Meanwhile, a recent study has reopened the question of automatic
evaluation, with the Q2 metric showing performance comparable to human annotation [89].

3 LaMDA pre-training

LaMDA was pre-trained to predict the next token in a text corpus. Unlike previous dialog models trained on dialog data
alone [17, 18], we pre-trained LaMDA on a dataset created from public dialog data and other public web documents.
Therefore, LaMDA can be used as a general language model prior to ﬁne-tuning.

The pre-training dataset consists of 2.97B documents, 1.12B dialogs, and 13.39B dialog utterances, for a total of 1.56T
words (Appendix E). Over 90% of the pre-training dataset is in the English language. We used the SentencePiece
library [90] to tokenize the dataset into 2.81T byte pair encoding (BPE) tokens [91], with a vocabulary of 32K tokens.
For comparison, the total number of words in the training set for Meena [17] was 40B words, which is nearly 40x
smaller.

The largest LaMDA model has 137B non-embedding parameters, which is ~50x more parameters than Meena [17].
We use a decoder-only Transformer [92] language model as the model architecture for LaMDA. The Transformer has
64 layers, dmodel = 8192, df f = 65536, h = 128, dk = dv = 128, relative attention as described in T5 [11], and
gated-GELU activation as described in Raffel et al. [93].

We pre-trained LaMDA on 1024 TPU-v3 chips for a total of about 57.7 days, and 256K tokens per batch. We used
the Lingvo framework [94] for training and achieved 123 TFLOPS/sec with 56.5% FLOPS utilization with the 2D
sharding algorithm, as described in GSPMD [95] (see Section 10 for carbon footprint estimates). We also trained
smaller 2B-parameter and 8B-parameter models to measure the effects of model scaling on our metrics. Hyperparameter
details for the models of different sizes can be found in Table 27, Appendix D.

Figure 2 gives an overview of the pre-training stage. We call the model before any ﬁne-tuning "PT", for PreTrained.

Figure 2: LaMDA pre-training as a language model.

4

PT uses the same sample-and-rank strategy as Meena [17] for decoding. We ﬁrst sample 16 independent candidate
responses using top-k (k = 40) sampling (no temperature). The ﬁnal output is the highest-scoring candidate, where the
score is based on the candidate’s log-likelihood and its length.

4 Metrics

Evaluating generative models in general, and open-ended dialog models in particular, is difﬁcult. See the Related
Work section for a general review of recent work in this area. In this section, we describe the metrics that we use for
evaluation.

4.1 Foundation metrics: Quality, Safety and Groundedness

Sensibleness, Speciﬁcity, Interestingness (SSI): Our overall quality score is an average of sensibleness, speciﬁcity,
and interestingness (SSI).

Adiwardana et al. [17] propose the sensibleness and speciﬁcity average (SSA) metric to measure the quality of Meena.
This metric is a simple average of two scores: sensibleness and speciﬁcity.

The ﬁrst score, sensibleness, measures whether a model’s responses make sense in context and do not contradict
anything that was said earlier. Humans tend to take this basic aspect of communication for granted, but generative
models often struggle to meet this requirement. However, if sensibleness alone is used to evaluate models, we could
inadvertently reward models for playing it safe by always producing short, generic, and boring responses. The
GenericBot algorithm [17], which answers every question with “I don’t know” and every statement with “Ok,” scores
70% on sensibleness, which even surpasses some large dialog models [17].

The second score, speciﬁcity, is used to measure whether a response is speciﬁc to a given context. For example, if a user
says “I love Eurovision” and the model responds “Me too,” then it would score 0 on speciﬁcity, since this response could
be used in many different contexts. If it answers “Me too. I love Eurovision songs,” then it would score 1. Adiwardana
et al. [17] report that Meena narrows the gap to average human performance in the SSA metric.

As the model’s performance increases, however, we ﬁnd that sensibleness and speciﬁcity are not sufﬁcient to measure
the quality of a dialog model. For example, a response to “How do I throw a ball?” could be “You can throw a ball by
ﬁrst picking it up and then throwing it”, which makes sense and is speciﬁc to the question. An alternative deeper and
more satisfying answer could be “One way to toss a ball is to hold it ﬁrmly in both hands and then swing your arm
down and up again, extending your elbow and then releasing the ball upwards.”

====
```

GPT-3.5 output:
```
1. 1024
2. TPU-v3
```

Correct.

GPT-4 output:

```
1. 1024
2. TPU-v3
```

Correct.

### Phenaki

Input:

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

2
2
0
2

t
c
O
5

]

V
C
.
s
c
[

1
v
9
9
3
2
0
.
0
1
2
2
:
v
i
X
r
a

PHENAKI: VARIABLE LENGTH VIDEO GENERATION
FROM OPEN DOMAIN TEXTUAL DESCRIPTIONS

Ruben Villegas‡
Google Brain
rubville@google.com

Mohammad Babaeizadeh‡
Google Brain
mbz@google.com

Pieter-Jan Kindermans‡
Google Brain
pikinder@google.com

Hernan Moraldo
Google Brain
hmoraldo@google.com

Han Zhang
Google Brain
zhanghan@google.com

Mohammad Taghi Saffar
Google Brain
msaffar@google.com

Santiago Castro∗
University of Michigan
sacastro@umich.edu

Julius Kunze∗
University College London
kjulius@google.com

Dumitru Erhan
Google Brain
dumitru@google.com

ABSTRACT

We present Phenaki, a model capable of realistic video synthesis, given a sequence
of textual prompts. Generating videos from text is particularly challenging due to
the computational cost, limited quantities of high quality text-video data and vari-
able length of videos. To address these issues, we introduce a new model for
learning video representation which compresses the video to a small representa-
tion of discrete tokens. This tokenizer uses causal attention in time, which allows
it to work with variable-length videos. To generate video tokens from text we
are using a bidirectional masked transformer conditioned on pre-computed text
tokens. The generated video tokens are subsequently de-tokenized to create the
actual video. To address data issues, we demonstrate how joint training on a large
corpus of image-text pairs as well as a smaller number of video-text examples can
result in generalization beyond what is available in the video datasets. Compared
to the previous video generation methods, Phenaki can generate arbitrary long
videos conditioned on a sequence of prompts (i.e. time variable text or a story) in
open domain. To the best of our knowledge, this is the ﬁrst time a paper studies
generating videos from time variable prompts. In addition, compared to the per-
frame baselines, the proposed video encoder-decoder computes fewer tokens per
video but results in better spatio-temporal consistency.

1

INTRODUCTION

It is now possible to generate realistic high resolution images given a description [34, 35, 32, 38,
59], but generating high quality videos from text remains challenging. In essence, videos are just
a sequence of images, but this does not mean that generating a long coherent video is easy.
In
practice, it is a signiﬁcantly harder task because there is much less high quality data available and
the computational requirements are much more severe [9]. For image generation, there are datasets
with billions of image-text pairs (such as LAION-5B [41] and JFT4B [60]) while the text-video
datasets are substantially smaller e.g. WebVid [4] with ∼10M videos, which is not enough given
the higher complexity of open domain videos. As for computation, training current state-of-the-
art image generation models is already pushing the state-of-the-art computational capabilities [59],
leaving little to no room for generating videos, particularly videos of variable length.

To make the matters worse, one can argue that a single short text prompt is not sufﬁcient to provide
a complete description of a video (except for short clips), and instead, a generated video must be
conditioned on a sequence of prompts, or a story, which narrates what happens over time. Ideally,

‡Equal contribution. * Intern at Google Brain while working on this project.

1

 
 
 
 
 
 
Figure 1. Time variable text (i.e. story) conditional video generation. The entire ﬁgure is one
continuous video generated auto-regressively. We start by generating the video conditioned on the
ﬁrst prompt and then after a couple of frames we change the prompt to the next one. Each row
contains a selected number of frames (from left to right in order) while the model was conditioned
on that particular prompt. The model manages to preserve the temporal coherence of the video
while adapting to the new prompt, usually taking the shortest path for the adaption (notice the
morphing of the teddy bear to the panda). Please note that the generated video has complex visual
features such as reﬂections, occlusions, interactions and scene transitions. Full video is available at
phenaki.github.io.

a video generation model must be able to generate videos of arbitrary length, all the while having
the capability of conditioning the generated frames at time t on prompts at time t that can vary over
time. Such capability can clearly distinguish the video from a “moving image” and open up the way
to real-world creative applications in art, design and content creation. To the best our knowledge,
story based conditional video generation has never been explored before and this is the ﬁrst paper to
take early steps towards that goal. A traditional deep learning approach of simply learning this task
from data is not possible, since there is no story-based dataset to learn from. Instead, to achieve this
we rely on a model that is designed speciﬁcally with this capability in mind.

In this paper, we introduce Phenaki, a text to video model trained on both text to video and text to
image data that can:

– Generate temporally coherent and diverse videos conditioned on open domain prompts even
when the prompt is a new composition of concepts (Fig. 3). The videos can be long (minutes)
even though the model is trained on 1.4 seconds videos (at 8 fps).

– Generate videos conditioned on a story (i.e. a sequence of prompts), e.g. Fig. 1 and Fig. 5.

2

Figure 2. The architecture of Phenaki. Left: C-ViViT encoder architecture. The embeddings of
images and video patches from raw frames x are processed by a spatial and then a causal transformer
(auto-regressive in time) to generate video tokens z. Center: MaskGiT is trained to reconstruct
masked tokens z predicted by a frozen C-ViViT encoder and conditioned on T5X tokens of a given
prompt p0. Right: How Phenaki can generate arbitrary long videos by freezing the past token and
generating the future tokens. The prompt can change over time to enable time-variable prompt (i.e.
story) conditional generation. The subscripts represent time (i.e. frame number).

To enable these capabilities, we could not rely on current video encoders, because they either can
only decode ﬁxed size videos or they encode frames independently. Hence, we introduce C-ViViT ,
a novel encoder-decoder architecture that:

– Exploits temporal redundancy in videos to improve reconstruction quality over a per frame model

while compressing the number of video tokens by 40% or more.

– Allows encoding and decoding of variable length videos given its causal structure.

2 THE PHENAKI MODEL

Inspired by the previous work in auto-regressive text to image [34, 59, 38] and text to video [54,
53, 18], Phenaki is designed with two main components (see Figure 2): an encoder-decoder model
which compresses videos to discrete embeddings (i.e. tokens) and a transformer model to translate
text embeddings to video tokens. To get the text embeddings, Phenaki uses a pre-trained language
model, T5X [37]. We will discuss each one of these components in the following subsections.

2.1 ENCODER-DECODER VIDEO MODEL: C-VIVIT

One of the primary challenges for generating video from text, is to get a compressed representation
of videos. Previous work on text to video either use per-frame image encoders [18, 54, 57] such
as VQ-GAN [12] or ﬁxed length video encoders [52] such as VideoVQVAE [49]. The former
allows for generating videos of arbitrary length, however in practice, the videos have to be short
because the encoder does not compress the videos in time and the tokens are highly redundant in
consecutive frames. The latter is more efﬁcient in the number of tokens but it does not allow to
generate variable length videos. In Phenaki, our goal is to generate videos of variable length while
keeping the number of video tokens to a minimum so they can be modeled with a transformer
within current computational limitations. To do so, we introduce C-ViViT , a causal variation of
ViViT [1] with additional architectural changes for video generation, which can compress the videos
in temporal and spatial dimensions, while staying auto-regressive in time, This capability allows for
generating videos of arbitrary length auto-regressively.

3

EmptyTokensTokensPatch  EmbPatch  EmbPatch  EmbSpatial TransformerSpatial TransformerSpatial TransformerCausal TransformerCausal TransformerCausal Transformer............C-ViViT EncoderT5X...TransformerRandom Masking......VideoTokensTokensMaskedReconstructed...Transformer...Shift Time......Transformer...T5XT5X..."Next Prompt"TokensTokensPredictedFrozne PastPredictedFuture TokensC-ViViT EncoderTraining TransformerVideo GenerationTokenMasked/Empty TokenTransformerFrozen ModelLinearEmbeddingOperation"1st Prompt""Prompt"DiscretizeDiscretizeDiscretize...Encoder architecture: As illustrated in Figure 2, we start with a video sequence of tx + 1 frames
with a resolution of wx × hx and cx channels: x ∈ R(tx+1)×hx×wx×cx . This sequence will be
compressed into a token representation of size (tz + 1) × wz × hz where the ﬁrst wz × hz tokens
represent the ﬁrst frame independently from the rest of the video, and the remaining tokens represent
spatio-temporal video tokens that auto-regressively depend on previous frames. To do so, we extract
non-overlapping image patches of size wp × hp × cp from the ﬁrst frame and video patches of size
tp × wp × hp × cp from the rest of the video. We typically use all channels at once such that the
number of patches equals the number of video tokens tz = tx
. Each of
tp
these patches is ﬂattened and linearly projected into a dz dimensional space. We combine the spatial
dimensions to have a tensor of shape (tz+1)×wz∗hz×dz where the spatial and temporal dimensions
are separated. Then multiple transformer layers are applied along the spatial dimensions with all-
to-all attention. This is followed by multiple transformer layers over the temporal dimension with
causal attention such that each spatial token only observes spatial tokens from previous frames in
an auto-regressive manner. The effect of this is that the ﬁrst frame can be completely independently
encoded. This opens up the possibility of text to image training to be embedded naturally into
our video model. The second advantage is that we can condition the video generation process on
a number of starting frames. The resulting patch embeddings z of shape tz × wz × hz × dz are
then tokenized into learned codewords cz by vector quantization. The codebook learning will be
discussed later together with the losses.

and hz = hx
hp

, wz = wx
wp

Decoder architecture: The C-ViViT decoder is simply an upside down version of the encoder.
First tokens are transformed into embeddings. This is followed by the temporal transformer, then the
spatial transformer. After the output of the spatial transformer, we apply a single linear projection
without activation to map the tokens back to pixel space.

Quantization and Losses: To learn a discrete latent space, we quantize our encoder outputs into
the entries of a learned codebook via the vector quantization (VQ) objective in VQVAEs [45],

LVQ = (cid:107)sg(z) − e(cid:107)2

2 + β(cid:107)z − sg(e)(cid:107)2
2,

(1)

where sg(x) ≡ x, and d
dx sg(x) ≡ 0 is the stop-gradient operator, β is the commitment loss weight,
and e is a codebook vector from codebook E. The index to the codebook vector closest to z is
found by i = argminj(cid:107)z − Ej(cid:107)2
2. In addition to the VQ objective, we adopt the factorized and (cid:96)2-
normalized codes from ViT-VQGAN [58] to improve codebook usage and reconstruction quality.

To train our model, we use a combination of L2 loss, image perceptual loss LIP [20, 61], video
perceptual loss LVP by using the I3D network [6] as feature extractor, and adversarial loss LAdv with
StyleGAN architecture [21]. As training objective, we use the following

L = LVQ + 0.1 × LAdv + 0.1 × LIP + 1.0 × LVP + 1.0 × L2.

(2)

Novelty over the ViViT architecture: While our proposed C-ViViT architecture is inspired by
the factorized encoder in ViViT [1], we modify their architecture to enable self-supervised learn-
ing from unlabeled videos. We ﬁrst remove the [CLS] tokens in the spatial and the temporal
transformers. Next, we apply temporal transformer for all spatial tokens computed by the spatial en-
coder, in contrast to single run of the temporal transformer over the [CLS] tokens in ViViT. Most
importantly, the ViViT encoder requires a ﬁxed length video input due to the all-to-all attention in
time. Therefore, we apply causal attention instead such that our C-ViViT encoder becomes auto-
regressive and allows for a variable number of input frames which are necessary to learn from image
datasets, and auto-regressively extrapolate video or single frames into the future.

====
```

GPT-3.5 output:

```
1. N/A
2. N/A
```

Correct.

GPT-4 output:

```
1. N/A
2. N/A
```

Correct.

### Robust speech recognition

Input:

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

4.2. Dataset Scaling

At 680,000 hours of labeled audio, the Whisper dataset is
one of the largest ever created in supervised speech recog-
nition. Exactly how important is the raw dataset size to
Whisper’s performance? To study this, we trained a series
of medium-sized models on subsampled versions of the
dataset which are 0.5%, 1%, 2%, 4%, and 8% of the full
dataset size and compared their performance with the same
medium-sized model trained on the whole dataset. Early
stopping based on the validation loss was used to select
model checkpoints for each dataset size. Evaluation was
performed on an exponential moving average estimate of
the parameters (Polyak & Juditsky, 1992) using a smooth-
ing rate of 0.9999 to help reduce the effect of the learning
rate not fully decaying to zero for the models trained on the
subsampled datasets due to early stopping. Performance
on English and multilingual speech recognition and X→en
translation is reported in Table 6.

WhisperABCDEFGHI ASR            human transcription            computer-assisted051015202530Word Error Rate (%)Robust Speech Recognition via Large-Scale Weak Supervision

11

Figure 8. Zero-shot Whisper performance scales reliably across tasks and languages with increasing model size. Lightly shaded
lines represent individual datasets or languages, showing that performance is more varied than the smooth trends in aggregate performance.
Large V2 distinguished with a dashed orange line since it includes several changes that are not present for the smaller models in this
analysis.

Dataset
size

3405
6811
13621
27243
54486
681070

English Multilingual
WER (↓) WER (↓)

X→En
BLEU (↑)

30.5
19.6
14.4
12.3
10.9
9.9

92.4
72.7
56.6
45.0
36.4
29.2

0.2
1.7
7.9
13.9
19.2
24.8

Table 6. Performance improves with increasing dataset size.
English speech recognition performance refers to an average over
12 datasets while the Multilingual speech recognition reports per-
formance on the overlapping subset of languages in Fleurs and
X→en translation reports average BLEU on CoVoST2. Dataset
size reported in hours.

All increases in the dataset size result in improved perfor-
mance on all tasks, although we see significant variability
in improvement rates across tasks and sizes. Performance
improves rapidly on English speech recognition from 3,000
to 13,000 hours and then slows down noticeably between
13,000 and 54,000 hours. Using the full dataset, which cor-
responds to another 12.5× increase in size results in only a
further 1 point drop in WER. This mirrors the diminishing
returns observed with model size scaling for English speech
recognition and could similarly be explained by saturation
effects when approaching human-level performance.

Improvements in WER follow a power-law trend for mul-
tilingual speech recognition till 54,000 hours and then de-
viate from this trend, improving only a further 7 points
when increasing to the full dataset size. For X→en transla-
tion, performance is practically zero when training on 7,000
hours of audio or less, and then follows a roughly log-linear
improvement trend till 54,000 hours before also showing

diminishing returns when further scaling to the full dataset
size.

The general trend across tasks of diminishing returns when
moving from 54,000 hours to our full dataset size of 680,000
hours could suggest that the current best Whisper models are
under-trained relative to dataset size and performance could
be further improved by a combination of longer training
and larger models. It could also suggest that we are nearing
the end of performance improvements from dataset size
scaling for speech recognition. Further analysis is needed to
characterize “scaling laws” for speech recognition in order
to decided between these explanations.

4.3. Multitask and Multilingual Transfer

A potential concern with jointly training a single model
on many tasks and languages is the possibility of negative
transfer where interference between the learning of several
tasks results in performance worse than would be achieved
by training on only a single task or language. To investigate
whether this is occurring, we compared the performance
of models trained on just English speech recognition with
our standard multitask and multilingual training setup and
measured their average performance across our suite of zero-
shot English speech recognition benchmarks. We adjust for
the amount of FLOPs spent training on the task of English
speech recognition as only 65% of compute is spent on this
task in a joint training setup; analysis would otherwise be
confounded by under-training on the task when compared
to a same-sized English-only model.

Our results visualized in Figure 9 show that for small models
trained with moderate amounts of compute, there is indeed
negative transfer between tasks and languages: joint mod-
els underperform English-only models trained for the same
amount of compute. However, multitask and multilingual

38M73M244M768M1549M1549MModel parameters0.02.55.07.510.012.515.017.520.0WER on 12 datasets (%)English Speech RecognitionAverageLarge V238M73M244M768M1549M1549MModel parameters020406080100WER on 67 languages (%)Multilingual Speech Recognition (Fleurs)AverageLarge V238M73M244M768M1549M1549MModel parameters01020304050BLEU on 21 languagesX->En Translation (CoVoST2)AverageLarge V238M73M244M768M1549M1549MModel parameters304050607080Accuracy on 102 languages (%)Language Identification (Fleurs)AverageLarge V2Robust Speech Recognition via Large-Scale Weak Supervision

12

Figure 9. Multitask and multilingual transfer improves with
scale. For small models, performance on English speech recogni-
tion degrades when trained jointly in a multitask and multilingual
setup. However, multilingual and multitask models benefit more
from scale and eventually outperform models trained on English
data only. 95% bootstrap estimate confidence intervals are shown.

models scale better and for our largest experiments outper-
form their English-only counterparts demonstrating positive
transfer from other tasks. For our largest experiments, joint
models also slightly outperform English-only models even
when not adjusting for compute spent per task.

4.4. Text Normalization

Since we developed our text normalization jointly with
Whisper to discount innocuous word errors, there is a risk
that our normalizer is overfitted to fixing Whisper’s peculiar-
ities rather than addressing general variation in transcription.
To check this, we compared the performance of Whisper
using our normalizer versus an independently developed
one from the FairSpeech project (Koenecke et al., 2020). In
Figure 10, we visualize the differences. On most datasets
the two normalizers perform similarly, without significant
differences in WER reduction between Whisper and com-
pared open-source models, while on some datasets, namely
WSJ, CallHome, and Switchboard, our normalizer reduces
the WER of Whisper models’ significantly more. The differ-
ences in reduction can be traced down to different formats
used by the ground truth and how the two normalizers are pe-
nalizing them. For example, in CallHome and Switchboard,
our standardizer did not penalize differences in common
English contractions such as “you’re” versus “you are”, and
in WSJ, our normalizer standardized the written and spo-

Figure 10. On most datasets, our text normalizer has similar
effect on reducing WERs between Whisper models and other
open-source models, compared to FairSpeech’s normalizer. For
each dataset, the boxplot shows the distribution of relative WER
reduction across different models in our eval suite, showing that
using our text normalizer generally results in lower WERs than
FairSpeech’s. On a few datasets our normalizer reduces WER
significantly and more so for Whisper models, such as CallHome
and Switchboard which have many contractions in the ground truth
and WSJ which contains many numerical expressions.

ken forms of numerical and monetary expressions, such as
“sixty-eight million dollars” versus “$68 million”.

4.5. Strategies for Reliable Long-form Transcription

Transcribing long-form audio using Whisper relies on ac-
curate prediction of the timestamp tokens to determine the
amount to shift the model’s 30-second audio context win-
dow by, and inaccurate transcription in one window may
negatively impact transcription in the subsequent windows.
We have developed a set of heuristics that help avoid fail-
ure cases of long-form transcription, which is applied in
the results reported in sections 3.8 and 3.9. First, we use
beam search with 5 beams using the log probability as the
score function, to reduce repetition looping which happens
more frequently in greedy decoding. We start with tem-
perature 0, i.e. always selecting the tokens with the high-
est probability, and increase the temperature by 0.2 up to
1.0 when either the average log probability over the gen-
erated tokens is lower than −1 or the generated text has a
gzip compression rate higher than 2.4. Providing the tran-
scribed text from the preceding window as previous-text
conditioning when the applied temperature is below 0.5
further improves the performance. We found that the proba-
bility of the <|nospeech|> token alone is not sufficient

10e+1910e+2010e+2110e+22FLOPs training on english speech recognition8101214161820Average WER on 11 english speech recognition datasetsEnglish OnlyMultilingual and Multitask01020304050Relative WER reduction compared to FairSpeech's normalizer (%)CORAALCommonVoice9.enAMI-SDM1CommonVoice5.1Fleurs.en_usAMI-IHMArtieLibriSpeechTED-LIUM3VoxPopuli.enWSJCallHomeSwitchboardOpen-source modelsWhisper modelsRobust Speech Recognition via Large-Scale Weak Supervision

13

3
M
U
I
L
-
D
E
T

e
l
i
h
w
n
a
e

M

6
4
d
i
a
c
n
i
K

6
1
v
e
R

1
2
-
s
g
n
i
n
r
a
E

2
2
-
s
g
n
i
n
r
a
E

L
A
A
R
O
C

Greedy decoding only
+ Beam search
+ Temperature fallback
+ Voice activity detection
+ Previous text conditioning
+ Initial timestamp constraint

3.95 5.16 9.69 11.7 10.7 14.0 22.0
4.16 5.71 9.42 11.5 10.2 13.4 20.0
4.16 5.71 9.42 11.5 10.2 13.4 20.0
3.56 4.61 9.45 11.4 10.1 13.2 19.4
3.42 6.16 8.72 11.0 9.63 13.3 18.1
3.51 5.26 8.41 11.5 9.73 12.6 19.1

e
g
a
r
e
v
A

11.0
10.6
10.6
10.2
10.0
10.0

Table 7. Long-form transcription performance improves incremen-
tally as additional decoding heuristics are employed. Details on
each intervention are described in Section 4.5.

to distinguish a segment with no speech, but combining
the no-speech probability threshold of 0.6 and the average
log-probability threshold of −1 makes the voice activity
detection of Whisper more reliable. Finally, to avoid a fail-
ure mode where the model ignores the first few words in
the input, we constrained the initial timestamp token to be
between 0.0 and 1.0 second. Table 7 shows that adding each
of the interventions above incrementally reduces the WER
overall, but not evenly across the dataset. These heuristics
serve as a workaround for the noisy predictions of the model,
and more research would be needed to further improve the
reliability of long-form decoding.

5. Related Work

Scaling Speech Recognition A consistent theme across
speech recognition research has been documenting the bene-
fits of scaling compute, models, and datasets. Early work ap-
plying deep learning to speech recognition found improved
performance with model depth and size and leveraged GPU
acceleration to make training these larger models tractable
(Mohamed et al., 2009). Further research demonstrated that
the benefit of deep learning approaches to speech recogni-
tion increased with dataset size, improving from being only
competitive with prior GMM-HMM systems when using
just 3 hours of TIMIT training data for phone recognition
to achieving a 30% word error rate reduction when trained
on the 2,000 hour Switchboard dataset (Seide et al., 2011).
Liao et al. (2013) is an early example of leveraging weakly
supervised learning to increase the size of a deep learn-
ing based speech recognition dataset by over 1,000 hours.
These trends continued with Deep Speech 2 (Amodei et al.,
2015) being a notable system developing high-throughput
distributed training across 16 GPUs and scaling to 12,000
hours of training data while demonstrating continuing im-
provements at that scale. By leveraging semi-supervised
pre-training, Narayanan et al. (2018) were able to grow
dataset size much further and study training on 162,000
hours of labeled audio. More recent work has explored

====
```

GPT-3.5 output:

```
1. N/A
2. N/A
```

Correct.

GPT-4 output:

```
1. N/A
2. N/A
```

Correct.

### Scaling autoregressive models (Parti)

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

2.2 Encoder-Decoder for Text-to-Image Generation

As shown in Figure 3, a standard encoder-decoder Transformer model is trained at the second stage,
by treating text-to-image as a sequence-to-sequence modeling problem. The model takes text as input

5

Figure 4: A learned super-resolution module to upsample 256 × 256 images to higher-resolution
1024 × 1024 ones based on a frozen ViT-VQGAN image tokenizer. The super-resolution module
takes 256 × 256 images as inputs without conditioning on text inputs.

Model

Encoder Layers Decoder Layers Model Dims MLP Dims Heads Total Params

Parti-350M
Parti-750M
Parti-3B
Parti

12
12
12
16

12
36
36
64

1024
1024
2048
4096

4096
4096
8192
16384

16
16
32
64

350M
750M
3B
20B

Table 1: Size variants of Parti. Both encoder and decoder are based on Transformers [4]. The
self-attention layer in decoder transformer is causally masked. Parameters of ViT-VQGAN image
tokenization are not included in the total parameter count and can be found in Section 2.1.

and is trained using next-token prediction of rasterized image latent codes generated from the ﬁrst
stage image tokenizer. For text encoding, we build a sentence-piece model [32, 33] of vocabulary
size 16,000 on a sampled text corpus from the training data (Section 4.1). Image tokens are produced
by a learned ViT-VQGAN image tokenizer (see Section 2.1). At inference time, the model samples
image tokens autoregressively, which are later decoded into pixels using the ViT-VQGAN decoder.

We use a maximum length of text tokens of 128, and the length of image tokens are ﬁxed to 1024
(i.e., 32×32 latent codes from a 256 × 256 input image). As an example, the 67-word description
of the Starry Night prompt given in Figure 1 has a total length of 92 text tokens. All models use
conv-shaped masked sparse attention [34]. We train four size variants ranging from 350 million to
20 billion parameters, as detailed in Table 1. Speciﬁcally, we conﬁgure the Transformers following
previous practice of those in scaling language models with default expansion ratio of 4× in MLP
dimensions. We double the number of heads when the model dimension is doubled. In the current
scaling variants, our conﬁguration prefers a larger decoder for modeling image tokens and as a result
the decoder has more layers (e.g., 3× in the 3B model and 4× in the 20B model).

Most of the existing two-stage text-to-image generation models, including DALL-E [2], CogView [3]
and Make-A-Scene [10], are decoder-only models. We found that at the model scale of 350-million
to 750-million parameters, the encoder-decoder variants of Parti outperformed decoder-only ones,
both in terms of training loss and text-to-image generation quality in our early exploration. We thus
chose to focus on scaling the encoder-decoder models.

2.3 Text Encoder Pretraining

The encoder-decoder architecture also decouples text encoding from image-token generation, so it is
straightforward to explore warm-starting the model with a pretrained text encoder. Intuitively, a text
encoder with representations based on generic language training should be more capable at handling
visually-grounded prompts. We pretrain the text encoder on two datasets: the Colossal Clean Crawled
Corpus (C4) [35] with BERT [36] pretraining objective, and our image-text data (see Section 4.1)
with a contrastive learning objective (image encoder from the contrastive pretraining is not used).
After pretraining, we continue training both encoder and decoder for text-to-image generation with
softmax cross-entropy loss on a vocabulary of 8192 discrete image tokens.

The text encoder after pretraining performs comparably to BERT [36] on GLUE (see Appendix G,
Table 9); however, the text encoder degrades after the full encoder-decoder training process on
text-to-image generation. We leave this observation as a future research topic on the difference and

6

Original Image256 ×256ViT-VQGANReconstructed Image256 ×256Reconstructed Image1024 ×1024nearest downsampleOriginal Image1024 ×1024frozenSuper-Resolution Upsampler trainableFigure 5: 4-way in-layer model parallelism with fully partitioned activations used to scale the 3B
model training. The ﬁgure shows a simpliﬁed Transformer feed-forward layer (with the sequence
dimension omitted); each color represents data on one device. We also use 128-way data parallelism.

uniﬁcation of generic language representation and visually-grounded language representation. Still,
the text-encoder pretraining marginally helps text-to-image generation loss with 3B-parameter Parti
models, so pretraining is used by default in our 20B model. We provide detailed training loss, GLUE
evaluation of text encoders, and some qualitative comparison in Appendix G.

2.4 Classiﬁer-Free Guidance and Reranking

Classiﬁer-free guidance [37] (CF-guidance in short) is critical in the context of improving the sample
quality of diffusion models [11, 12, 13] without pretrained classiﬁers. In this setup, a generative
model G is trained to be able to perform unconditional generation G(z) (where z represents random
noise) and conditional generation G(z, c) (where c represents some condition, such as language
descriptions). It is implemented as randomly dropping out the conditional vector (masking out or
switching to a learned embedding) with some probability. During the inference process, sampling of
an output I is done by using a linear combination of the unconditional and conditional predictions:

I = G(z) + λ(G(z, c) − G(z)),

(1)

where λ is a hyperparameter representing the weight of classiﬁer-free guidance.
Intuitively, it
decreases the unconditional likelihood of the sample while increasing the conditional likelihood,
which can be viewed as encouraging alignment between the generated sample and the text condition.

Classiﬁer-free guidance has been similarly applied in the context of autoregressive models for text-
to-image generation [10, 38] to great effect. Make-A-Scene [10] ﬁnetunes the model by randomly
replacing the text prompts with padded tokens. During inference, tokens are sampled from a linear
combination of logits sampled from an unconditional model and a conditional model on a text prompt.
We also apply CF-guidance in Parti, and ﬁnd it has a signiﬁcant improvement on the output image-text
alignment, especially on challenging text prompts.

With batch-sampled images per text prompt, contrastive reranking is used in DALL-E [2] which
produces image-text alignment scores after the generation. We apply contrastive reranking in our
work and ﬁnd it is complementary to classiﬁer-free guidance. Compared with the 512 images used in
DALL-E [2], we sample just 16 images per text prompt for the experiments reported in this paper.
We rerank each output set based on the alignment score of image and text embedding of a Contrastive
Captioners model (CoCa) [25]. A CoCa base-size model (Table 1 in [25]) is trained on the same
dataset with details in Section 4.1. We note that reranking over a small set of batch-sampled images
is computationally cheap in the text-to-image sampling process, and produces helpful image-text
alignment scores among diverse image outputs.

3 Scaling

We implement our models in Lingvo [39] and scale with GSPMD [28] on CloudTPUv4 hardware
for both training and inference. GSPMD is an XLA compiler-based model partitioning system that
allows us to treat a cluster of TPUs as a single virtual device and use sharding annotations on a few
tensors to instruct the compiler to automatically distribute data and compute on thousands of devices.

7

matmulrelumatmulbdfdbddfbfall-gather inputsreduce-scatter outputsdevice0device1device3device2Figure 6: An illustration of 16-stage GSPMD pipelines used to scale the 20B model training. The
ﬁgure shows how the 16 devices are used for data parallelism in the quantizer, embedding and softmax
layers, but repurposed for pipelining in the encoder and decoder layers. Each color represents data
or layer assigned to one device. The decoder uses 4-round circular schedule to further reduce the
pipeline bubble ratio. On top of this, we use additional 64-way data parallelism for all layers.

Training. We train both 350M and 750M models simply with data parallelism. For the 3B model,
we use 4-way in-layer model parallelism (see Figure 5), and 128-way data parallelism. Partitioning a
single dimension in each tensor is sufﬁcient to scale a 3B model. The model weights are partitioned
on the feed-forward hidden dimension and the number of attention heads dimension; the internal
activation tensors of the feed-forward and attention layers are also partitioned on the hidden and heads
dimensions. One difference from Megatron-LM [27] is we fully partition the output activations of
feed-forward and attention layers on a different dimension, with the details illustrated as the ﬁnalized
2d sharding in the GSPMD work [28]. This strategy will result in ReduceScatter and AllGather
communication patterns instead of AllReduce, which signiﬁcantly reduce peak activation memory.

The 20B model has 16 encoder layers, and 64 decoder layers (see Table 1). The size of the weights
per layer is moderate (as opposed to being very wide), which makes pipeline parallelism [40] a good
option for scaling. We use a generic pipelining wrapper layer allowing us to specify a single-stage
program, which will later be automatically transformed into a multi-stage pipelining program; the
wrapper layer uses vectorization and shifting buffers to reduce pipelining into a tensor partitioning
problem (see Section 3.3 of [28]). Thus, all lower-level infrastructure can be reused for pipelining.
There are two additional beneﬁts in adopting GSPMD pipelining: 1) it allows us to conveniently
conﬁgure pipelines within model sub-components, simplifying the overall complexity for encoder-
decoder models, and 2) since pipelining is implemented as tensor partitioning on vectorized programs,
we can reuse the same set of devices for other types of parallelism outside the transformer layers.

We conﬁgure the model to have separate encoder and decoder pipelines, each with 16 stages. We
also use 64-way data parallelism in addition to pipelining. However this makes per-core batch size
small, exposing an additional challenge of excessive pipeline stalls due to inter-stage data dependency
(known as bubbles in pipeline parallelism [40]). To reduce the ratio of bubbles, we adapt the circular
schedule as described in [28] in the decoder pipeline (a similar technique was also proposed in [41]),
where the 4 layers in each stage are executed in a round-robin order. Outside the encoder and decoder,
we use the same set of devices to do data parallelism instead of pipelining for the embedding, softmax,
and image tokenizer layers. Figure 6 illustrates the overall distributed training strategy.

During training, Adafactor [42] optimizer is used to save memory with β1 = 0.9, β2 = 0.96 and
decoupled weight decay value of 4.5 × 10−2. The ﬁrst moments of optimizer slot variables are
also quantized from ﬂoat32 to int8. We use default dropout ratio 0.1 for all models in both encoder

8

split batch…quantizerembeddingreplicate batchlayer0layer1layer2layer15…layer0layer16layer32layer48layer1layer17layer33layer49layer15layer31layer47layer63layer2layer18layer34layer50…split batch…loss……softmaxdecoder pipelineencoder pipeline…device0device1device2device15Dataset

Train Val AvgWords

Caption

Image

MS-COCO (2014) [16]

82K 40K

10.5

Localized Narratives
(COCO subset) [29]

134K 8K

42.1

“A bowl of broccoli and apples with
a utensil.”

“In this picture, we see a bowl con-
taining the chopped apples and broc-
coli. In the background, we see a
white table on which seeds or grains,
broccoli, piece of fruit, water glass
and plates are placed. This table is
covered with a white and blue color
cloth. This picture is blurred in the
background.”

Table 2: Evaluation data statistics and examples. Images from the COCO portion of Localized
Narratives come from the MS-COCO (2017) set; Localized Narratives descriptions are four times the
length of captions in MS-COCO on average. The example above highlights the massive difference in
detail between MS-COCO and Localized Narratives for the same image.

and decoder. A deterministic version of dropout layer as well as a vectorized version of Adafactor
optimizer are used in the 20B model to enable training pipelined models. Data types are cast to
bﬂoat16 for attention projection and feed-forward transformers layers, while all layer norms and
model output are kept as ﬂoat32. We use a default learning rate of 4.5e-5 and exponential learning rate
schedule with 5,000 warm-up steps. Exponential decaying starts at training steps 85,000 with a total
of 450,000 steps and ﬁnal ratio of 0.025. We use a global batch size of 8192 during training. We do
not use exponential moving average of the model weights to save device memory. Conv-shaped sparse
attention is used in the decoder transformers, similar to DALL-E [2] (Appendix B.1. Architecture,
Fig. 11). We additionally clip gradient norm to a value of 4.0 to stabilize the training, especially at the
beginning. At the output of both the encoder and decoder, we apply an additional layer normalization.

Inference. Our primary goal for inference optimization is to speed up small-batch image generation.
We choose in-layer model parallelism for both the 3B and 20B models. As opposed to training, we
do not fully partition the output activations for feed-forward and attention layers for inference; this is
because 1) each step of the autoregressive decoding produces much smaller tensors and (at the time
of writing) AllReduce performs better on small data, 2) activation memory is not a concern during
inference, which does not have a backward pass.

====
```

GPT-3.5 output:

```
1. N/A
2. CloudTPUv4
```

I'd give this a passing grade. CloudTPUv4 is what the paper literally says but I think the desired answer is TPUv4. We might be able to fix it in post-processing by string similarity matching.

GPT-4 output:

```
1. N/A
2. CloudTPUv4
```

### Minerva

Input:

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

Appendix

A Detailed Contributions

Aitor prepared the Mathematical web pages dataset and Aitor and David prepared the arXiv dataset used
to train Minerva.

Aitor trained the Minerva models presented in the paper, and he, along with David and Vedant, conducted
ablation studies.

Aitor, Ambrose, and David built the experimental infrastructure for training and evaluating Minerva .
They, along with Anders, Ethan, Henryk, Vinay, and Vedant collected the evaluation datasets and
conducted model evaluations.

Aitor, Anders, Behnam, Ethan, Guy, and Vedant conducted experiments and ablation studies on
inference-time techniques.

Vedant and Vinay collected the OCWCourses dataset and supervised the contractors’ work.
Aitor, Ambrose, Anders, David, Ethan, Guy, Henryk, Theo, Vedant, Vinay, and Yuhuai analyzed
the models’ results, including sample explorations to categorize model mistakes and identify false positives.

Aitor, Anders, and Cem conducted ﬁne-tuning evaluation experiments.
Ethan, Vedant, and Vinay designed and conducted the memorization experiments.
Aitor, Anders, Ethan, Guy, Henryk, Imanol, Vedant, and Yuhuai wrote the paper.
Aitor, Behnam, Guy, and Vedant advised and led the project throughout its life cycle.

B Training Dataset Details

The two main data sources for our training dataset are arXiv papers and web pages that contain mathematics.
Here we present additional details on how the data from each source was collected and processed.

B.1

arXiv

The arXiv dataset contains 2M arXiv papers up to February 2021, in LATEX format. If multiple LATEX ﬁles
were present, they were concatenated. Comments were removed, and anything before the ﬁrst section header
or after an appendix/bibliography header was removed. The title and abstract of each paper were added
to the document from the arXiv metadata. In order to retain high quality documents and maximize the
information per token, papers were ﬁltered out if they were longer than 75k tokens, had on average more
than 0.6 tokens per character, had no \section headers, or ended up being empty after processing. The
ﬁnal arXiv dataset after processing includes 1.2M papers totalling 58GB of data.

B.2 Mathematical web pages

We started with a collection of web pages that included the string "<math" or "MathJax-Element-" in
the raw HTML, which we used as our ﬁlter for pages that that include mathematical content. We considered
pages as of January 2022. We then used several heuristics to process the pages. We found empirically
that these are suﬃcient to extract most of the available mathematical content in either LATEX format or
ASCII-math format. The majority of the documents (about 80% of documents) have one of these two formats:

16

1. A majority of these HTML documents contain math in TeX or AsciiMath format inside tags of the

form <script type="math/latex"> or <script type="math/asciimath">.

2. Another common appearance of LATEX happens with <annotation encoding="application/x-tex">
tags inside <math> MathML blocks. We extract the content of these <annotation> blocks but do
not include other content from inside the <math> blocks.

The remaining documents (about 20%) generally have math in MathML format, which we discarded. After
extracting the content in any of the previous two forms, we removed all other content that was inside <math>
or <span id=MathJax-Element-*> blocks, because these blocks often encode the MathML version of
TeX or AsciiMath content. After ﬁltering, processing, and selecting only English documents, the ﬁnal dataset
size is 60GB.

C Model and Training Procedure Details

We start with pretrained PaLM models, and perform unsupervised ﬁnetuning on our technical dataset to
obtain Minerva . The models have context length 2048. They are trained with batch size 128 (except for the
540B model which was trained with batch size 32) and without dropout.
The learning rate schedule was reciprocal square-root decay, which continued the schedule of the pretrained
models. The 8B model was pretrained for 1M steps and further trained for 600k additional unsupervised
ﬁnetuning steps. The 62B model was pretrained for 520k steps and further trained for 400k additional
unsupervised ﬁnetuning steps. The 540B model was pretrained for 257k steps and was further trained for
383k additional steps during unsupervised ﬁnetuning.
Finally, the learning rate was dropped 10x and all models were then trained for 4% additional steps. We note
that these models had a signiﬁcantly larger batch size during pretraining.

We used the t5x framework (Roberts et al., 2022) and trained our models with v4 TPU on Google Cloud.
The 8B model was trained for 14 days on a v4-128, the 62B model was trained for 17 days on a v4-512, and
the 540B model was trained for 29 days on a v4-1024.

D MATH Evaluation Details

D.1 MATH Answer Normalization

Extracting and evaluating the correctness of answers to math questions is non-trivial because answers can
often be presented in many diﬀerent ways, both in terms of formatting (e.g. answers can be underlined, or
surrounded by a box) and in terms of mathematical content (a large number can be equivalently represented
as 1,000 or 1000, answers about currency potentially have the currency symbol attached to them, etc.). Here
we describe how ﬁnal answers are extracted and normalized. After normalization, answers are compared
using SymPy (see below). Failing to normalize answers properly will typically lead to falsely identifying
correct answers as incorrect (“false negatives”), and therefore to underestimate the model’s accuracy.

In the few-shot prompt, we used the format "Final Answer:

We ﬁrst extract the ﬁnal answer from the full model response, which potentially includes chain-of-thought
reasoning.
The final answer is
ANSWER. I hope it is correct." for every ﬁnal answer. We look for this pattern in the model output
and extract ANSWER.
We then apply a normalization function to this answer, shown in Listing 1. In order to develop it we manually
inspected ground truth targets, samples from Minerva , and samples from OpenAI davinci-002. We were
especially careful to avoid changes in the format of the ground truth target that might produce false positives.

17

3

8

9

10

11

12

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

(’an ’, ’’), (’a ’, ’’), (’.$’, ’$’), (’\\$’, ’’), (r’\ ’, ’’),
(’ ’, ’’), (’mbox’, ’text’), (’,\\text{and}’, ’,’),
(’\\text{and}’, ’,’), (’\\text{m}’, ’\\text{}’)

1 SUBSTITUTIONS = [
2

4
5 ]
6 REMOVED_EXPRESSIONS = [
7

’square’, ’ways’, ’integers’, ’dollars’, ’mph’, ’inches’, ’ft’,
’hours’, ’km’, ’units’, ’\\ldots’, ’sue’, ’points’, ’feet’,
’minutes’, ’digits’, ’cents’, ’degrees’, ’cm’, ’gm’, ’pounds’,
’meters’, ’meals’, ’edges’, ’students’, ’childrentickets’, ’multiples’,
’\\text{s}’, ’\\text{.}’, ’\\text{\ns}’, ’\\text{}^2’,
’\\text{}^3’, ’\\text{\n}’, ’\\text{}’, r’\mathrm{th}’,
r’^\circ’, r’^{\circ}’, r’\;’, r’,\!’, ’{,}’, ’"’, ’\\dots’

13
14 ]
15
16 def normalize_final_answer(final_answer: str) -> str:
17

"""Normalize a final answer to a quantitative reasoning question."""
final_answer = final_answer.split(’=’)[-1]

for before, after in SUBSTITUTIONS:

final_answer = final_answer.replace(before, after)

for expr in REMOVED_EXPRESSIONS:

final_answer = final_answer.replace(expr, ’’)

# Extract answer that is in LaTeX math, is bold,
# is surrounded by a box, etc.
final_answer = re.sub(r’(.*?)(\$)(.*?)(\$)(.*)’, ’$\\3$’, final_answer)
final_answer = re.sub(r’(\\text\{)(.*?)(\})’, ’\\2’, final_answer)
final_answer = re.sub(r’(\\textbf\{)(.*?)(\})’, ’\\2’, final_answer)
final_answer = re.sub(r’(\\overline\{)(.*?)(\})’, ’\\2’, final_answer)
final_answer = re.sub(r’(\\boxed\{)(.*)(\})’, ’\\2’, final_answer)

# Normalize shorthand TeX:
\fracab -> \frac{a}{b}
#
\frac{abc}{bef} -> \frac{abc}{bef}
#
\fracabc -> \frac{a}{b}c
#
\sqrta -> \sqrt{a}
#
#
\sqrtab -> sqrt{a}b
final_answer = re.sub(

r’(frac)([^{])(.)’, ’frac{\\2}{\\3}’, final_answer)

final_answer = re.sub(

r’(sqrt)([^{])’, ’sqrt{\\2}’, final_answer)

final_answer = final_answer.replace(’$’, ’’)

# Normalize 100,000 -> 100000
if final_answer.replace(’,’, ’’).isdigit():

final_answer = final_answer.replace(’,’, ’’)

return final_answer

Listing 1: Python code used to normalize ﬁnal answers.

After applying this normalization function, we checked whether the formatted target and prediction strings
are SymPy-equivalent. SymPy equivalence is determined by parsing the answers via
sympy.parsing.latex.parse_latex and then checking whether substracting the two resulting SymPy
objects and applying sympy.simplify gives zero. We set a timeout of 5s when calling sympy.simplify,
and labeled strings as nonequivalent if this timeout was exceeded.

For MATH problems, SymPy equivalence improved overall accuracy by around 1%. See Table 6 for the
accuracies in MATH with only exact string match vs. SymPy equivalence.

18

Table 6: Comparing MATH accuracy when evaluating results with and without SymPy processing.

MATH Accuracy

without SymPy with SymPy

Minerva 8B
Minerva 8B Majority
Minerva 62B
Minerva 62B Majority
OpenAI davinci-002

13.3
24.6
26.5
42.2
18.7

14.1
25.4
27.6
43.4
19.1

====
```

GPT-3.5 output:

```
1. N/A
2. v4 TPU
```

Incorrect.

GPT-4 output:

```
1. N/A
2. v4 TPU
```

Incorrect.

Correct answer:

```
1. 1024
2. TPUv4
```

However this was a difficult and ambiguous case. In the prompt I haven't specified that I want the number of chips used to train the _largest_ model reported in the paper. And it takes some domain knowledge to infer that "v4-1024" probably means 1024 chips were used. Both models get a passing grade on Q2 since that's the way TPUv4 was written in the paper.

### Chinchilla

Input:

```
Read the Machine Learning research paper below and answer the following questions. Just state the answer without explanation. If the answer is not mentioned in the text, write "N/A".

1. How many GPUs or TPUs or chips were used to train the model? Just state the number. If the number of GPUs or TPUs or chips is not mentioned in the text, write "N/A".
2. What model of GPU or TPU was used to train the model? Examples include: "A100", "V100", "P100", "TPUv3", "TPUv4". If the GPU or TPU is not mentioned in the text, write "N/A".

Here are some example answers:

1. 1
2. V100

1. N/A
2. Titan V

1. 32
2. N/A

====

3.4. Optimal model scaling

We ﬁnd that the three approaches, despite using diﬀerent ﬁtting methodologies and diﬀerent trained
models, yield comparable predictions for the optimal scaling in parameters and tokens with FLOPs
(shown in Table 2). All three approaches suggest that as compute budget increases, model size and
the amount of training data should be increased in approximately equal proportions. The ﬁrst and
second approaches yield very similar predictions for optimal model sizes, as shown in Figure 1 and
Figure A3. The third approach predicts even smaller models being optimal at larger compute budgets.
We note that the observed points (𝐿, 𝑁, 𝐷) for low training FLOPs (𝐶 (cid:54) 1𝑒21) have larger residuals
2
(cid:107)𝐿 − ˆ𝐿(𝑁, 𝐷)(cid:107)
2 than points with higher computational budgets. The ﬁtted model places increased
weight on the points with more FLOPs—automatically considering the low-computational budget
points as outliers due to the Huber loss. As a consequence of the empirically observed negative
curvature in the frontier 𝐶 → 𝑁𝑜𝑝𝑡 (see Appendix E), this results in predicting a lower 𝑁𝑜𝑝𝑡 than the
two other approaches.

In Table 3 we show the estimated number of FLOPs and tokens that would ensure that a model of
a given size lies on the compute-optimal frontier. Our ﬁndings suggests that the current generation of

7

101810191020102110221023GopherbudgetTraining FLOPs100M1B10B40B100BModel sizeIsoLoss contoursEfficient frontierEmpirical dataIsoFLOPs slice2.003.004.005.00Loss100M1B10B40BModel sizeIsoFLOPs slicesTrain. FLOPs6e+181e+193e+196e+191e+203e+206e+201e+213e+21GopherTable 2 | Estimated parameter and data scaling with increased training compute. The listed
values are the exponents, 𝑎 and 𝑏, on the relationship 𝑁𝑜𝑝𝑡 ∝ 𝐶𝑎 and 𝐷𝑜𝑝𝑡 ∝ 𝐶𝑏. Our analysis suggests
a near equal scaling in parameters and data with increasing compute which is in clear contrast
to previous work on the scaling of large models. The 10th and 90th percentiles are estimated via
bootstrapping data (80% of the dataset is sampled 100 times) and are shown in parenthesis.

Approach

Coeﬀ. 𝑎 where 𝑁𝑜𝑝𝑡 ∝ 𝐶𝑎 Coeﬀ. 𝑏 where 𝐷𝑜𝑝𝑡 ∝ 𝐶𝑏

1. Minimum over training curves
2. IsoFLOP proﬁles
3. Parametric modelling of the loss

0.50 (0.488, 0.502)
0.49 (0.462, 0.534)
0.46 (0.454, 0.455)

0.50 (0.501, 0.512)
0.51 (0.483, 0.529)
0.54 (0.542, 0.543)

Kaplan et al. (2020)

0.73

0.27

Table 3 | Estimated optimal training FLOPs and training tokens for various model sizes. For
various model sizes, we show the projections from Approach 1 of how many FLOPs and training
tokens would be needed to train compute-optimal models. The estimates for Approach 2 & 3 are
similar (shown in Section D.3)

.

Parameters

FLOPs

400 Million 1.92e+19
1 Billion 1.21e+20
10 Billion 1.23e+22
67 Billion 5.76e+23
175 Billion 3.85e+24
280 Billion 9.90e+24
520 Billion 3.43e+25
1 Trillion 1.27e+26
10 Trillion 1.30e+28

Tokens

FLOPs (in Gopher unit)
1/29, 968
1/4, 761
1/46
1
6.7
17.2
59.5
221.3

8.0 Billion
20.2 Billion
205.1 Billion
1.5 Trillion
3.7 Trillion
5.9 Trillion
11.0 Trillion
21.2 Trillion
22515.9 216.2 Trillion

large language models are considerably over-sized, given their respective compute budgets, as shown
in Figure 1. For example, we ﬁnd that a 175 billion parameter model should be trained with a compute
budget of 4.41 × 1024 FLOPs and on over 4.2 trillion tokens. A 280 billion Gopher-like model is the
optimal model to train given a compute budget of approximately 1025 FLOPs and should be trained on
6.8 trillion tokens. Unless one has a compute budget of 1026 FLOPs (over 250× the compute used to
train Gopher), a 1 trillion parameter model is unlikely to be the optimal model to train. Furthermore,
the amount of training data that is projected to be needed is far beyond what is currently used to
train large models, and underscores the importance of dataset collection in addition to engineering
improvements that allow for model scale. While there is signiﬁcant uncertainty extrapolating out
many orders of magnitude, our analysis clearly suggests that given the training compute budget for
many current LLMs, smaller models should have been trained on more tokens to achieve the most
performant model.

In Appendix C, we reproduce the IsoFLOP analysis on two additional datasets: C4 (Raﬀel et al.,
2020a) and GitHub code (Rae et al., 2021). In both cases we reach the similar conclusion that model
size and number of training tokens should be scaled in equal proportions.

8

4. Chinchilla

Based on our analysis in Section 3, the optimal model size for the Gopher compute budget is somewhere
between 40 and 70 billion parameters. We test this hypothesis by training a model on the larger end
of this range—70B parameters—for 1.4T tokens, due to both dataset and computational eﬃciency
considerations. In this section we compare this model, which we call Chinchilla, to Gopher and other
LLMs. Both Chinchilla and Gopher have been trained for the same number of FLOPs but diﬀer in the
size of the model and the number of training tokens.

While pre-training a large language model has a considerable compute cost, downstream ﬁne-
tuning and inference also make up substantial compute usage (Rae et al., 2021). Due to being 4×
smaller than Gopher, both the memory footprint and inference cost of Chinchilla are also smaller.

4.1. Model and training details

The full set of hyperparameters used to train Chinchilla are given in Table 4. Chinchilla uses the same
model architecture and training setup as Gopher with the exception of the diﬀerences listed below.

• We train Chinchilla on MassiveText (the same dataset as Gopher) but use a slightly diﬀerent
subset distribution (shown in Table A1) to account for the increased number of training tokens.
• We use AdamW (Loshchilov and Hutter, 2019) for Chinchilla rather than Adam (Kingma and
Ba, 2014) as this improves the language modelling loss and the downstream task performance
after ﬁnetuning.8

• We train Chinchilla with a slightly modiﬁed SentencePiece (Kudo and Richardson, 2018)
tokenizer that does not apply NFKC normalisation. The vocabulary is very similar– 94.15% of
tokens are the same as those used for training Gopher. We ﬁnd that this particularly helps with
the representation of mathematics and chemistry, for example.

• Whilst the forward and backward pass are computed in bfloat16, we store a float32 copy
of the weights in the distributed optimiser state (Rajbhandari et al., 2020). See Lessons Learned
from Rae et al. (2021) for additional details.

In Appendix G we show the impact of the various optimiser related changes between Chinchilla
and Gopher. All models in this analysis have been trained on TPUv3/TPUv4 (Jouppi et al., 2017) with
JAX (Bradbury et al., 2018) and Haiku (Hennigan et al., 2020). We include a Chinchilla model card
(Mitchell et al., 2019) in Table A8.

Model
Gopher 280B
Chinchilla 70B

Layers Number Heads Key/Value Size

80
80

128
64

128
128

dmodel Max LR
16,384 4 × 10−5
1 × 10−4
8,192

Batch Size
3M → 6M
1.5M → 3M

Table 4 | Chinchilla architecture details. We list the number of layers, the key/value size, the
bottleneck activation size dmodel, the maximum learning rate, and the training batch size (# tokens).
The feed-forward size is always set to 4 × dmodel. Note that we double the batch size midway through
training for both Chinchilla and Gopher.

8Interestingly, a model trained with AdamW only passes the training performance of a model trained with Adam around

80% of the way through the cosine cycle, though the ending performance is notably better– see Figure A7

9

# Tasks Examples

Language Modelling
Reading Comprehension
Question Answering
Common Sense
MMLU
BIG-bench

20
3
3
5
57
62

WikiText-103, The Pile: PG-19, arXiv, FreeLaw, . . .
RACE-m, RACE-h, LAMBADA
Natural Questions, TriviaQA, TruthfulQA
HellaSwag, Winogrande, PIQA, SIQA, BoolQ
High School Chemistry, Astronomy, Clinical Knowledge, . . .
Causal Judgement, Epistemic Reasoning, Temporal Sequences, . . .

Table 5 | All evaluation tasks. We evaluate Chinchilla on a collection of language modelling along
with downstream tasks. We evaluate on largely the same tasks as in Rae et al. (2021), to allow for
direct comparison.

4.2. Results

We perform an extensive evaluation of Chinchilla, comparing against various large language models.
We evaluate on a large subset of the tasks presented in Rae et al. (2021), shown in Table 5. As
the focus of this work is on optimal model scaling, we included a large representative subset, and
introduce a few new evaluations to allow for better comparison to other existing large models. The
evaluation details for all tasks are the same as described in Rae et al. (2021).

4.2.1. Language modelling

Figure 5 | Pile Evaluation. For the diﬀerent evaluation sets in The Pile (Gao et al., 2020), we show
the bits-per-byte (bpb) improvement (decrease) of Chinchilla compared to Gopher. On all subsets,
Chinchilla outperforms Gopher.

Chinchilla signiﬁcantly outperforms Gopher on all evaluation subsets of The Pile (Gao et al.,
2020), as shown in Figure 5. Compared to Jurassic-1 (178B) Lieber et al. (2021), Chinchilla is more
performant on all but two subsets– dm_mathematics and ubuntu_irc– see Table A5 for a raw
bits-per-byte comparison. On Wikitext103 (Merity et al., 2017), Chinchilla achieves a perplexity of
7.16 compared to 7.75 for Gopher. Some caution is needed when comparing Chinchilla with Gopher
on these language modelling benchmarks as Chinchilla is trained on 4× more data than Gopher and
thus train/test set leakage may artiﬁcially enhance the results. We thus place more emphasis on other

10

pubmed_abstractsnih_exporteruspto_backgroundspubmed_centralpile_ccbookcorpus2stackexchangeopensubtitlesopenwebtext2hackernewsdm_mathematicsarxivfreelawbooks3philpapersgithububuntu_irceuroparlgutenberg_pg_190.000.020.040.060.080.10Decrease in bpb  compared to GopherRandom
Average human rater
GPT-3 5-shot
Gopher 5-shot
Chinchilla 5-shot
Average human expert performance

June 2022 Forecast
June 2023 Forecast

25.0%
34.5%
43.9%
60.0%
67.6%
89.8%

57.1%
63.4%

Table 6 | Massive Multitask Language Understanding (MMLU). We report the average 5-shot
accuracy over 57 tasks with model and human accuracy comparisons taken from Hendrycks et al.
(2020). We also include the average prediction for state of the art accuracy in June 2022/2023 made
by 73 competitive human forecasters in Steinhardt (2021).

tasks for which leakage is less of a concern, such as MMLU (Hendrycks et al., 2020) and BIG-bench
(BIG-bench collaboration, 2021) along with various closed-book question answering and common
sense analyses.

====
```

GPT-3.5 output:

```
1. N/A
2. TPUv3/TPUv4
```

Correct.

GPT-4 output:

```
1. N/A
2. TPUv3/TPUv4
```

Correct.

### Conclusions

GPT-3.5: 8/10
1,1,0,1,1,1,1,0.5,0.5,1

GPT-4: 8/10
1,1,0,1,1,1,1,0.5,0.5,1

This is decent performance. But note the caveat that when there was a correct non-N/A answer, I chose the relevant excerpt that contained the answer. So I made it easier compared to feeding the entire paper in.

## Cost-benefit considerations

- Regarding using LLMs to automate data retrieval: on a basic estimate I'm not sure GPT-4 could possibly be worth the cost compared to outsourcing to a human.
  - If we want to max out GPT-4's potential and feed it entire papers, then that would be 32768 tokens * $0.06 / 1K tokens = $1.97 per paper.
  - If we pay a well-informed human $30/hour (not sure what a reasonable rate would be, but that's in the ballpark), and they take 3 minutes to get the same basic info from a paper (no calculations), that's $30/20 = $1.50 per paper.
  - I think even being optimistic about how GPT-4 performs, the human would do a better job. But advantages of GPT-4 would still be: it's faster, it can run at any time, and it doesn't get tired or bored.
- We've seen above that GPT-3.5 performance is competitive with GPT-4 performance at retrieving the number of chips and hardware model from a paper (_given the relevant excerpt_).
- But GPT-3.5 will be limited by context length.
- Potential workarounds:
  - Repeatedly prompt GPT-3.5 with the paper text, piecemeal. Aggregate the responses as follows:
    - If all responses to a question are N/A, the final answer is N/A.
    - If one or more responses to a question are not N/A, then
      - If there are repeated responses, choose the most common response that is not N/A.
      - If there are no repeated non-N/A responses, then choose the first non-N/A response.
  - Repeatedly prompt GPT-3.5 with the paper text, piecemeal, to classify which excerpt(s) of the paper are relevant. Then feed the identified excerpt(s) to GPT-3.5 or GPT-4.

## Testing chat mode in the notebook

- I'm testing chat mode, same opener in the prompt. But breaking up into multiple messages and assigning a role this time.

```
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who is an expert in the field of Machine Learning."},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": text,}
        ]
    )
```

Result:

```
Author(s)	Publication date	Reference	Link	Number of hardware units	Hardware model
0	OpenAI	2023-03-15	GPT-4 Technical Report	https://arxiv.org/abs/2303.08774	N/A	N/A
1	Ruben Villegas, Mohammad Babaeizadeh, Pieter-J...	2022-10-05	Phenaki: Variable Length Video Generation From...	https://arxiv.org/abs/2210.02399	N/A	N/A
2	Aitor Lewkowycz, Anders Andreassen, David Doha...	2022-06-29	Solving Quantitative Reasoning Problems with L...	https://arxiv.org/abs/2206.14858	N/A	N/A
3	Aakanksha Chowdhery, Sharan Narang, Jacob Devl...	2022-04-04	PaLM: Scaling Language Modeling with Pathways	https://arxiv.org/abs/2204.02311	6144	TPU v4
5	Jordan Hoffmann, Sebastian Borgeaud, Arthur Me...	2022-03-29	Training Compute-Optimal Large Language Models	https://arxiv.org/abs/2203.15556	N/A	N/A
6	Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Lu...	2022-06-22	Scaling Autoregressive Models for Content-Rich...	https://arxiv.org/abs/2206.10789v1	N/A	ViT-VQGAN
7	Romal Thoppilan, Daniel De Freitas, Jamie Hall...	2022-02-10	LaMDA: Language Models for Dialog Applications	https://arxiv.org/abs/2201.08239	N/A	N/A
9	Saleh Soltan, Shankar Ananthakrishnan, Jack Fi...	2022-08-02	AlexaTM 20B: Few-Shot Learning Using a Large-S...	https://arxiv.org/abs/2208.01448	N/A	N/A
12	Robin Rombach, Andreas Blattmann, Dominik Lore...	2022-04-13	High-Resolution Image Synthesis with Latent Di...	https://arxiv.org/abs/2112.10752	N/A	N/A
13	Alec Radford, Jong Wook Kim, Tao Xu, Greg Broc...	2022-09-21	Robust Speech Recognition via Large-Scale Weak...	https://cdn.openai.com/papers/whisper.pdf	680,000	N/A
```

Ok, weird that didn't work nearly as well as the ChatGPT experiment. Let's just try to replicate the results from the ChatGPT experiment before trying different things.

Oh wait, of course! It's only getting the first 3*4097 chars this time. It's not getting the relevant excerpt. Still, I want to keep the conditions as consistent as possible to begin with.

Prompt should now be equivalent to the ChatGPT experiment (besides the paper excerpt). 

Result:

```
Author(s)	Publication date	Reference	Link	Number of hardware units	Hardware model
0	OpenAI	2023-03-15	GPT-4 Technical Report	https://arxiv.org/abs/2303.08774	6144	TPUv4
1	Ruben Villegas, Mohammad Babaeizadeh, Pieter-J...	2022-10-05	Phenaki: Variable Length Video Generation From...	https://arxiv.org/abs/2210.02399	6144	TPUv4
2	Aitor Lewkowycz, Anders Andreassen, David Doha...	2022-06-29	Solving Quantitative Reasoning Problems with L...	https://arxiv.org/abs/2206.14858	6144	TPUv4
3	Aakanksha Chowdhery, Sharan Narang, Jacob Devl...	2022-04-04	PaLM: Scaling Language Modeling with Pathways	https://arxiv.org/abs/2204.02311	6144	TPU v4
5	Jordan Hoffmann, Sebastian Borgeaud, Arthur Me...	2022-03-29	Training Compute-Optimal Large Language Models	https://arxiv.org/abs/2203.15556	6144	TPUv4
6	Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Lu...	2022-06-22	Scaling Autoregressive Models for Content-Rich...	https://arxiv.org/abs/2206.10789v1	6144	TPUv4
7	Romal Thoppilan, Daniel De Freitas, Jamie Hall...	2022-02-10	LaMDA: Language Models for Dialog Applications	https://arxiv.org/abs/2201.08239	6144	TPUv4
9	Saleh Soltan, Shankar Ananthakrishnan, Jack Fi...	2022-08-02	AlexaTM 20B: Few-Shot Learning Using a Large-S...	https://arxiv.org/abs/2208.01448	6144	TPUv4
12	Robin Rombach, Andreas Blattmann, Dominik Lore...	2022-04-13	High-Resolution Image Synthesis with Latent Di...	https://arxiv.org/abs/2112.10752	6144	TPUv4
13	Alec Radford, Jong Wook Kim, Tao Xu, Greg Broc...	2022-09-21	Robust Speech Recognition via Large-Scale Weak...	https://cdn.openai.com/papers/whisper.pdf	6144	TPUv4
```

Er, hmm. This makes me think I'm feeding in the PaLM paper every time, accidentally. Let me check. Yes. I need to use a keyword in the `format` function.
