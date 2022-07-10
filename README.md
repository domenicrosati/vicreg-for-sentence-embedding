# VicReg loss and JEPA for Sentence Embeddings
=================================

This repo is a sketch illustrating how a Joint-embedding predictive architecture (JEPA) might be used for sequence modeling.
Specifically it is a sketch of how a JEPA trained wth VicReg loss could be used to produce sentence embeddings.

Where X and X' are a text sequence and a view of a text sequence X'.
The goal is that the trained encoder should minimize the Energy of X,X'.

Please see the following papers for background on the JEPA:
- (A Path Towards Autonomous Machine Intelligence)[https://openreview.net/pdf?id=BZ5a1r-kVsf]
- (VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning)[https://openreview.net/forum?id=xm6YD62D1Ub]
- (Intra-Instance VICReg: Bag of Self-Supervised Image Patch Embedding)[https://arxiv.org/abs/2206.08954]

**Note:** If you use the code from this repo or inspired by it, please let me know!
I'd love to hear from you and work on this further. A mention in acknowledgements is also appreciated ;) .

## Details

See `vicreg.py` which implements the JEPA and VicReg loss for the (sentence transformer framework)[https://www.sbert.net/].
Specifically please see the following components:
- The expander (which provides a projection on the embedding space to compute loss)
- VicReg loss
- WordCropDataset (which provides the data for the JEPA)

WordCrop produces a view of X, X' that is a random crop (selection of N words) of the sentence.


## Results

While the JEPA works and produces results similar or slighly better than MLM.
(1) **It is not clear if the JEPA as formulated is better than MLM**.
(2) **I have found no configuration that beats contrastive methods or TSDAE.**
So I have not reported the results here since they are not that meaningful, other than JEPA works "OK".
The best result I was able to get through fiddling with hyperparameters was:

**56.04 MAP** on Ask Ubuntu test.

| Model | MAP-Score on test set |
| ---- | :----: |
| TSDAE (bert-base-uncased) | 59.4 |
| JEPA w VicREG (bert-base-uncased) | 56.04 |
| **pretrained SentenceTransformer models** | |
| nli-bert-base | 50.7 |
| paraphrase-distilroberta-base-v1 | 54.8 |
| stsb-roberta-large | 54.6 |

----------------------


You can also reset the bert-base-uncased model and finetune with JEPA.
Surprisingly, the JEPA works about the same as MLM under the STSb task.

Based on these JEPA with VicReg loss as formulated here is interesting because it doesn't require contrastive objectives or a generative model achieving results on par with MLM (though not as great as contrastive.)

Notes:
- JEPA appears to do best with the biggest batch size and expander dimensions as possible.
- JEPA appears to do best with Lambda and Mu set to 25 and 25. Making Mu smaller tends to result in collapse.

## Further work

While the benefits of JEPA are clear in the language domain:
- JEPA does not require generative models or pretext objectives, meaning we **could** build much smaller and less complex models that perform as well as others.
- VICReg, not requiring, a contrastive objective **could** be more sample effeciency.

My conjecture is that, in the language domain, the VICReg loss is going to need auxiliary pretext tasks in order to align the embeddings with utility for language.
This is mentioned as potentially necessary in both:
- (A Path Towards Autonomous Machine Intelligence)[https://openreview.net/pdf?id=BZ5a1r-kVsf]
- (VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning)[https://openreview.net/forum?id=xm6YD62D1Ub]

Without auxiliary objectives that speak to **how you'd like your embeddings to be meaningful**, the embeddings could be highly informative, but not under conditions
that are useful to downstream tasks. I'd suggest exploring lingustically informated pretext tasks like the following:
- Can the projected embeddings predict language transformations like synonym/hyponym/antonym swap, masking, noise corruption.

If you'd like to explore this conjecture with a collaborator feel free to let me know as I am actively investigating it!
