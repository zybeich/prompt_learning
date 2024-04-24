# Prompt Learning
Prompt learning, learned automatically during the fine-tuning stage, was first used in NLP followed by the adaptation in V-L and vision-only models. [Model Repromgramming](https://arxiv.org/abs/2202.10629) for visual tasks is visual prompting.



## For different models: vision-only models/vision-language/other foundation models
### Vision-Language models
V-L models: CLIP, ALIGN, LiT, FILIP, and Florence. Focus on zero-shot and few-shot setting.
[CoOp](https://arxiv.org/pdf/2109.01134.pdf): (Context Optimization) text prompt tuning.

[CoCoOp](https://arxiv.org/pdf/2203.05557.pdf): (Conditional Context Optimization) trains a meta net (CNN) to generate input-specific text prompts.

[MaPLe](https://arxiv.org/pdf/2210.03117.pdf): (Multi-modal prompt learning) learns a linear layer to transform the text prompt token to a visual prompt token.

[PLOT](https://openreview.net/pdf?id=zqwryBoXYnh): (Prompt learning with Optimal Transport) utilizes the OT distance between a bunch of text prompts for each class and local visual features before attention pooling to learn text prompts. 

[ProDA](https://arxiv.org/pdf/2205.03340.pdf): (Prompt distribution learning) predefines a parametric distribution and fits the parameters of this distribution during training. (**Both PLOT and ProDA are trying to learn more diverse text prompts**)

[PromptSRC](https://arxiv.org/pdf/2307.06948.pdf):  mutual agreement maximazition + cross entropy


### Vision-only models
[VPT](https://arxiv.org/abs/2203.12119) learns prompts with labeled data for the downstream tasks: Shallow Prompt and Deep Prompt. It empirically shows that adding prompts in token space is better than in input space.

[AutoVP](https://arxiv.org/abs/2310.08381): an end-to-end prompt learning framework including visual prompt and label mapping. It adds prompts to images directly and mainly focuses on CNNs.

[DoPrompt](https://arxiv.org/abs/2208.08914): learns visual prompts for each source domain and a prompt adapter during training time. Target representations are fed into the prompt adapter, which then determines the weights used to compute the appropriate prompt for each target input. (**a DG method**)


## At test time (beyond supervision)
### Vision-Language models
|                            method                            |  src/ref data  | text prompting | visual prompting | input-specific  prompt |   **Objective Function**          |                                                              |
| :----------------------------------------------------------: | :----------: | :------------: | :--------------: | :--------------------: | -------------------------------- | ------------------------------------------------------------ |
| Unsupervised Prompt Learning ([UPL](https://arxiv.org/pdf/2204.03649.pdf)) |              |  $\checkmark$  |                  |                        | Cross-entropy with pseudo labels | Use the pseudo label to select the top-K confident samples per class, then minimize the cross-entropy loss |
| Test-Time Prompt Tuning ([TPT](https://proceedings.neurips.cc/paper_files/paper/2022/file/5bf2b802e24106064dc547ae9283bb0c-Paper-Conference.pdf)) |              |  $\checkmark$  |                  |      $\checkmark$      | marginal entropy                 | Select the augmented test images with low entropy, then minimize the average entropy |
|    [SwapPrompt](https://openreview.net/pdf?id=EhdNQiOWgQ)    |              |  $\checkmark$  |                  |                        | Pseudo cross entropy + swap      | Use two augmentations of test input, swap them to compute cross cross entropy |
|   [PromptAlign](https://openreview.net/pdf?id=CusNOTRkQw)    | $\checkmark$ |  $\checkmark$  |   $\checkmark$   |      $\checkmark$      | ent+align                        | Use reference  dataset-ImageNet to align the tokens' statistics of the visual encoder. Text prompt is learned by entropy reduction. |

### Vision-only models
[DePT](https://arxiv.org/pdf/2210.04831.pdf): need access to the training process to learn source prompts. Then adapt the learned prompt and linear classifier to the unseen target task via a memory bank.


         
## For different model properties: robustness, fairness, privacy, and uncertainty
### Robustness
[AVP](https://arxiv.org/pdf/2210.06284.pdf): (Adversarial visual prompt) input space, class-wise prompt.

[APT](https://arxiv.org/pdf/2403.01849.pdf): (Adversarial prompt tuning) text prompt/context, universal/class-wise, for V-L models. (**The learned context is not very meaningful despite the improvement**)


### Fairness
[Fairness Reprogramming](https://arxiv.org/pdf/2209.10222.pdf)


### Privacy
[Exploring the Benefits of Visual Prompting in Differential Privacy](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Exploring_the_Benefits_of_Visual_Prompting_in_Differential_Privacy_ICCV_2023_paper.pdf)
[Quantifying Privacy Risks of Prompts in Visual Prompt Learning](https://www.usenix.org/system/files/sec24summer-prepub-176-wu-yixin.pdf)

### Uncertainty
[Neural Clamping: Joint Input Perturbation and Temperature Scaling for Neural Network Calibration](https://arxiv.org/abs/2209.11604)
[C-TPT](https://openreview.net/pdf?id=jzzEHTBFOT): minimize Average Text Feature Dispersion + TPT loss (variance) to improve ECE. 


# Related to Diffusion Models
[GDA: Generalized Diffusion for Robust Test-time Adaptation](https://arxiv.org/pdf/2404.00095.pdf): OOD classification, the gradient of (marginal entropy + style loss (CLIP) + content loss (UNet)) will be used in the denoise process.



# Related to Optimal Transport
[Otter](https://arxiv.org/pdf/2404.08461.pdf): (Optimal Transport Adapter) focuses on label shift and requires the estimate of the label distribution to improve the zero-shot classification.



# Possible research questions
1. Adversarial universal prompt token tuning. $\gamma^* = \min_\gamma  \mathbb{E} \max_{\|x-x'\|<\delta}\ell(x; \gamma) $.
2. DRO with prompt learning. $\gamma^* = \min_\gamma \sup_{\|P-P'\|<\delta}\mathbb{E}_P' \ell(x; \gamma)$ in representation space could be used as a transferability metric. Or in token space?
3. Pure visual prompt at test time. Intralabel/Marginal+Interlabel, target data selection with entropy, $lambda$ selection with entropy, corrupted data, learning the prompt from sub-class (base to new setting).
4. V-L. V: marginal distance + separability (can be used for any downstream task), L: entropy. 
