# OwensLab / Community Forensics 224

- **Model Identifier:** `OwensLab/commfor-model-224`
- **Architecture:** Vision Transformer (`vit_small_patch16_224.augreg_in21k_ft_in1k`) via `timm`
- **Classification Head:** `Linear(in_features=384, out_features=1, bias=True)`
- **Input:** $224 \times 224$ RGB image tensor
- **Normalization:** ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Output:** Single logit ($\sigma(\text{logit}) \in [0.0, 1.0]$ representing synthetic/AI score)
- **Label Orientation:** `0 = Real`, `1 = Synthetic / AI-generated`
- **Upstream Paper:** *Community Forensics: Using Community Feedback to Detect AI-Generated Images* (arXiv:2411.04125)
- **Upstream Code:** [https://github.com/JeongsooP/Community-Forensics](https://github.com/JeongsooP/Community-Forensics)
- **Upstream Model Card:** [https://huggingface.co/OwensLab/commfor-model-224](https://huggingface.co/OwensLab/commfor-model-224)
- **License:** MIT License (Permissive open source)

### Application Context in TRUSTLENS
This pretrained checkpoint is utilized as an authoritative multi-scale patch forensic detector within the TRUSTLENS forensic verification pipeline. TRUSTLENS preserves native image geometry, extracts pixel-level noise heuristics, computes multi-patch consensus via trimmed-mean and median aggregation, and enforces an INCONCLUSIVE decision state when evidence is weak or conflicting.
