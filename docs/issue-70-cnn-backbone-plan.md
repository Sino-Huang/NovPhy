# #70: deadline-scoped CNN visual-backbone replacement (v6)

## Plan recorded before implementation

Replace the current fixed RGB/Sobel + flattened-image MLP visual front end with
one small learned RGB CNN. The v5 objective audit remains a preserved negative
result (informative-state regret 0.641975; top-3 0.604938). We have not established
that CEM is at fault: that run stopped while scoring actual endpoint images.

### Deliberately small change

- Input: the same prepared 96x64 RGB images, scaled to [-1,1]. No new capture,
  image regeneration, pretrained-weight download, resolution sweep or augmentation.
- CNN: three 3x3 convolutions, channels 16/32/64, strides 1/2/2, padding 1,
  GroupNorm (4 groups) and LeakyReLU(0.1) after each. The first layer retains
  full resolution to avoid immediately discarding small-object detail.
- Adaptive average pool to 4x6, flatten, linear projection to 128, LayerNorm and
  LeakyReLU(0.1). Keep the existing nonlinear slot fusion and prediction heads.
- Keep all 18 authored slots and the 236-value downstream carrier interface.
  New architecture/parser/carrier identities prevent mixing CNN and MLP weights.
- Train from scratch: one fixed seed (7002001), 20 epochs, batch 128, AdamW
  learning rate 0.001, existing weight decay and gradient clipping. Retain v5's
  fixed training-only balanced pig/block losses and all other supervision.
- Reuse the existing training shards read-only. Keep old checkpoints and results;
  write only to new v6 artifact/report/video locations. No new hashes.

### Testing and compute boundary

1. Test output shapes, trainable convolution gradients, independent object-presence
   learning, checkpoint incompatibility/reload and training-only exposure checks.
2. Run a bounded real CUDA training/save/reload/resume smoke and the no-write
   dry run. Report parameter count and measured speed; leave full training to the
   operator with unbuffered progress, epoch ETA and per-class diagnostics.
3. Provide a perception-only command that trains the CNN, parses the existing
   calibration endpoints and runs Stage A, then stops **even if it passes**.
   This prevents an automatic expensive world-model retrain near the deadline.
4. Report pig/block count errors and the unchanged objective-ranking gate against
   the preserved v5 report. This is an exploratory historical comparison, not a
   matched causal backbone ablation: initialization/training schedules differ.
5. Only if Stage A passes, let the operator explicitly resume the existing full
   workflow. Downstream models must be retrained on CNN-produced carriers; old
   MLP-carrier world-model checkpoints are not interchangeable.

No CEM redesign, objective rounding/reweighting, gate relaxation, ensemble sweep,
new benchmark access or post-hoc selection of favorable calibration epochs.
The already-opened calibration data can guide this exploratory repair but cannot
serve as fresh confirmatory evidence. If the fixed CNN run fails, record that
result and reassess; do not automatically launch another search. #70 remains open
and #64 remains unauthorized. One ticket is sufficient for this bounded change.

### Intended operator entry point

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-perception --device cuda \
  2>&1 | tee -a data/issue-70-parser-repair-v6.log
```

The implementation and verified follow-up commands will be recorded in a further
comment after tests complete. Full experiment results are not claimed by this plan.
