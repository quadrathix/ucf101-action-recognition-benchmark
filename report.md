Evaluation Method: All three architectures were pretrained on Kinetics-400. R(2+1)D and I3D were fine-tuned on UCF101 (5 epochs each, full-video uniform frame sampling via linspace — 16 frames for R(2+1)D, 8 frames for I3D). For VideoMAE, two variants were evaluated: (1) a third-party checkpoint already fine-tuned on UCF101 (nateraw/videomae-base-finetuned-ucf101), evaluated with its official segment-based sampling protocol, and (2) our own 5-epoch fine-tune of MCG-NJU/videomae-base, using the same full-video uniform sampling as R(2+1)D/I3D for consistency. All accuracy figures are top-1, single-clip, on the UCF101 test split (testlist01, 3783 videos).



Scores



|R2plus1D|I3D|VideoMAE (5 epoch finetune)|VideoMAE (nateraw)|
|-|-|-|-|
|Approximate Epoch time: 17 minutes|Approximate Epoch time: 7.2 minutes|Approximate Epoch time: 12 minutes|-|
|Accuracy: 82.82%|Accuracy: 86.73%|Accuracy: %48.72|Accuracy: 71.00%|
|Total Parameters:      31,351,938<br />Trainable Param:       31,351,938<br />Model Size:            119.60 MB<br />GFLOPs (fvcore):       40.64<br />Inference Time:        19.60 ms<br />Peak GPU Memory:       348.88 MB<br />Test Environment:      NVIDIA RTX 4060 Laptop GPU|Total Parameters:      27,430,821<br />Trainable Param:       27,430,821<br />Model Size:            104.64 MB<br />GFLOPs (fvcore):       28.54<br />Inference Time:        15.22 ms<br />Peak GPU Memory:       303.85 MB<br />Test Environment:      NVIDIA RTX 4060 Laptop GPU<br />|Total Parameters:      86,314,085<br />Trainable Param:       86,314,085<br />Model Size:            329.26 MB<br />GFLOPs (fvcore):       135.18<br />GFLOPs (paper):        180.00<br />Inference Time:        62.59 ms<br />Peak GPU Memory:       410.89 MB<br />Test Environment:      NVIDIA RTX 4060 Laptop GPU<br />|Total Parameters:      86,314,085<br />Trainable Param:       86,314,085<br />Model Size:            329.26 MB<br />GFLOPs (fvcore):       135.18<br />GFLOPs (paper):        180.00<br />Inference Time:        62.59 ms<br />Peak GPU Memory:       410.89 MB<br />Test Environment:      NVIDIA RTX 4060 Laptop GPU<br />|



Complexity Note (I3D): fvcore may undercount ops for some 3D-conv/pooling layers depending on operator support.

Complexity Note (VideoMAE): fvcore tends to undercount transformer ops (softmax, LayerNorm, GELU), so the fvcore figure likely understates real GFLOPs relative to the paper-reported 180.



For reference, the original VideoMAE paper reports \~91.3% top-1 / \~97.6% top-5 accuracy on UCF101 — well above both of our measurements.



Conclusion



VideoMAE is, on paper, the strongest of the three — its reported UCF101 accuracy (\~91.3%) is well above what we measured either way. The nateraw checkpoint (71.00%) is a third-party fine-tune with an undocumented training protocol, and its own model card notes it doesn't follow the paper's full evaluation setup (multi-segment inference), so it's likely underselling the architecture. Our own from-scratch fine-tune is more telling: train accuracy hit 95.93% after 5 epochs, but test accuracy was only 48.72% — a much larger train/test gap than R(2+1)D or I3D, pointing to overfitting rather than insufficient training time (5 epochs took about the same time as R(2+1)D, \~1 hour, so time wasn't really the constraint). Likely causes: no data augmentation was used for any model, and a large, low-inductive-bias transformer (86M params) is far more sensitive to that than a CNN; the classification head was randomly initialized with only 5 epochs to learn; and the shared learning rate (1e-4, kept constant across all three models for consistency) is on the high side for transformer fine-tuning. A more dedicated setup — real augmentation, a lower/warmup learning rate, and a schedule closer to the \~100 epochs typically used in the literature — would be needed to get a result actually comparable to the paper. VideoMAE should be treated as unresolved, not ruled out.



Among the models we could fairly compare (same data, hardware, and protocol), I3D is the clear recommendation: highest accuracy (86.73%) while also being the cheapest across every other metric — fewer parameters, lower GFLOPs, faster inference, less GPU memory than R(2+1)D. R(2+1)D is a solid competitor but doesn't beat I3D on any axis here. I3D's combination of accuracy and low latency also makes it the more practical pick for a real-time pipelines. VideoMAE is worth revisiting if more time/compute becomes available for a more dedicated and proper fine-tune.

