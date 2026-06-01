	
Leather Scratch Segmentation Handoff

What This Notebook Shows

This is a short handoff for the current zero-shot leather scratch segmentation pipeline.

The working architecture is:

Qwen3.5-VL (local MLX 4-bit) -> box proposal -> SAM 2.1 Hiera Large -> mask refinement -> overlay / comparison output

The objective is practical, not academic: produce a useful scratch mask overlay on local images with minimal dependencies and a pipeline that runs on macOS Apple Silicon.

Current Status

Primary target image 222 / data/input.jpg is working.
The new VLM-guided pipeline is materially better than the earlier SAM-only pass.
Best performance is on isolated or moderately localized scratches.
Hardest cases remain broad wear, stitched edges, and before/after composites where the problem is closer to damage-zone segmentation than thin-line scratch tracing.
Executive Summary

Recommended Takeaway

Use the current pipeline as a strong prototype / handoff baseline, not as a final production detector.

Why it works

Qwen gives a plausible defect region before segmentation.
SAM 2.1 turns that region into a refined mask.
The fallback path still recovers output when the VLM proposal is weak.
Where it still fails

Large worn regions can be over-segmented.
Hands / fingers in frame can bias segmentation if the proposed box is too broad.
Composite before/after images are harder than single-scene defect photos.
Manual Visual Audit

These are qualitative ratings, not benchmark metrics.

Image	Source Path	Quality	Notes
222 / input	Qwen -> SAM2	8.5/10	Good localization on subtle scratch cluster
Before-11	Qwen -> SAM2	8.5/10	Much better after bbox normalization fix
Before-8	Qwen -> SAM2	8/10	Tight mask on small localized defect
Before-10	Qwen -> SAM2	6.5/10	Region is plausible but still fragmented
Guide Before	Qwen -> SAM2	6.5/10	Partial scratch capture, some surrounding material
BeforeAfter3	Manual -> SAM2	6/10	Captures wear zone more than scratch lines
BeforeAfter2	Fallback -> SAM2	5/10	Broad wear region, weakest fit for current method
Setup in One Page

Runtime

OS: macOS on Apple Silicon
Segmentation model: facebook/sam2.1-hiera-large
Proposal model: local Qwen3.5-VL-9B-MLX-4bit
Acceleration:
Qwen runs through MLX
SAM 2.1 runs on MPS
Core Python packages

sam2
mlx
mlx-vlm
torch
transformers
opencv-python
numpy
huggingface_hub
Main project files

src/qwen_perception.py: VLM proposal layer
src/sam2_pipeline.py: box-prompted SAM 2.1 refinement
src/main.py: orchestration and output generation
outputs/: masks, overlays, comparisons, and perception JSON
Pipeline Schema

Input image
   -> crop to "before" panel when image is a before/after composite
   -> Qwen strict prompt (localized defect only)
       -> if empty: Qwen relaxed prompt (best plausible scratch guess)
   -> parse JSON bbox proposals
   -> normalize bbox coordinates
   -> SAM 2.1 box-prompted segmentation
   -> heuristic mask ranking + post-processing
   -> save:
      - binary mask
      - overlay
      - side-by-side comparison
      - perception JSON
Important implementation details

Multimodal chat template was required. Without apply_chat_template(..., num_images=1), Qwen image handling was unreliable in the local MLX path.

Thinking mode had to be disabled. In this setup, enable_thinking=False improved structured JSON output reliability.

bbox_2d was not always raw pixels. Some Qwen outputs behaved like coordinates in a 0..1000 space. Normalizing those boxes materially improved non-square image results.

Example VLM Output Contract

This is the proposal JSON for the primary 222 / input image.

{
  "model_path": "/Users/karimghandour/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-4bit",
  "raw_text": "```json\n[\n\t{\"bbox_2d\": [215, 585, 350, 665], \"label\": \"scratch\"}\n]\n```",
  "detections": [
    {
      "bbox_xyxy": [
        188,
        576,
        306,
        654
      ],
      "confidence": 0.5,
      "label": "scratch",
      "source_key": "bbox_2d"
    }
  ],
  "prompt_tokens": 962,
  "generation_tokens": 43,
  "prompt_tps": 77.67686241330932,
  "generation_tps": 18.099692903099303,
  "peak_memory_gb": 6.958557174
}
This is the proposal JSON for Before-11, one of the strongest secondary examples.

{
  "model_path": "/Users/karimghandour/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-4bit",
  "raw_text": "```json\n[\n\t{\"bbox_2d\": [375, 545, 534, 674], \"label\": \"scratch\"},\n\t{\"bbox_2d\": [355, 414, 515, 545], \"label\": \"scratch\"}\n]\n```",
  "detections": [
    {
      "bbox_xyxy": [
        384,
        314,
        547,
        388
      ],
      "confidence": 0.5,
      "label": "scratch",
      "source_key": "bbox_2d"
    },
    {
      "bbox_xyxy": [
        364,
        238,
        527,
        314
      ],
      "confidence": 0.5,
      "label": "scratch",
      "source_key": "bbox_2d"
    }
  ],
  "prompt_tokens": 702,
  "generation_tokens": 77,
  "prompt_tps": 121.96955730162635,
  "generation_tps": 29.869062471051347,
  "peak_memory_gb": 7.106078724
}
Best Showcase Cases

These are the strongest examples to show first because they best demonstrate the intended handoff behavior: proposal -> segmentation -> visual overlay.

1. Primary Target: 222 / data/input.jpg

No description has been provided for this image
Why it matters

This was the seed image used to tune the method.
The final mask is well localized relative to the visible scratch cluster.
It demonstrates that the VLM handoff can help on subtle, non-obvious defects.
2. Before-11-1024x576

No description has been provided for this image
Why it is strong

After the bbox normalization fix, the proposal moved off the finger region and onto the actual scratch line.
This is the clearest example of the VLM improving where SAM alone would be less targeted.
3. Before-8-1024x768

No description has been provided for this image
Why it is strong

Small, localized defect.
Good match between visible damage and final mask footprint.
Mid-Tier Results

Before-10-1024x768

No description has been provided for this image
Readout

The system identifies a plausible damage region.
Output is usable for triage, but not yet pixel-clean.
Leather_Scratch_Scuff_Guide_Before

No description has been provided for this image
Readout

Reasonable region proposal.
Captures only part of the visible scratch pattern.
Suggests the current box proposal is helpful but still under-informative for elongated irregular damage.
Hard Cases / Failure Modes

BeforeAfter2-v1_480x480

No description has been provided for this image
What happens

The current pipeline segments a large worn zone rather than a clean scratch trace.
This is a mismatch between model capability and image type.
BeforeAfter3-v1_480x480

No description has been provided for this image
What happens

The output is visually coherent, but it behaves more like wear area segmentation than scratch-line extraction.
Failure mode pattern

The weak cases all share one or more of the following:

before/after composites
broad abrasion instead of narrow scratch lines
curved upholstery geometry
seams / highlights / hands close to the defect
Why LM Studio Often Felt More Reliable Than Custom Local Calls

This repo reproduced the same issue people often hit in local multimodal environments:

the model file is present,
the image is supplied,
but the runtime formatting is slightly wrong,
so the model behaves inconsistently or returns unstructured text.
What mattered here

Qwen needed the multimodal chat template applied correctly.
The local MLX path needed enable_thinking=False to make JSON output more stable.
Once those were fixed, the perception layer became usable.
Practical interpretation

LM Studio hides a lot of multimodal request formatting. A custom local pipeline has to get that wiring right explicitly.

Useful references:

Qwen3-VL official repo
LM Studio image input docs
Recommended Next Steps

Best next engineering move

Keep the current Qwen -> SAM2 architecture, but improve the proposal contract.

High-value upgrades

Ask Qwen for points + box instead of box only. SAM is usually stronger when a box can be paired with one or more positive points.

Add a lightweight mask penalty for skin / finger-colored regions. This would directly reduce the finger-dominance failure mode.

Add a small defect-specific detector or fine-tuned proposal model. The current approach is zero-shot. A detector trained on leather defects would likely improve recall and reduce broad-zone proposals.

Split the problem by image type. Single-scene scratch photos and before/after composites should probably not share exactly the same prompting and ranking rules.

What I would test next

Qwen proposal with explicit point prompts
a defect-only binary classifier before segmentation
a scratch-vs-broad-wear routing step
detector fine-tuning if labeled masks or boxes become available
How To Re-Run

From the project root:

source venv/bin/activate
python src/main.py
Primary outputs:

outputs/mask.png
outputs/overlay.png
outputs/comparison_input.png
outputs/perception/input.json
Batch outputs for the other images are saved alongside them in outputs/.
