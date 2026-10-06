# Dataset quality analysis

The analyzed sample contains 1 videos, 720 decoded frames and 24.00 seconds of footage.

Decoder/probe diagnostics occurred in 0 clips; metadata or expected-format discrepancies occurred in 0 clips.

Across decoded frames, 0.00% have low mean luminance, 0.00% exceed the bright-pixel threshold, and 0.00% fall below the image-detail threshold. These are diagnostic candidates; they do not by themselves establish darkness, glare or blur defects.

0.00% of decoded frames repeat the preceding frame exactly. Duplicate search found 0 matching or similar clip pairs.

Visual analysis produced 11 dimension-level observations from sampled frames. Counts below describe model observations, not verified defect prevalence. Missing context and unexamined intervals remain unknown.

The model most often raised concerns about visual quality (1/1 analyzed clips); camera fov (1/1 analyzed clips); hand visibility (1/1 analyzed clips). Read the descriptions and evidence below to assess their significance.

## Visual observations and coverage

| Dimension | Clips analyzed | Clips with model concerns |
|---|---:|---:|
| visual quality | 1/1 | 1 |
| camera fov | 1/1 | 1 |
| hand visibility | 1/1 | 1 |
| object visibility | 1/1 | 1 |
| workspace visibility | 1/1 | 1 |
| instruction compliance | 0/1 | 0 |
| action completeness | 1/1 | 1 |
| task success | 0/1 | 0 |
| action correctness | 0/1 | 0 |
| interaction quality | 1/1 | 1 |
| failures | 1/1 | 0 |
| temporal quality | 1/1 | 1 |
| demonstration clarity | 1/1 | 0 |
| distractors | 1/1 | 0 |
| privacy safety | 0/1 | 0 |

## Per-clip findings

### epic-prep

24.000s · 30.000 FPS · 1280×720 · 720 frames

- visual_quality: Frames show moderate motion blur and slight exposure issues, with no significant glare, noise, or compression artifacts observed. Evidence: [[0.0, 0.7333], [0.7333, 7.2333], [2.9, 5.8]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: Camera positioning shifts significantly during the clip, with noticeable bumps and shifts, especially when the user opens the cabinet. Evidence: [[0.0, 0.7333], [0.7333, 1.4333], [1.4333, 7.2333]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Hands are consistently visible and well-tracked throughout the clip, with no cropping or occlusion issues observed. Evidence: [[0.0, 7.9667]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: Relevant objects such as vegetables, cutting board, and kitchen tools are visible, with no significant occlusion or cropping issues. Evidence: [[0.0, 7.9667]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
- workspace_visibility: Workspace is visible with sufficient context to understand the action of retrieving and placing a cutting board. Evidence: [[0.0, 0.7333], [0.7333, 1.4333], [1.4333, 2.1667], [2.1667, 2.9], [2.9, 3.6333], [3.6333, 4.3333], [4.3333, 5.0667], [5.0667, 5.8], [5.8, 6.5333], [6.5333, 7.2333], [7.2333, 7.9667]]. Subjective confidence: high.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The action begins with opening a cabinet, retrieving a cutting board, and placing it on the counter, with no indication of truncation within the observed window. Evidence: [[0.0, 0.7333], [0.7333, 1.4333], [1.4333, 2.1667], [2.1667, 2.9], [2.9, 3.6333], [3.6333, 4.3333], [4.3333, 5.0667], [5.0667, 5.8], [5.8, 6.5333], [6.5333, 7.2333], [7.2333, 7.9667]]. Subjective confidence: high.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object interaction is visible and understandable as the person retrieves and places the cutting board. Evidence: [[0.0, 0.7333], [0.7333, 1.4333], [1.4333, 2.1667], [2.1667, 2.9], [2.9, 3.6333], [3.6333, 4.3333], [4.3333, 5.0667], [5.0667, 5.8], [5.8, 6.5333], [6.5333, 7.2333], [7.2333, 7.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No failures, drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are observed in the provided frames. Evidence: [[0.0, 0.7333], [0.7333, 1.4333], [1.4333, 2.1667], [2.1667, 2.9], [2.9, 3.6333], [3.6333, 4.3333], [4.3333, 5.0667], [5.0667, 5.8], [5.8, 6.5333], [6.5333, 7.2333], [7.2333, 7.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: The clip shows smooth motion without noticeable cuts, time jumps, pauses, or abnormal speed. Evidence: [[0.0, 0.7333], [0.7333, 1.4333], [1.4333, 2.1667], [2.1667, 2.9], [2.9, 3.6333], [3.6333, 4.3333], [4.3333, 5.0667], [5.0667, 5.8], [5.8, 6.5333], [6.5333, 7.2333], [7.2333, 7.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
- demonstration_clarity: The action of retrieving a cutting board from a cabinet and placing it on the counter is unambiguous. Evidence: [[0.0, 0.7333], [0.7333, 1.4333], [1.4333, 2.1667], [2.1667, 2.9], [2.9, 3.6333], [3.6333, 4.3333], [4.3333, 5.0667], [5.0667, 5.8], [5.8, 6.5333], [6.5333, 7.2333], [7.2333, 7.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
- distractors: No other people, hands, clutter, screens, or mirrors are visible that cause ambiguity. Evidence: [[0.0, 0.7333], [0.7333, 1.4333], [1.4333, 2.1667], [2.1667, 2.9], [2.9, 3.6333], [3.6333, 4.3333], [4.3333, 5.0667], [5.0667, 5.8], [5.8, 6.5333], [6.5333, 7.2333], [7.2333, 7.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- visual_quality: Frames show moderate motion blur and some glare from lighting, with no significant compression artifacts or lens obstructions observed. Evidence: [[8.0, 8.7333], [9.4333, 10.1667], [10.9, 11.6333], [12.3333, 13.0667], [13.8, 14.5333], [15.2333, 15.9667]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: Camera positioning shifts significantly between frames, with noticeable bumps and shifts during movement, particularly when transitioning between areas. Evidence: [[8.7333, 9.4333], [9.4333, 10.1667], [10.9, 11.6333], [12.3333, 13.0667], [13.8, 14.5333], [15.2333, 15.9667]]. Subjective confidence: medium.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Hands are visible in multiple frames, with consistent tracking during manipulation, though some frames show partial occlusion or cropping. Evidence: [[8.7333, 9.4333], [10.9, 11.6333], [12.3333, 13.0667], [13.8, 14.5333], [15.2333, 15.9667]]. Subjective confidence: medium.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
- object_visibility: Relevant objects such as the sink, stove, drawer, and utensils are visible throughout, with no significant occlusion or cropping of key items. Evidence: [[8.0, 8.7333], [9.4333, 10.1667], [10.9, 11.6333], [12.3333, 13.0667], [13.8, 14.5333], [15.2333, 15.9667]]. Subjective confidence: medium.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: Workspace is visible but partially obscured by camera angle and motion blur, limiting full understanding of the action. Evidence: [[8.0, 8.7333], [9.4333, 10.1667], [10.9, 12.3333], [13.8, 15.2333]]. Subjective confidence: medium.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: Action begins with hand reaching toward sink, continues with drawer opening and knife selection, but ending is not visible in provided frames. Evidence: [[8.0, 15.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object interaction visible as hand reaches for sink, opens drawer, and selects knife, but motion blur and camera movement reduce clarity. Evidence: [[8.7333, 9.4333], [10.9, 15.2333]]. Subjective confidence: medium.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance observed in the provided frames. Evidence: [[8.0, 15.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: The clip contains motion blur and rapid camera movement, suggesting potential time jumps or abnormal speed, particularly between frames 9.4333s and 10.1667s. Evidence: [[8.7333, 9.4333], [10.1667, 10.9]]. Subjective confidence: medium.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
- demonstration_clarity: The action of retrieving a knife from a drawer is unambiguous, though the camera movement obscures some details. Evidence: [[12.3333, 13.8]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
- distractors: No other people, hands, clutter, screens, or mirrors are visible that cause ambiguity. Evidence: [[8.0, 15.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- visual_quality: Frames show moderate motion blur and some glare from the sink faucet, with generally adequate exposure and minimal noise or compression artifacts. Evidence: [[16.0, 16.7333], [18.9, 23.9667]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: Camera maintains a consistent top-down view of the kitchen counter and sink area, with minor bumps and shifts during hand movements. Evidence: [[18.9, 19.6333]]. Subjective confidence: medium.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Hands are consistently visible and well-tracked, with no cropping or occlusion issues, though motion blur affects clarity during rapid movements. Evidence: [[16.7333, 18.9], [21.0667, 23.9667]]. Subjective confidence: medium.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: Relevant objects (knife, cutting board, vegetables, sink) are consistently visible and not occluded, though some motion blur affects detail during rapid movements. Evidence: [[16.7333, 23.9667]]. Subjective confidence: medium.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: Sufficient workspace visible including countertop, sink, and cutting board, allowing understanding of the action. Evidence: [[16.0, 23.9667]]. Subjective confidence: high.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: Action begins with placing knife on counter, then picking up and washing a cucumber; ending not visible in clip. Evidence: [[16.0, 23.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object interactions visible: placing knife, picking up cucumber, and washing it under running water. Evidence: [[16.0, 23.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance observed. Evidence: [[16.0, 23.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses, or abnormal speed observed in the provided frames. Evidence: [[16.0, 16.0], [16.7333, 16.7333], [17.4333, 17.4333], [18.1667, 18.1667], [18.9, 18.9], [19.6333, 19.6333], [20.3333, 20.3333], [21.0667, 21.0667], [21.8, 21.8], [22.5333, 22.5333], [23.2333, 23.2333], [23.9667, 23.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action of washing a cucumber and the target object are unambiguous in all frames. Evidence: [[16.0, 16.0], [16.7333, 16.7333], [17.4333, 17.4333], [18.1667, 18.1667], [18.9, 18.9], [19.6333, 19.6333], [20.3333, 20.3333], [21.0667, 21.0667], [21.8, 21.8], [22.5333, 22.5333], [23.2333, 23.2333], [23.9667, 23.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
- distractors: No other people, hands, clutter, screens, or mirrors are causing ambiguity in the provided frames. Evidence: [[16.0, 16.0], [16.7333, 16.7333], [17.4333, 17.4333], [18.1667, 18.1667], [18.9, 18.9], [19.6333, 19.6333], [20.3333, 20.3333], [21.0667, 21.0667], [21.8, 21.8], [22.5333, 22.5333], [23.2333, 23.2333], [23.9667, 23.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.
- Visual windows: 3; windows with unavailable analysis: 0. Window coverage is sampled analysis, not continuous event detection.

## Interpretation limits

- No acceptance labels or composite quality score; these are descriptive measurements.
- This report describes the analyzed files, not unseen source footage or the full HOT3D dataset.
- Image thresholds are uncalibrated diagnostic heuristics; texture, fisheye borders and intended lighting affect them.
- Visual observations are model judgments on sampled stills, not verified event labels; subjective confidence is uncalibrated.
- Annotation coverage and modeled visibility do not prove that hands are visibly trackable.
- Missing frames cannot always be detected after timestamps are rewritten. Speed needs a capture reference.
- Near-duplicate search compares aligned full clips; partial overlaps, crops and speed changes may be missed.
- Semantic diversity needs supplied labels; monocular monochrome video cannot establish color diversity.

## Dataset diversity


### background

Supplied labels for 0 clips: {}

- epic-prep: wooden countertop, stainless steel sink, white walls, kitchen cabinets, dish rack, various kitchen items (sampled model description)

### lighting

Supplied labels for 0 clips: {}

- epic-prep: even, moderate brightness (sampled model description)

### object instances

Supplied labels for 0 clips: {}

- epic-prep: person's hands; green cutting board; knife with gray handle; zucchini; carrot; stainless steel sink; dish rack; bottles and sponges on counter; cabinet drawer with utensils; stove (sampled model description)

### object poses

Supplied labels for 0 clips: {}

- epic-prep: hands reaching for cabinet; hands placing cutting board on counter; hand turning on faucet; hands washing zucchini under running water; hands holding knife and zucchini over sink (sampled model description)

### viewpoint

Supplied labels for 0 clips: {}

- epic-prep: overhead, slightly angled (sampled model description)

### starting state

Supplied labels for 0 clips: {}

- epic-prep: hands reaching toward countertop near sink (sampled model description)

### execution trajectory

Supplied labels for 0 clips: {}

- epic-prep: retrieves cutting board, washes zucchini, retrieves knife, places knife on counter (sampled model description)
