# Dataset quality analysis

The analyzed sample contains 11 videos, 1,650 decoded frames and 55.00 seconds of footage.

Decoder/probe diagnostics occurred in 0 clips; metadata or expected-format discrepancies occurred in 0 clips.

Across decoded frames, 0.00% have low mean luminance, 0.00% exceed the bright-pixel threshold, and 0.00% fall below the image-detail threshold. These are diagnostic candidates; they do not by themselves establish darkness, glare or blur defects.

0.00% of decoded frames repeat the preceding frame exactly. Duplicate search found 0 matching or similar clip pairs.

Visual analysis produced 121 dimension-level observations from sampled frames. Counts below describe model observations, not verified defect prevalence. Missing context and unexamined intervals remain unknown.

The model most often raised concerns about visual quality (9/11 analyzed clips); action completeness (8/11 analyzed clips); workspace visibility (3/11 analyzed clips). Read the descriptions and evidence below to assess their significance.

Source metadata identifies 5 participants and 8 sequences. This alone does not establish task, object or environment diversity.

These HOT3D inputs are short clips. An action continuing beyond a clip boundary does not establish that the original demonstration was recorded incompletely.

## Visual observations and coverage

| Dimension | Clips analyzed | Clips with model concerns |
|---|---:|---:|
| visual quality | 11/11 | 9 |
| camera fov | 11/11 | 2 |
| hand visibility | 11/11 | 0 |
| object visibility | 11/11 | 0 |
| workspace visibility | 11/11 | 3 |
| instruction compliance | 0/11 | 0 |
| action completeness | 11/11 | 8 |
| task success | 0/11 | 0 |
| action correctness | 0/11 | 0 |
| interaction quality | 11/11 | 1 |
| failures | 11/11 | 0 |
| temporal quality | 11/11 | 0 |
| demonstration clarity | 11/11 | 0 |
| distractors | 11/11 | 0 |
| privacy safety | 0/11 | 0 |

## Per-clip findings

### clip-000000

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are black and white, showing some motion blur and compression artifacts, with no significant glare or lens obstruction visible. Evidence: [[0.0, 5.0]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The camera maintains a consistent fisheye perspective with minimal bumps or shifts, though slight tilting occurs during hand movements. Evidence: [[0.0, 5.0]]. Subjective confidence: medium.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Hands are consistently visible and well-tracked throughout the clip, with no cropping or occlusion issues. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: Relevant objects (spoon, bottles, glasses, remote) remain visible and unoccluded throughout the clip. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: The workspace is clearly visible, showing a table with various objects and surrounding equipment, providing sufficient context to understand the action. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The action begins with hands on the table, proceeds to picking up and manipulating a spatula, and ends with the spatula being placed back on the table, indicating a complete task within the clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- interaction_quality: Hand-object contact is clearly visible as the person picks up, holds, and manipulates the spatula, with interactions being understandable and well-captured. Evidence: [[0.9, 4.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are observed in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses or abnormal speed observed in the 5-second clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action of picking up and examining a wooden spoon is unambiguous and clearly visible. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
- distractors: No other people, hands, clutter, screens or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-000001

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are black and white, showing some motion blur and compression artifacts, with no significant glare or lens obstruction visible. Evidence: [[0.0, 1.0]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The camera maintains a consistent fisheye perspective with minimal bumps or shifts, capturing the entire scene from a top-down view. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Both hands are consistently visible throughout the clip, with no cropping or occlusion, and tracking remains consistent. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: Relevant objects including a remote, bottle, and spoon are visible throughout, with no significant occlusion or cropping. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: Sufficient workspace is visible to understand the action, including a table with various objects and surrounding environment. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The clip shows the beginning of an action (picking up a remote) and manipulation (placing it down, picking up a bottle, and handling it), but the ending or completion of the task is not visible within the 5-second clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object interactions are visible and understandable, including placing the remote on the table, picking up a bottle, and handling it with both hands. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are observed in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses or abnormal speed observed in the 5-second clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action of placing a remote on the table and handling a bottle of ranch dressing is unambiguous. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
- distractors: No other people, hands, clutter, screens or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-000006

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are black and white, showing no motion blur or significant noise, but exhibit fisheye lens distortion and some glare from overhead lights. Evidence: [[0.0, 1.0]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The camera maintains a consistent fisheye perspective with minimal bumps or shifts, capturing the user's hands and desk setup from a top-down view. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Both hands are consistently visible throughout the clip, with no cropping or occlusion, and tracking remains consistent despite hand movements. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: Relevant objects including the keyboard, mouse, mug, and Rubik's cube are consistently visible and not occluded throughout the clip. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: The workspace is sufficiently visible, showing a desk with a keyboard, mouse, mug, and other items, allowing understanding of the action. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The clip shows the beginning (0.0s) and manipulation (0.4667s-4.9667s) of the task, but the ending is not fully captured as the action appears to be ongoing at the end of the clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object interactions are visible and understandable, including picking up and placing down the keyboard and using the mouse. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are observed in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses, or abnormal speed observed in the 5-second clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action of typing and using a mouse is unambiguous, with clear visibility of the keyboard and mouse as target objects. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
- distractors: No other people, hands, clutter, screens, or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-000007

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are black and white, showing some motion blur on hands and keyboard due to movement, with no significant glare, noise, or lens obstruction visible. Evidence: [[0.0, 5.0]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The camera has a wide-angle fisheye lens, showing a top-down view of the desk and surrounding area with minimal bumps or shifts in positioning. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Both hands are consistently visible on the keyboard and mouse, with no cropping or occlusion, and tracking appears consistent across frames. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: Relevant objects including the keyboard, mouse, mug, Rubik's cube, and coffee pot are visible and not occluded or cropped out in any frame. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
- workspace_visibility: The workspace is clearly visible, showing a desk with a keyboard, mouse, mug, and other items, providing sufficient context to understand the action. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The clip shows the beginning (typing on keyboard) and manipulation (switching to mouse), but the ending is not fully captured as the action appears to be ongoing at the end of the clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object interactions are clearly visible, with hands actively typing on the keyboard and using the mouse, demonstrating understandable manipulation. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are observed in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses, or abnormal speed observed in the 5-second clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action of typing and using a mouse is unambiguous, with clear visibility of hands and target objects. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
- distractors: No other people, hands, clutter, screens, or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-000008

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are in black and white, showing moderate motion blur from hand movements, with no visible glare, noise, or lens obstruction, and adequate exposure despite the monochrome format. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The fisheye lens creates a wide, curved field of view with consistent framing throughout the clip, showing no noticeable bumps, shifts, or repositioning of the camera. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- hand_visibility: Both hands are consistently visible and well-tracked throughout the clip, with no cropping or occlusion, and clear visibility of finger movements on the keyboard and mouse. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: Relevant objects including the keyboard, mouse, mug, Rubik's cube, and phone are consistently visible and not occluded or cropped out in any frame. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: The workspace is clearly visible, showing a desk with a keyboard, mouse, mug, Rubik's cube, and other items, providing sufficient context to understand the action. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The clip shows the beginning (typing and mouse use) and manipulation (typing and mouse movements) of a computer task, but the ending is not visible, and there is no indication of recording truncation. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object contact is visible and understandable, with hands interacting with the keyboard and mouse in a natural manner. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are observed in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses, or abnormal speed are visible in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action of typing and using a mouse is unambiguous, with clear visibility of the hands, keyboard, and mouse. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
- distractors: No other people, hands, clutter, screens, or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-000100

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are black and white, showing some motion blur and compression artifacts, with no significant glare or lens obstruction visible. Evidence: [[0.0, 1.0]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The fisheye lens creates significant barrel distortion, and the camera remains relatively stable with minimal bumps or shifts. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Both hands are consistently visible and well-tracked throughout the clip, with no cropping or occlusion issues. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: The orange juice carton is clearly visible and readable throughout the clip, with no occlusion or cropping. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: Sufficient workspace visible to understand the action, including a table, chair, and surrounding equipment. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The action begins with picking up a juice carton, continues with manipulation (turning it), and appears to be ongoing at the end of the clip, suggesting possible recording truncation. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object contact is visible and understandable throughout the clip, with clear manipulation of the juice carton. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance observed. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses or abnormal speed observed in the 5-second clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action of picking up and examining an orange juice carton is unambiguous and clearly visible. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
- distractors: No other people, hands, clutter, screens or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-000200

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are black and white, showing some motion blur and compression artifacts, with no significant glare or lens obstruction visible. Evidence: [[0.0, 5.0]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The camera has a wide-angle fisheye lens, showing a circular field of view with some distortion at the edges, and there are no noticeable bumps or shifts in positioning. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: The hand holding the masher is consistently visible throughout the clip, with no cropping or occlusion, and tracking appears consistent. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: The wooden bowl and masher are clearly visible throughout the clip, with no significant occlusion or cropping of relevant objects. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: Sufficient workspace is visible, including a table with a bowl and masher, and surrounding equipment, though the fisheye lens distorts the view. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The action begins with the masher being positioned over the bowl and ends with it being lifted away, suggesting a complete mashing motion within the clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object contact is visible and understandable, with the hand holding the masher and pressing it into the bowl's contents. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are visible in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: The clip shows continuous, smooth motion without noticeable cuts, time jumps, pauses, or abnormal speed. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
- demonstration_clarity: The action of mashing food in a bowl with a masher is unambiguous and clearly visible throughout the clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- distractors: No other people, hands, clutter, screens, or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-000500

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are black and white, showing some motion blur and compression artifacts, with no significant glare or lens obstruction visible. Evidence: [[0.0, 5.0]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The fisheye lens creates a wide-angle, distorted view with consistent positioning, showing minimal bumps or shifts. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Hands are consistently visible and well-tracked as they manipulate the bowl, with no cropping or occlusion issues. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: The wooden bowl and its contents are clearly visible throughout, with no significant occlusion or cropping. Evidence: [[0.0, 5.0]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: Workspace is visible with a table, utensils, and bowls, but the fisheye lens distorts the view and limits peripheral context. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: Action begins with reaching for a bowl, manipulation involves holding and rotating it, and the clip ends mid-manipulation without showing completion or truncation. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object contact is visible and clear as the person grasps and rotates the bowl, with no failed grasps or interruptions observed. Evidence: [[1.8, 4.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are observed in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: The clip shows smooth, continuous motion without noticeable cuts, time jumps, pauses, or abnormal speed. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
- demonstration_clarity: The action of picking up and holding a wooden bowl is unambiguous, with the target object clearly visible throughout. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- distractors: No other people, hands, clutter, screens, or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-000700

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are black and white, showing some motion blur and compression artifacts, with no significant glare or lens obstruction visible. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The camera has a wide-angle fisheye lens, showing a circular field of view with some distortion at the edges, and there are no noticeable bumps or shifts in positioning. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Both hands are consistently visible throughout the clip, with no cropping or occlusion, and tracking appears consistent. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: Relevant objects including a bowl, candy, and a TV are visible throughout the clip, with no significant occlusion or cropping. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: Workspace is visible with a table, bowl, and items, but the fisheye lens distorts the view and limits peripheral context. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: Action appears to be a pointing gesture followed by a hand movement toward objects on the table, but the clip ends before the action is fully completed. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hands are visible and appear to interact with objects on the table, but the interaction is not clearly defined due to the lack of object manipulation. Evidence: [[0.0, 4.9667]]. Subjective confidence: low.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are visible in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses, or abnormal speed observed in the 5-second clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action (pointing at a screen) and target object (screen) are unambiguous throughout the clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- distractors: No other people, hands, clutter, screens, or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-001000

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are in black and white with visible motion blur and some glare from overhead lights, but no significant compression artifacts or lens obstruction. Evidence: [[0.0, 1.0]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The fisheye lens creates a wide, curved field of view with consistent positioning, showing minimal bumps or shifts during the clip. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
- hand_visibility: Hands are consistently visible and well-tracked throughout the clip, with no cropping or occlusion issues. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: Relevant objects (milk carton, cup, table) are clearly visible and not occluded or cropped out in any frame. Evidence: [[0.0, 1.0]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
- workspace_visibility: The workspace is visible with a table and various objects, but the fisheye lens distorts the view and limits the extent of visible area. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The action begins with handling a container, progresses to opening a milk carton, and ends with pouring milk into a cup, but the clip may be truncated as the pouring action is not fully completed. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object interactions are visible and understandable, including opening a carton and pouring milk, with clear hand contact and manipulation. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are observed in the provided frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses, or abnormal speed observed within the 5-second clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action of pouring milk from a carton into a cup is unambiguous and clearly visible throughout the clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- distractors: No other people, hands, clutter, screens, or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

### clip-001100

5.000s · 30.000 FPS · 768×960 · 150 frames

- visual_quality: The frames are black and white, showing no motion blur or significant noise, but exhibit fisheye distortion and some glare from overhead lighting. Evidence: [[0.0, 4.9667]]. Subjective confidence: medium.
  Limitation: Visual artifact judgments are unverified; image-detail and luminance measurements are separate evidence.
- camera_fov: The camera maintains a consistent fisheye perspective throughout, with no noticeable bumps or shifts in positioning. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sampled frames cannot establish absence of intervening bumps or shifts.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- hand_visibility: Both hands are consistently visible and well-tracked throughout the clip, with no cropping or occlusion issues. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: No image-based hand tracking accuracy was measured; visibility at sample times does not prove continuous trackability.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- object_visibility: The keyboard, mouse, and other desk objects remain clearly visible and unoccluded throughout the clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Object naming and occlusion are model judgments at sampled times, not verified object identities.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- workspace_visibility: The workspace is clearly visible, showing a desk with a keyboard, mouse, mug, and other items, with sufficient area to understand the action. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Sufficiency of workspace context is a model judgment.
- action_completeness: The clip shows the beginning (hands on keyboard/mouse), manipulation (typing, mouse use), and ending (hands on keyboard at end), with no clear truncation of the task. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: The original task boundaries are unknown; visible first/last states do not prove the full task was recorded.
- interaction_quality: Hand-object interactions are visible and understandable, including typing on the keyboard and using the mouse. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Contact is visually inferred; no physical contact labels were measured.
- failures: No drops, spills, collisions, failed grasps, resets, interruptions, or outside assistance are observed. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Brief failures or outside assistance can occur between sampled frames.
- temporal_quality: No cuts, time jumps, pauses or abnormal speed observed in the 5-second clip. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Normal speed, continuity and absence of cuts cannot be established from these sampled stills.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- demonstration_clarity: The action (typing and using a mouse) and target objects (keyboard, mouse) are consistently unambiguous across all frames. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Action and object descriptions have not been independently verified.
  Wording caveat: Model wording implies continuity/completeness that sampled images cannot verify.
- distractors: No other people, hands, clutter, screens or mirrors are visible that cause ambiguity. Evidence: [[0.0, 4.9667]]. Subjective confidence: high.
  Limitation: Ambiguity is a subjective model judgment.
- instruction_compliance: No task supplied; cannot judge this requirement.
- task_success: No final_state supplied; cannot judge this requirement.
- action_correctness: No required_steps supplied; cannot judge this requirement.
- privacy_safety: No collection_rules supplied; cannot judge this requirement.

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

- clip-000000: indoor room with table, chair, and equipment (sampled model description)
- clip-000001: indoor workspace with desk, chair, and equipment (sampled model description)
- clip-000006: indoor workspace with visible cables, equipment, and wall-mounted fixtures (sampled model description)
- clip-000007: indoor desk setup with cables, wall-mounted equipment, and a chair (sampled model description)
- clip-000008: indoor workspace with equipment and cables (sampled model description)
- clip-000100: indoor room with furniture and equipment (sampled model description)
- clip-000200: indoor setting with overhead lighting rig and wooden table (sampled model description)
- clip-000500: indoor room with table, chairs, and overhead lighting fixtures (sampled model description)
- clip-000700: indoor room with equipment, wall-mounted monitor, framed picture, and guitar (sampled model description)
- clip-001000: indoor lab with overhead equipment and cables (sampled model description)
- clip-001100: dark curtains, framed photos on walls, studio lighting equipment (sampled model description)

### lighting

Supplied labels for 0 clips: {}

- clip-000000: even, moderate brightness (sampled model description)
- clip-000001: even, moderate brightness (sampled model description)
- clip-000006: even, moderate brightness with no strong shadows (sampled model description)
- clip-000007: even, moderate brightness with no strong shadows (sampled model description)
- clip-000008: even, moderate brightness (sampled model description)
- clip-000100: even, moderate brightness with high contrast (sampled model description)
- clip-000200: bright, high contrast with strong highlights and shadows (sampled model description)
- clip-000500: even, moderate brightness with high contrast (sampled model description)
- clip-000700: bright, high-contrast from overhead lights (sampled model description)
- clip-001000: even, moderate brightness with high contrast (sampled model description)
- clip-001100: bright, high contrast from overhead light source (sampled model description)

### object instances

Supplied labels for 0 clips: {}

- clip-000000: person; table; wooden spoon; glasses; bottle; remote control; chair (sampled model description)
- clip-000001: hands; remote control; pitcher; bottle; spoon (sampled model description)
- clip-000006: person's arms and hands; keyboard; computer mouse; white mug; rubik's cube; coffee pot; phone; desk surface; chair (sampled model description)
- clip-000007: hands; keyboard; computer mouse; cup; Rubik's cube; coffee pot; phone; desk; chair (sampled model description)
- clip-000008: hands, keyboard, mouse, mug, Rubik's cube, phone, coffee pot (sampled model description)
- clip-000100: person; orange juice carton; dumbbell; chair; table; rubik's cube; marker; star-shaped object (sampled model description)
- clip-000200: hand holding masher, wooden bowl, small objects in bowl, table surface (sampled model description)
- clip-000500: person wearing dark ribbed sweater; wooden bowl; white plate with lid; wooden spatula; small objects inside bowl (sampled model description)
- clip-000700: person wearing striped shirt; watch on left wrist; wooden bowl with two round objects; two small rectangular packages on table; guitar; monitor; framed picture; ceiling-mounted lights and equipment (sampled model description)
- clip-001000: person's hands; milk carton; cup; table; ceiling-mounted lights; wires and equipment (sampled model description)
- clip-001100: keyboard, mouse, mug, phone, pen, paper, t-shirt with graphic (sampled model description)

### object poses

Supplied labels for 0 clips: {}

- clip-000000: person standing, hands on table; person holding wooden spoon; person examining wooden spoon; person placing wooden spoon back on table (sampled model description)
- clip-000001: hands holding remote; remote placed on table; bottle picked up; spoon picked up (sampled model description)
- clip-000006: hands initially on desk, right hand near keyboard; right hand lifts keyboard; hands hold keyboard; hands place keyboard back on desk; right hand moves to mouse; right hand uses mouse; left hand on desk, right hand on mouse (sampled model description)
- clip-000007: hands typing on keyboard; right hand moving to mouse; left hand on keyboard; right hand using mouse (sampled model description)
- clip-000008: hands typing and using mouse, keyboard stationary, mug and cube static (sampled model description)
- clip-000100: person standing, reaching for carton; person holding carton with both hands; person rotating carton to view label (sampled model description)
- clip-000200: hand moves masher from above bowl to press down, then lifts slightly (sampled model description)
- clip-000500: person standing, hands initially at sides; person reaches for bowl with both hands; person holds bowl with both hands, tilting it slightly (sampled model description)
- clip-000700: person's arms extended forward, hands near monitor; left hand points at monitor, right hand near monitor; both hands point at monitor; left hand moves to table, right hand remains near monitor; left hand reaches for bowl, right hand moves to table (sampled model description)
- clip-001000: hands holding cup and milk carton; milk carton being opened; milk being poured into cup; cup held steady during pouring (sampled model description)
- clip-001100: hands moving across keyboard and mouse, arms extended, body seated (sampled model description)

### viewpoint

Supplied labels for 0 clips: {}

- clip-000000: top-down fisheye view (sampled model description)
- clip-000001: top-down fisheye (sampled model description)
- clip-000006: overhead fisheye perspective from above the user (sampled model description)
- clip-000007: overhead fisheye view from above the user (sampled model description)
- clip-000008: overhead fisheye view (sampled model description)
- clip-000100: top-down fisheye perspective (sampled model description)
- clip-000200: overhead fisheye perspective from above the user (sampled model description)
- clip-000500: top-down fisheye perspective (sampled model description)
- clip-000700: first-person, fisheye lens, looking down at table and arms (sampled model description)
- clip-001000: top-down fisheye view (sampled model description)
- clip-001100: overhead fisheye view from above the user (sampled model description)

### starting state

Supplied labels for 0 clips: {}

- clip-000000: person standing with hands on table, objects on table (sampled model description)
- clip-000001: hands holding remote control (sampled model description)
- clip-000006: hands resting on desk, right hand near keyboard, left hand on desk (sampled model description)
- clip-000007: hands positioned on keyboard, right hand near mouse (sampled model description)
- clip-000008: hands on keyboard and mouse, typing and moving mouse (sampled model description)
- clip-000100: person reaching for orange juice carton on table (sampled model description)
- clip-000200: hand holding masher above bowl, bowl contains small objects (sampled model description)
- clip-000500: person standing, hands at sides, bowl on table (sampled model description)
- clip-000700: person seated, arms extended forward, hands near monitor (sampled model description)
- clip-001000: person holding cup and milk carton, preparing to pour (sampled model description)
- clip-001100: hands on keyboard, right hand near mouse (sampled model description)

### execution trajectory

Supplied labels for 0 clips: {}

- clip-000000: person picks up wooden spoon, examines it, and places it back on table (sampled model description)
- clip-000001: placing remote, picking up bottle and spoon, examining bottle (sampled model description)
- clip-000006: person lifts keyboard, places it back, then uses mouse (sampled model description)
- clip-000007: hands type on keyboard, then right hand moves to mouse and clicks (sampled model description)
- clip-000008: hands alternate between keyboard and mouse, slight movements (sampled model description)
- clip-000100: person picks up carton, holds it, and rotates it to view label (sampled model description)
- clip-000200: masher is pressed down into bowl, then lifted slightly, repeated (sampled model description)
- clip-000500: person picks up bowl, lifts and tilts it slightly while holding (sampled model description)
- clip-000700: person points at monitor, then moves left hand to bowl, right hand to table (sampled model description)
- clip-001000: person pours milk from carton into cup (sampled model description)
- clip-001100: hands alternate between keyboard and mouse, then both on keyboard (sampled model description)
