# Annotation audit

{
  "requested_spans": 26,
  "captions": 26,
  "missing": 0,
  "annotations_with_findings": 23,
  "verified_annotations": 26,
  "finding_counts": {
    "annotation_rule": 14,
    "visual_contradicted": 31,
    "visual_unknown": 19,
    "uncertain_annotation": 1
  }
}

Original annotations were preserved. Model judgments are unverified second opinions.

## epic-prep#000.000

[0.0, 1.312]: Reach for the cabinet with the right hand

- annotation_rule: A2: 8 words below 10
- visual_contradicted: action: The hand is not reaching for the cabinet; it is pulling a drawer open and then reaching into the cabinet space.
- visual_unknown: details: No specific details about the cabinet (color, brand, material, location) are provided in the frames.

## epic-prep#001.312

[1.312, 3.062]: Reach into the cabinet with the left hand

- annotation_rule: A2: 8 words below 10
- visual_contradicted: action: The left hand is not reaching into the cabinet; it is pulling out a cloth and later moving away. The right hand is the one interacting with items inside the cabinet.
- visual_contradicted: hand: The left hand is not reaching into the cabinet; it is pulling out a cloth and later moving away. The right hand is the one interacting with items inside the cabinet.
- visual_unknown: details: No specific details about the cabinet's color, brand, material, or location are provided in the frames.

## epic-prep#003.062

[3.062, 4.938]: Grasp the red container under the cabinet with both hands

- visual_contradicted: object: The object being grasped is dark and appears to be a cooking pot or pan, not a red container.
- visual_contradicted: details: The object is not red; it is dark-colored, likely black or dark gray, and appears to be a cooking pot or pan.

## epic-prep#004.938

[4.938, 6.938]: Place the green cutting board on the wooden countertop

- annotation_rule: A2: 9 words below 10

## epic-prep#006.938

[6.938, 9.438]: Reach for the sink with the right hand

- annotation_rule: A2: 8 words below 10
- visual_contradicted: action: The hand is not reaching for the sink; it is moving away from it and then out of frame.
- visual_contradicted: hand: The hand is not reaching for the sink; it is moving away from it and then out of frame.
- visual_unknown: details: No specific details about color, brand, material, location, contact or success claims can be established from the provided frames.

## epic-prep#009.438

[9.438, 10.938]: Reach for the blue bowl under the wooden counter

- annotation_rule: A2: 9 words below 10
- visual_contradicted: action: The candidate claims 'reach' for a 'bowl', but the hand is reaching for a blue object under the counter, not a bowl, and the object is not clearly identifiable as a bowl.
- visual_contradicted: object: The blue object under the counter is not clearly identifiable as a bowl; it appears to be a container or lid, not a bowl.
- visual_contradicted: details: The object is blue, but not clearly a bowl; the candidate's description of the object as a 'blue bowl' is contradicted by the visual evidence.

## epic-prep#010.938

[10.938, 12.438]: Grasp the drawer handle with the right hand to pull it open

- annotation_rule: A3: 3 verb cores in text: grasp+open+pull
- visual_contradicted: action: The hand is not grasping the drawer handle; it is pulling the drawer open with fingers on the handle, and the action is more accurately 'pulling' than 'grasping'.
- visual_contradicted: hand: The hand is the right hand, but the action is not 'grasping' the handle; it is pulling it open.
- visual_unknown: details: No specific details about color, brand, material, or location beyond the drawer handle are provided or visible.

## epic-prep#012.438

[12.438, 14.438]: Grasp the knife from the drawer with the right hand

- visual_contradicted: action: The person is not grasping the knife from the drawer; they are pulling it out after already having it in hand.
- visual_unknown: details: No specific details about the knife's color, brand, or material are visible or verifiable from the frames.

## epic-prep#014.438

[14.438, 16.688]: Grasp the knife from the drawer with the right hand

- visual_unknown: details: The color, brand, material, or specific location details of the knife are not discernible from the provided frames.

## epic-prep#016.688

[16.688, 18.438]: Place the knife on the wooden countertop near the sink

- visual_unknown: details: No specific details about the knife's color, brand, material, or location beyond being placed near the sink can be confirmed from the frames.

## epic-prep#018.438

[18.438, 20.312]: Grasp the zucchini with the left hand

- annotation_rule: A2: 7 words below 10
- visual_contradicted: action: The zucchini is grasped with the left hand in frames 18.7000-19.3667, but the candidate claims the action is 'grasp' which is correct, however the hand is not the left hand as stated in the candidate annotation. The candidate annotation incorrectly specifies 'LEFT' hand, while the frames show the left hand grasping the zucchini.
- visual_contradicted: hand: The left hand is grasping the zucchini in frames 18.7000-19.3667, contradicting the candidate's claim of 'LEFT' hand. The candidate annotation incorrectly specifies 'LEFT' hand, while the frames show the left hand grasping the zucchini.
- visual_unknown: details: No specific details such as color, brand, material, location, or contact claims are provided in the candidate annotation that can be verified or contradicted from the frames.

## epic-prep#020.312

[20.312, 21.688]: Rinse the cucumber under running water with both hands

- annotation_rule: A2: 9 words below 10
- visual_unknown: details: No specific details about color, brand, material, location, or contact/success claims are visible or verifiable from the frames.

## epic-prep#021.688

[21.688, 24.0]: Rinse the green cucumber under running water with both hands

- visual_unknown: details: No specific details about color, brand, material, location, or contact claims are visible or verifiable from the frames.

## epic-cook#000.000

[0.0, 1.562]: Reach into the cabinet with the right hand

- annotation_rule: A2: 8 words below 10
- visual_contradicted: action: The hand is not reaching into the cabinet; it is hovering near the edge or outside the cabinet opening.
- visual_unknown: details: No specific details about color, brand, material, or location beyond the cabinet being white and open are provided or verifiable from the frames.

## epic-cook#001.562

[1.562, 2.938]: Reach into the cabinet with the right hand

- annotation_rule: A2: 8 words below 10
- visual_contradicted: action: The right hand is not reaching into the cabinet; it is holding or manipulating an object outside the cabinet opening.
- visual_contradicted: hand: The right hand is not reaching into the cabinet; it is holding or manipulating an object outside the cabinet opening.
- visual_unknown: details: No specific details about color, brand, material, or location are visible or verifiable from the frames.

## epic-cook#002.938

[2.938, 4.688]: Grasp the metal grater with the right hand

- annotation_rule: A2: 8 words below 10
- visual_contradicted: action: The hand is not grasping the grater; it is moving away from it after retrieving it from the cabinet.
- visual_unknown: object: The object being handled is not clearly identifiable as a metal grater in the provided frames.
- visual_contradicted: hand: The hand is not grasping the object; it is moving away from it after retrieving it from the cabinet.
- visual_contradicted: visibility: The object is not fully visible in the frames; it is partially obscured or out of frame.
- visual_contradicted: details: The object is not clearly identifiable as a metal grater in the provided frames.

## epic-cook#004.688

[4.688, 6.062]: Reach into the cabinet with both hands to retrieve an object

- uncertain_annotation: Uncertainty is true, missing or malformed
- visual_unknown: object: The specific object being retrieved is not clearly identifiable from the frames.
- visual_unknown: details: No specific details about the object, color, brand, or material can be determined from the provided frames.

## epic-cook#006.062

[6.062, 7.438]: Grasp the black pan handle with the left hand

- annotation_rule: A2: 9 words below 10
- visual_contradicted: action: The left hand is grasping the pan handle, but the right hand is also holding the pan, indicating the action is not solely performed by the left hand.
- visual_contradicted: hand: The left hand is grasping the handle, but the right hand is also holding the pan, contradicting the claim that only the left hand is involved.

## epic-cook#009.562

[9.562, 11.812]: Place the black frying pan on the stovetop burner

- annotation_rule: A2: 9 words below 10
- visual_contradicted: action: The pan is being placed on the stovetop, but the candidate annotation incorrectly states the hand as 'LEFT' while the frames show the right hand holding the pan handle during placement.
- visual_contradicted: hand: The pan is held and placed by the right hand, not the left hand as stated in the candidate annotation.

## epic-cook#011.812

[11.812, 13.812]: Release the frying pan on the stovetop burner

- annotation_rule: A2: 8 words below 10
- visual_contradicted: hand: The left hand is seen releasing the frying pan in frame 11.8333, but by frame 12.0667, the hand is no longer visible near the pan, suggesting the action was completed. The candidate annotation incorrectly assigns the hand as 'LEFT' for the entire action, while the right hand is not involved in the release.
- visual_unknown: details: The candidate annotation does not provide any details about the frying pan's color, brand, material, or location beyond the stovetop. The frames do not provide sufficient information to confirm or deny these details.

## epic-cook#013.812

[13.812, 16.062]: Grasp the zucchini with the left hand on the green cutting board

- visual_contradicted: action: The zucchini is grasped with the right hand, not the left hand.
- visual_contradicted: hand: The zucchini is held with the right hand, not the left hand.
- visual_unknown: details: No specific details about color, brand, material, or location beyond the general kitchen setting are provided in the frames.

## epic-cook#016.062

[16.062, 18.188]: Slice the zucchini on the green cutting board with the knife

- visual_contradicted: action: The object being sliced is an eggplant (aubergine), not a zucchini. The candidate annotation incorrectly identifies the vegetable.
- visual_contradicted: object: The object being sliced is an eggplant, not a zucchini. The candidate annotation incorrectly identifies the vegetable.
- visual_unknown: details: The candidate annotation does not provide details about the color, brand, material, location, or contact/success claims of the object or tools. The frames show a green cutting board, a knife with a white handle, and an eggplant, but no specific details are provided in the annotation.

## epic-cook#019.562

[19.562, 21.438]: Slice the zucchini on the green cutting board with the knife

- visual_unknown: details: The color, brand, material, location, and contact or success claims of the zucchini, knife, or cutting board are not specified or verifiable from the frames provided.

## Missing captions


## Input span issues
