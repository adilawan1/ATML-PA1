from __future__ import annotations

"""Shape/texture cue-conflict generation (Task 1, Step 3): AdaIN style transfer between >=5
unordered class pairs from the chosen dataset, both directions where feasible, targeting
>=200 valid conflicts balanced across pairs/directions.

Planned approach: clone a public AdaIN implementation (e.g. naoto0804/pytorch-AdaIN) as
`third_party/pytorch-AdaIN/` (NOT committed -- see .gitignore -- clone fresh each session, or
vendor just the model + weights loading code and cite it in the top-level README's
attribution section). Content image = shape/content class A, style image = texture/style
class B; repeat with A/B swapped.

Rejection rule (define BEFORE running any model on the conflicts, per the assignment --
do not use model predictions to decide what to keep): a stylization is REJECTED if, by visual
inspection, the transferred texture has not visibly overwritten the content image's original
texture/color statistics (e.g. AdaIN style weight too low) or the object silhouette from the
content image is no longer identifiable. Record accepted/rejected counts either way.

TODO (Task 1, Day 2):
1. Pick >=5 unordered STL-10 class pairs.
2. For each pair (A, B), run AdaIN with content=A images, style=B images, and content=B,
   style=A, at your chosen style-transfer strength (a stated experimental-design choice).
3. Apply the rejection rule; save (image_path_or_array, shape_label, texture_label,
   pair_id, direction) for every ACCEPTED conflict to `task1/data/cue_conflicts_seed6304.json`
   (or a small saved tensor file), plus separately record rejected counts per pair/direction.
"""

from dataclasses import dataclass
from typing import List

REJECTION_RULE = (
    "Reject if the content image's original texture/color statistics are still visually "
    "dominant (style transfer too weak) or the object silhouette is no longer identifiable "
    "(style transfer destroyed content structure)."
)


@dataclass
class CueConflictRecord:
    image_path: str
    shape_label: int
    texture_label: int
    pair_id: str
    direction: str  # "A_shape_B_texture" or "B_shape_A_texture"


def generate_cue_conflicts(*args, **kwargs) -> List[CueConflictRecord]:
    raise NotImplementedError("Wire up a public AdaIN implementation per the module docstring on Task 1's Day 2.")
