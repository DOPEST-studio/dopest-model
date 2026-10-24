"""Code-labelled decision generators. Each module exposes generate_case(seed) -> list of items (one case's group)."""
from . import jaggedness, judge_response, oncall_routing, probability, state_tracking, temporal_numeric, temporal_v2, water_claim

GENERATORS = {"water_claim": water_claim.generate_case, "temporal_numeric": temporal_numeric.generate_case,
              "probability": probability.generate_case, "jaggedness": jaggedness.generate_case,
              "temporal_v2": temporal_v2.generate_case, "judge_response": judge_response.generate_case,
              "oncall_routing": oncall_routing.generate_case, "state_tracking": state_tracking.generate_case}
