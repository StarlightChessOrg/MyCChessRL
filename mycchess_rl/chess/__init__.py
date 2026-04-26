"""棋盘特征编码（无 cchess；规则在 ``xqwl_core``）。"""

from mycchess_rl.chess.features import (
    FEATURE_LIST,
    encode_model_planes,
    encode_picker_planes,
    encode_signed_seven_planes,
    orient_planes_for_model,
)
from mycchess_rl.iccs_util import parse_move_squares
from mycchess_rl.chess.plane_extras import EXTRA_HINT_PLANE_COUNT, encode_extra_hint_planes
from mycchess_rl.chess.rationale import (
    PIECE_PLANE_COUNT,
    POLICY_GRID_NUMEL,
    POLICY_MAX_LEGAL_MOVES,
    POLICY_SELECT_IN_CHANNELS,
    RATIONALE_PLANE_COUNT,
    RED_OUTCOME_DRAW,
    RED_OUTCOME_LOSS,
    RED_OUTCOME_WIN,
    STM_OUTCOME_DRAW,
    STM_OUTCOME_LOSS,
    STM_OUTCOME_WIN,
    VALUE_LABEL_IGNORE,
    stm_outcome_class_from_red_outcome,
)
from mycchess_rl.chess.session import GamePlay, legal_moves_iccs_for_board

__all__ = [
    "FEATURE_LIST",
    "PIECE_PLANE_COUNT",
    "RATIONALE_PLANE_COUNT",
    "EXTRA_HINT_PLANE_COUNT",
    "POLICY_GRID_NUMEL",
    "POLICY_MAX_LEGAL_MOVES",
    "POLICY_SELECT_IN_CHANNELS",
    "RED_OUTCOME_WIN",
    "RED_OUTCOME_DRAW",
    "RED_OUTCOME_LOSS",
    "STM_OUTCOME_WIN",
    "STM_OUTCOME_DRAW",
    "STM_OUTCOME_LOSS",
    "VALUE_LABEL_IGNORE",
    "encode_model_planes",
    "encode_picker_planes",
    "encode_signed_seven_planes",
    "orient_planes_for_model",
    "parse_move_squares",
    "encode_extra_hint_planes",
    "stm_outcome_class_from_red_outcome",
    "GamePlay",
    "legal_moves_iccs_for_board",
]
