"""对弈会话：仅 ``xqwl_core`` 规则（见 ``mycchess_rl.xqwl_state``）。"""

from __future__ import annotations

from mycchess_rl.xqwl_state import XqwlGameState

GamePlay = XqwlGameState


def legal_moves_iccs_for_board(gp: XqwlGameState) -> list[tuple[int, int, int, int]]:
    return gp.legal_moves_iccs()
