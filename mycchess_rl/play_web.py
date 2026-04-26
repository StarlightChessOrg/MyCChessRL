"""网页对弈（Sanic）；规则与特征均依赖 ``xqwl_core``。"""
from __future__ import annotations

import argparse
import asyncio
import threading
from pathlib import Path

import numpy as np
import torch

from mycchess_rl.iccs_util import parse_move_squares
from mycchess_rl.chess.session import GamePlay
from mycchess_rl.model import load_successor_policy_for_play
from mycchess_rl.policy_inference import infer_greedy_move_string

STRATEGY_HUMAN = "人类"
STRATEGY_NEURAL = "纯网络"
STRATEGIES = (STRATEGY_HUMAN, STRATEGY_NEURAL)

_PIECE_CHAR = {
    "R": "车",
    "N": "马",
    "B": "相",
    "A": "仕",
    "K": "帅",
    "C": "炮",
    "P": "兵",
    "r": "车",
    "n": "马",
    "b": "象",
    "a": "士",
    "k": "将",
    "c": "炮",
    "p": "卒",
}


def _piece_side(ch: str | None) -> str | None:
    if not ch:
        return None
    return "red" if ch.isupper() else "black"


def _select_device(gpu: int) -> torch.device:
    if gpu < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{int(gpu)}")


class XqwlWebSession:
    def __init__(self, model, device: torch.device, flist: dict) -> None:
        self._lock = threading.Lock()
        self.model = model
        self.device = device
        self.flist = flist
        self.game = GamePlay()
        self.sel_from: tuple[int, int] | None = None
        self.last_move: tuple[int, int, int, int] | None = None
        self.strategy_red = STRATEGY_NEURAL
        self.strategy_black = STRATEGY_HUMAN
        self._toasts: list[dict[str, str]] = []
        self._ai_busy = False

    def _raw_board(self) -> np.ndarray:
        return self.game.board_view()

    def _legal_strings(self) -> set[str]:
        return set(self.game.legal_moves_iccs_str())

    def _apply_human_move(self, mv: str) -> None:
        if mv not in self._legal_strings():
            return
        self.game.make_move_iccs(mv)
        x1, y1, x2, y2 = parse_move_squares(mv)
        self.last_move = (x1, y1, x2, y2)
        self._check_terminal()

    def _check_terminal(self) -> None:
        t, r = self.game.terminal()
        if not t:
            return
        if r == "checkmate":
            stm = "红方" if self.game.red_to_move else "黑方"
            self._toasts.append({"kind": "info", "title": "终局", "body": f"{stm} 被将死。"})
        else:
            self._toasts.append({"kind": "info", "title": "终局", "body": r})

    def snapshot(self) -> dict:
        with self._lock:
            arr = self._raw_board()
            rows: list[list[dict[str, str | None]]] = []
            for iy in range(10):
                row: list[dict[str, str | None]] = []
                for ix in range(9):
                    ch = arr[iy, ix]
                    if not ch:
                        row.append({"ch": None, "side": None, "label": ""})
                    else:
                        s = str(ch)
                        row.append({"ch": s, "side": _piece_side(s), "label": _PIECE_CHAR.get(s, "?")})
                rows.append(row)
            side = self.game.get_side()
            return {
                "board": rows,
                "visual_sig": "|".join(str(arr[iy, ix] or ".") for iy in range(10) for ix in range(9)),
                "side_to_move": side,
                "sel_from": list(self.sel_from) if self.sel_from else None,
                "last_move": list(self.last_move) if self.last_move else None,
                "strategy_red": self.strategy_red,
                "strategy_black": self.strategy_black,
                "strategies": list(STRATEGIES),
                "status_text": f"轮到 {'红方' if side == 'red' else '黑方'} 走棋",
                "ai_busy": self._ai_busy,
                "current_strategy": self.strategy_red if side == "red" else self.strategy_black,
            }

    def pop_client_messages(self) -> dict:
        with self._lock:
            t, self._toasts = self._toasts, []
            return {"toasts": t, "ai_errors": []}

    def set_strategies(self, red: str, black: str) -> dict | None:
        if red not in STRATEGIES or black not in STRATEGIES:
            return {"error": "非法策略"}
        with self._lock:
            self.strategy_red = red
            self.strategy_black = black
        self.maybe_ai()
        return None

    def new_game(self) -> dict | None:
        with self._lock:
            self.game.reset()
            self.sel_from = None
            self.last_move = None
        self.maybe_ai()
        return None

    def click_cell(self, ix: int, iy: int) -> dict | None:
        with self._lock:
            if self._ai_busy:
                return {"error": "AI 思考中"}
            side = self.game.get_side()
            strat = self.strategy_red if side == "red" else self.strategy_black
            if strat != STRATEGY_HUMAN:
                return {"error": "当前非人类行棋"}
            if not (0 <= ix <= 8 and 0 <= iy <= 9):
                return {"error": "坐标越界"}
            arr = self._raw_board()
            ch = arr[iy, ix]
            if self.sel_from is None:
                if _piece_side(str(ch) if ch else None) == side:
                    self.sel_from = (ix, iy)
                return None
            fx, fy = self.sel_from
            mv = f"{fx}{fy}-{ix}{iy}"
            if mv not in self._legal_strings():
                if _piece_side(str(ch) if ch else None) == side:
                    self.sel_from = (ix, iy)
                else:
                    self.sel_from = None
                return None
            self._apply_human_move(mv)
            self.sel_from = None
        self.maybe_ai()
        return None

    def maybe_ai(self) -> None:
        with self._lock:
            if self._ai_busy:
                return
            side = self.game.get_side()
            strat = self.strategy_red if side == "red" else self.strategy_black
            if strat != STRATEGY_NEURAL:
                return
            if self.game.terminal()[0]:
                return
            if not self.game.legal_moves_iccs_str():
                return
            self._ai_busy = True
            model, device, flist = self.model, self.device, self.flist
            g_copy = self.game.copy()

        def worker() -> None:
            try:
                mv = infer_greedy_move_string(g_copy, model, device, flist)
            except Exception as e:
                with self._lock:
                    self._ai_busy = False
                    self._toasts.append({"kind": "info", "title": "AI 错误", "body": str(e)})
                return
            with self._lock:
                self._ai_busy = False
                if mv not in self._legal_strings():
                    self._toasts.append({"kind": "info", "title": "AI", "body": f"非法着法 {mv}"})
                    return
                self._apply_human_move(mv)
            self.maybe_ai()

        threading.Thread(target=worker, daemon=True).start()


def _html_page() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>MyCChessRL 象棋对弈</title>
  <style>
    :root { --bg0:#1a1510; --bg1:#2d2419; --panel:#352a22; --panel2:#2a2218; --line:rgba(74,50,37,.45);
      --board-bg0:#f2e8d4; --board-bg1:#e5d3b6; --red:#c62828; --black:#1565c0; --text:#f2ebe3;
      --muted:rgba(242,235,227,.72); --accent:#ff9800; --sel:#ffeb3b; --radius:14px; }
    *{box-sizing:border-box} body{margin:0;font-family:"Microsoft YaHei","PingFang SC",sans-serif;color:var(--text);
      min-height:100vh;background:radial-gradient(120% 80% at 50% 0%,var(--bg1) 0%,var(--bg0) 55%,#120e0a 100%)}
    .shell{max-width:1320px;margin:0 auto;min-height:100vh;padding:clamp(14px,2.2vw,28px);
      display:grid;grid-template-columns:minmax(0,1fr) minmax(260px,320px);gap:clamp(16px,2.5vw,32px);align-items:center}
    @media(max-width:860px){.shell{grid-template-columns:1fr;align-items:start}}
    .board-wrap{display:flex;justify-content:center;align-items:center}
    .board-card{background:linear-gradient(145deg,#faf3e6 0%,var(--board-bg1) 100%);border-radius:var(--radius);
      padding:clamp(10px,1.4vw,16px);box-shadow:0 4px 0 rgba(62,39,35,.35),0 18px 48px rgba(0,0,0,.45);
      border:1px solid rgba(62,39,35,.25)}
    .board{--cs:clamp(42px,min((100vw - 48px)/9.6,(100vh - 120px)/10.2),76px);display:grid;
      grid-template-columns:repeat(9,var(--cs));grid-template-rows:repeat(10,var(--cs));
      width:calc(9 * var(--cs));height:calc(10 * var(--cs));
      background:linear-gradient(180deg,var(--board-bg0) 0%,var(--board-bg1) 100%);border-radius:10px;overflow:hidden}
    .cell{border:1px solid var(--line);cursor:pointer;display:flex;align-items:center;justify-content:center;user-select:none}
    .cell:hover{background:rgba(255,255,255,.14)}
    .piece-red,.piece-black{color:#fff;border-radius:50%;width:calc(var(--cs)*.78);height:calc(var(--cs)*.78);
      min-width:32px;min-height:32px;display:flex;align-items:center;justify-content:center;border:2px solid #3e2723;
      font-size:clamp(15px,calc(var(--cs)*.38),30px);font-weight:700;box-shadow:0 2px 6px rgba(0,0,0,.22)}
    .piece-red{background:linear-gradient(165deg,#e53935 0%,var(--red) 55%,#8b0000 100%)}
    .piece-black{background:linear-gradient(165deg,#42a5f5 0%,var(--black) 55%,#0d47a1 100%)}
    .sel{outline:3px solid var(--sel);outline-offset:-3px;border-radius:4px}
    .last-from,.last-to{box-shadow:inset 0 0 0 3px var(--accent)}
    .sidepanel{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);padding:clamp(16px,2vw,22px);
      border-radius:var(--radius);border:1px solid rgba(255,255,255,.06);box-shadow:0 12px 40px rgba(0,0,0,.35)}
    h1{font-size:clamp(1.05rem,2.2vw,1.25rem);margin:0 0 6px;font-weight:600}
    .subtitle{font-size:12px;color:var(--muted);margin-bottom:14px}
    label{display:block;margin-top:10px;font-size:13px;color:var(--muted)}
    select{width:100%;padding:10px 12px;margin-top:6px;border-radius:10px;border:1px solid rgba(93,78,58,.6);
      background:rgba(0,0,0,.25);color:var(--text);font-size:14px}
    button{margin-top:16px;padding:12px 16px;border:none;border-radius:10px;
      background:linear-gradient(180deg,#8d6e63 0%,#6d4c41 100%);color:#fff;font-size:15px;font-weight:600;cursor:pointer;width:100%}
    #status{margin-top:16px;white-space:pre-wrap;font-size:13px;padding:12px 14px;background:rgba(0,0,0,.22);border-radius:10px;min-height:4.5em}
    .ai-busy .board{opacity:.92;pointer-events:none}
  </style>
</head>
<body>
  <div class="shell" id="shell">
    <div class="board-wrap"><div class="board-card"><div class="board" id="board"></div></div></div>
    <div class="sidepanel">
      <h1>MyCChessRL 象棋对弈</h1>
      <div class="subtitle">XQWL 规则核 · Sanic · 两阶段策略网络</div>
      <label>红方策略</label><select id="sel-red"></select>
      <label>黑方策略</label><select id="sel-black"></select>
      <button type="button" id="btn-new">新局</button>
      <div id="status"></div>
    </div>
  </div>
<script>
(function(){
  const shell=document.getElementById("shell"),boardEl=document.getElementById("board"),statusEl=document.getElementById("status");
  const selRed=document.getElementById("sel-red"),selBlack=document.getElementById("sel-black"),btnNew=document.getElementById("btn-new");
  let lastVisualSig=null,pollTimer=null;
  function showAlert(t,b){alert(t+"\\n\\n"+b);}
  function fillStrategiesOnce(strategies){
    if(selRed.options.length>0)return;
    strategies.forEach(function(t){var o=document.createElement("option");o.value=o.textContent=t;selRed.appendChild(o);});
    strategies.forEach(function(t){var o=document.createElement("option");o.value=o.textContent=t;selBlack.appendChild(o);});
  }
  function renderCells(snap){
    var lm=snap.last_move,sf=snap.sel_from,frag=document.createDocumentFragment();
    for(var iy=0;iy<10;iy++)for(var ix=0;ix<9;ix++){
      var cell=document.createElement("div");cell.className="cell";cell.dataset.ix=String(ix);cell.dataset.iy=String(iy);
      if(lm&&ix===lm[0]&&iy===lm[1])cell.classList.add("last-from");
      if(lm&&ix===lm[2]&&iy===lm[3])cell.classList.add("last-to");
      if(sf&&ix===sf[0]&&iy===sf[1])cell.classList.add("sel");
      var sq=snap.board[iy][ix];
      if(sq.ch){var span=document.createElement("span");span.textContent=sq.label||"?";
        span.className=sq.side==="red"?"piece-red":"piece-black";cell.appendChild(span);}
      frag.appendChild(cell);
    }
    boardEl.replaceChildren(frag);
  }
  function applySnap(snap){
    fillStrategiesOnce(snap.strategies||[]);selRed.value=snap.strategy_red;selBlack.value=snap.strategy_black;
    statusEl.textContent=snap.status_text||"";
    var sig=snap.visual_sig!=null?snap.visual_sig:JSON.stringify(snap.board);
    if(sig!==lastVisualSig){lastVisualSig=sig;renderCells(snap);}
    shell.classList.toggle("ai-busy",!!snap.ai_busy);
  }
  function handleMessages(msg){
    (msg.toasts||[]).forEach(function(t){if(t.kind==="info")showAlert(t.title||"提示",t.body||"");});
    (msg.ai_errors||[]).forEach(function(e){showAlert("AI 错误",e);});
  }
  function armPoll(ms){if(pollTimer)clearTimeout(pollTimer);pollTimer=setTimeout(onePoll,ms);}
  function onePoll(){
    pollTimer=null;
    Promise.all([fetch("/api/state",{cache:"no-store"}).then(function(r){return r.json();}),
      fetch("/api/messages",{cache:"no-store"}).then(function(r){return r.json();})])
      .then(function(pair){applySnap(pair[0]);handleMessages(pair[1]);armPoll(pair[0].ai_busy?160:380);})
      .catch(function(){armPoll(700);});
  }
  boardEl.addEventListener("click",function(ev){
    var cell=(ev.target.closest&&ev.target.closest(".cell"))||null;
    if(!cell||shell.classList.contains("ai-busy"))return;
    var ix=parseInt(cell.dataset.ix,10),iy=parseInt(cell.dataset.iy,10);
    if(isNaN(ix)||isNaN(iy))return;
    fetch("/api/click",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({ix:ix,iy:iy})})
      .then(function(r){return r.json();}).then(function(j){if(j.error)showAlert("走子",j.error);else armPoll(25);}).catch(function(){armPoll(25);});
  });
  selRed.addEventListener("change",function(){
    fetch("/api/strategies",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({red:selRed.value,black:selBlack.value})}).then(function(r){return r.json();})
      .then(function(j){if(j.error)showAlert("策略",j.error);else armPoll(25);});
  });
  selBlack.addEventListener("change",function(){
    fetch("/api/strategies",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({red:selRed.value,black:selBlack.value})}).then(function(r){return r.json();})
      .then(function(j){if(j.error)showAlert("策略",j.error);else armPoll(25);});
  });
  btnNew.addEventListener("click",function(){
    fetch("/api/new_game",{method:"POST"}).then(function(r){return r.json();})
      .then(function(j){if(j.error)showAlert("新局",j.error);else armPoll(25);});
  });
  onePoll();
})();
</script>
</body>
</html>"""


def main() -> None:
    try:
        from sanic import Sanic
        from sanic.response import html, json
    except ImportError as e:
        raise SystemExit("请安装: pip install 'sanic>=23.12' 或 pip install -e '.[play]'") from e

    p = argparse.ArgumentParser(description="MyCChessRL 网页对弈（Sanic）")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="Sanic worker 数（>1 时勿依赖本进程内单会话）")
    args = p.parse_args()

    device = _select_device(int(args.gpu))
    model, flist = load_successor_policy_for_play(args.checkpoint, device)
    session = XqwlWebSession(model, device, flist)

    app = Sanic("mycchess_rl_play_web")
    app.config.RESPONSE_TIMEOUT = 120

    @app.get("/")
    async def _index(_request):
        return html(_html_page())

    @app.get("/api/state")
    async def _api_state(_request):
        return json(session.snapshot())

    @app.get("/api/messages")
    async def _api_messages(_request):
        return json(session.pop_client_messages())

    @app.post("/api/click")
    async def _api_click(request):
        data = request.json
        if not isinstance(data, dict):
            return json({"error": "无效 JSON"}, status=400)
        try:
            ix = int(data.get("ix", -1))
            iy = int(data.get("iy", -1))
        except (TypeError, ValueError):
            return json({"error": "坐标无效"}, status=400)
        err = session.click_cell(ix, iy)
        return json(err or {})

    @app.post("/api/strategies")
    async def _api_strategies(request):
        data = request.json
        if not isinstance(data, dict):
            return json({"error": "参数无效"}, status=400)
        red, black = data.get("red"), data.get("black")
        if not isinstance(red, str) or not isinstance(black, str):
            return json({"error": "参数无效"}, status=400)
        err = session.set_strategies(red, black)
        return json(err or {})

    @app.post("/api/new_game")
    async def _api_new_game(_request):
        err = session.new_game()
        return json(err or {})

    @app.after_server_start
    async def _kick_ai(_app, _loop):
        await asyncio.sleep(0.2)
        session.maybe_ai()

    print(f"[play] http://{args.host}:{args.port}/  (Sanic workers={args.workers})", flush=True)
    app.run(
        host=str(args.host),
        port=int(args.port),
        workers=int(args.workers),
        access_log=False,
        motd=False,
    )


if __name__ == "__main__":
    main()
