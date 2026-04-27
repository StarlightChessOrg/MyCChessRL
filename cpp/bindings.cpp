#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <string>
#include <vector>

#include "xqwl_portable_prefix.h"
#include "xqwl_portable.cpp"

namespace {

struct ZobInitOnce {
  ZobInitOnce() { InitZobrist(); }
} g_zob_once;

inline int COORD_XY_ICCS(int x, int y) { return (x + 3) + ((y + 3) << 4); }

inline int SRC(int mv) { return mv & 255; }
inline int DST(int mv) { return mv >> 8; }
inline int MOVE(int sqSrc, int sqDst) { return sqSrc + sqDst * 256; }

inline void sq_to_iccs(int sq, int &x, int &y) {
  x = (sq & 15) - 3;
  y = (sq >> 4) - 3;
}

std::string mv_to_iccs(int mv) {
  int x1, y1, x2, y2;
  sq_to_iccs(SRC(mv), x1, y1);
  sq_to_iccs(DST(mv), x2, y2);
  return std::to_string(x1) + std::to_string(y1) + "-" + std::to_string(x2) + std::to_string(y2);
}

std::optional<int> iccs_to_move(const std::string &s) {
  if (s.size() != 5 || s[2] != '-')
    return std::nullopt;
  int x1 = s[0] - '0', y1 = s[1] - '0', x2 = s[3] - '0', y2 = s[4] - '0';
  if (x1 < 0 || x1 > 8 || y1 < 0 || y1 > 9 || x2 < 0 || x2 > 8 || y2 < 0 || y2 > 9)
    return std::nullopt;
  return MOVE(COORD_XY_ICCS(x1, y1), COORD_XY_ICCS(x2, y2));
}

char piece_to_fen_char(BYTE pc) {
  if (!pc)
    return '\0';
  static const char *RED = "KABNRCP";
  static const char *BLK = "kabnrcp";
  if (pc < 16) {
    int t = int(pc) - 8;
    if (t < 0 || t > 6)
      return '?';
    return RED[t];
  }
  int t = int(pc) - 16;
  if (t < 0 || t > 6)
    return '?';
  return BLK[t];
}

std::string position_to_fen(const PositionStruct &pos) {
  std::string fen;
  for (int y = 0; y < 10; ++y) {
    int empty = 0;
    for (int x = 0; x < 9; ++x) {
      int sq = COORD_XY_ICCS(x, y);
      BYTE pc = pos.ucpcSquares[sq];
      if (!pc) {
        ++empty;
        continue;
      }
      if (empty) {
        fen += char('0' + empty);
        empty = 0;
      }
      fen += piece_to_fen_char(pc);
    }
    if (empty)
      fen += char('0' + empty);
    if (y < 9)
      fen += '/';
  }
  fen += (pos.sdPlayer == 0) ? " w - - 0 1" : " b - - 0 1";
  return fen;
}

} // namespace

namespace py = pybind11;

struct XQWLPosition {
  PositionStruct pos{};

  XQWLPosition() { reset(); }

  void reset() { pos.Startup(); }

  std::vector<int> legal_moves_mv() const {
    int mvs[MAX_GEN_MOVES];
    int n = pos.GenerateMoves(mvs, FALSE);
    if (n > MAX_GEN_MOVES) {
      n = MAX_GEN_MOVES;
    }
    std::vector<int> out;
    out.reserve(static_cast<size_t>(n));
    // 必须在局面副本上试走：在 const 方法里 const_cast 原地 MakeMove/Undo
    // 若与 Python 侧 fen()/其它调用交错或引擎内部边界情况，会污染真实 pos，表现为随机堆损坏 / double free。
    for (int i = 0; i < n; ++i) {
      PositionStruct trial = pos;
      if (trial.MakeMove(mvs[i])) {
        out.push_back(mvs[i]);
      }
    }
    return out;
  }

  std::vector<std::string> legal_moves_iccs() const {
    auto mvs = legal_moves_mv();
    std::vector<std::string> s;
    s.reserve(mvs.size());
    for (int mv : mvs)
      s.push_back(mv_to_iccs(mv));
    return s;
  }

  bool make_move_mv(int mv) { return pos.MakeMove(mv); }

  bool make_move_iccs(const std::string &iccs) {
    auto mv = iccs_to_move(iccs);
    if (!mv)
      return false;
    return pos.MakeMove(*mv);
  }

  void undo() { pos.UndoMakeMove(); }

  bool in_check() const { return pos.InCheck(); }

  int rep_status(int n_recur = 3) const { return pos.RepStatus(n_recur); }

  int rep_value(int st) const { return pos.RepValue(st); }

  int ply_count() const { return pos.nMoveNum; }

  bool is_mate() const {
    PositionStruct trial = pos;
    return trial.IsMate() != 0;
  }

  int terminal_kind() const {
    if (is_mate())
      return 1;
    int rs = pos.RepStatus(3);
    if (rs > 0)
      return 2;
    if (pos.nMoveNum > 100)
      return 3;
    return 0;
  }

  int rep_value_if_any() const {
    int rs = pos.RepStatus(3);
    if (rs > 0)
      return pos.RepValue(rs);
    return 0;
  }

  std::string fen() const { return position_to_fen(pos); }

  int side_to_move() const { return pos.sdPlayer; }

  /** 深拷贝局面（供多线程 AI 推理等） */
  XQWLPosition copy() const {
    XQWLPosition q;
    q.pos = pos;
    return q;
  }
};

PYBIND11_MODULE(xqwl_core, m) {
  m.doc() = "象棋小巫师 XQWL06 局面核心（重复局面/长打分值/自然限着 ply>100）";

  py::class_<XQWLPosition>(m, "Position")
      .def(py::init<>())
      .def("reset", &XQWLPosition::reset)
      .def("legal_moves_mv", &XQWLPosition::legal_moves_mv)
      .def("legal_moves_iccs", &XQWLPosition::legal_moves_iccs)
      .def("make_move_mv", &XQWLPosition::make_move_mv)
      .def("make_move_iccs", &XQWLPosition::make_move_iccs)
      .def("undo", &XQWLPosition::undo)
      .def("in_check", &XQWLPosition::in_check)
      .def("rep_status", &XQWLPosition::rep_status, py::arg("n_recur") = 3)
      .def("rep_value", &XQWLPosition::rep_value)
      .def("rep_value_if_any", &XQWLPosition::rep_value_if_any)
      .def("ply_count", &XQWLPosition::ply_count)
      .def("is_mate", &XQWLPosition::is_mate)
      .def("terminal_kind", &XQWLPosition::terminal_kind)
      .def("fen", &XQWLPosition::fen)
      .def("side_to_move", &XQWLPosition::side_to_move)
      .def("copy", &XQWLPosition::copy);

  m.attr("MATE_VALUE") = MATE_VALUE;
  m.attr("BAN_VALUE") = BAN_VALUE;
  m.attr("WIN_VALUE") = WIN_VALUE;
}
