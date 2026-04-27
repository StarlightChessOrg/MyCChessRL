#pragma once
// 从「象棋小巫师」XQWL06.CPP 抽离时的 MSVC 类型替换（不含 Windows / 搜索 / UI）
#include <cstdint>
#include <cstring>

#ifndef TRUE
#define TRUE true
#endif
#ifndef FALSE
#define FALSE false
#endif

using BYTE = std::uint8_t;
using BOOL = bool;
using DWORD = std::uint32_t;
using WORD = std::uint16_t;

// 棋盘范围与棋子编号（与原文件一致）
const int RANK_TOP = 3;
const int RANK_BOTTOM = 12;
const int FILE_LEFT = 3;
const int FILE_RIGHT = 11;

const int PIECE_KING = 0;
const int PIECE_ADVISOR = 1;
const int PIECE_BISHOP = 2;
const int PIECE_KNIGHT = 3;
const int PIECE_ROOK = 4;
const int PIECE_CANNON = 5;
const int PIECE_PAWN = 6;

const int MAX_GEN_MOVES = 128;
// 历史着法栈；原 256 不足以覆盖长棋谱（IMSA 等），越界写 mvsList 会触发 stack smashing / 堆损坏
const int MAX_MOVES = 1024;
const int LIMIT_DEPTH = 64;
const int MATE_VALUE = 10000;
const int BAN_VALUE = MATE_VALUE - 100;
const int WIN_VALUE = MATE_VALUE - 200;
const int DRAW_VALUE = 20;
const int ADVANCED_VALUE = 3;
const int NULL_MARGIN = 400;
