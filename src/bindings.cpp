#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "board.h"
#include "mcts.h"
#include "mcts_node.h"
#include "features.h"

namespace py = pybind11;
using namespace alphago;

PYBIND11_MODULE(alphago_core, m) {
    m.doc() = "C++ accelerated Go engine and MCTS for AlphaZero";

    // ── Move ───────────────────────────────────────────────────────
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("row", &Move::row)
        .def_readwrite("col", &Move::col)
        .def("is_pass", &Move::is_pass)
        .def("__repr__", [](const Move& m) {
            if (m.is_pass()) return std::string("Move(pass)");
            return "Move(" + std::to_string(m.row) + ", " + std::to_string(m.col) + ")";
        });

    m.def("pass_move", &pass_move);
    m.def("board_move", &board_move);

    // ── Board ──────────────────────────────────────────────────────
    py::class_<Board>(m, "CBoard")
        .def(py::init<int>(), py::arg("size") = 13)
        .def("size", &Board::size)
        .def("at", &Board::at)
        .def("hash", &Board::hash)
        .def("is_legal", &Board::is_legal)
        .def("play", &Board::play)
        .def("legal_moves", &Board::legal_moves)
        .def("score", [](const Board& b) {
            auto s = b.score();
            py::dict d;
            d["black"] = s.black;
            d["white"] = s.white;
            d["winner"] = s.winner == BLACK ? "black" : (s.winner == WHITE ? "white" : "draw");
            d["margin"] = s.margin;
            d["black_territory"] = s.black_territory;
            d["white_territory"] = s.white_territory;
            d["black_stones"] = s.black_stones;
            d["white_stones"] = s.white_stones;
            return d;
        })
        .def("captured_by", &Board::captured_by)
        .def("move_count", &Board::move_count)
        .def("copy", [](const Board& b) { return Board(b); })
        .def_readwrite("komi", &Board::komi)
        // Expose grid as numpy array (read-only view, no copy)
        .def_property_readonly("grid", [](const Board& b) {
            int s = b.size();
            auto arr = py::array_t<uint8_t>({s, s});
            auto buf = arr.mutable_unchecked<2>();
            for (int r = 0; r < s; r++)
                for (int c = 0; c < s; c++)
                    buf(r, c) = b.at(r, c);
            return arr;
        });

    // ── Features ───────────────────────────────────────────────────
    m.def("board_to_features_cpp", [](const Board& board, uint8_t color, int last_r, int last_c) {
        Move last = (last_r >= 0) ? board_move(last_r, last_c) : pass_move();
        auto f = board_to_features(board, color, last);
        int s = board.size();
        auto arr = py::array_t<float>({NUM_FEATURES, s, s});
        auto buf = arr.mutable_unchecked<3>();
        for (int ch = 0; ch < NUM_FEATURES; ch++)
            for (int r = 0; r < s; r++)
                for (int c = 0; c < s; c++)
                    buf(ch, r, c) = f.data[ch][r][c];
        return arr;
    }, py::arg("board"), py::arg("color"), py::arg("last_r") = -1, py::arg("last_c") = -1);

    // ── MCTS Config ────────────────────────────────────────────────
    py::class_<MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("num_simulations", &MCTSConfig::num_simulations)
        .def_readwrite("c_puct", &MCTSConfig::c_puct)
        .def_readwrite("dirichlet_alpha", &MCTSConfig::dirichlet_alpha)
        .def_readwrite("dirichlet_weight", &MCTSConfig::dirichlet_weight);

    // ── MCTS Result ────────────────────────────────────────────────
    py::class_<MCTSResult>(m, "MCTSResult")
        .def_readonly("best_move", &MCTSResult::best_move)
        .def_property_readonly("policy_vec", [](const MCTSResult& r) {
            return py::array_t<float>(r.policy_vec.size(), r.policy_vec.data());
        })
        .def_readonly("total_visits", &MCTSResult::total_visits);

    // ── NetOutput ──────────────────────────────────────────────────
    py::class_<NetOutput>(m, "NetOutput")
        .def(py::init<>())
        .def_readwrite("policy", &NetOutput::policy)
        .def_readwrite("value", &NetOutput::value);

    // ── MCTS Search ────────────────────────────────────────────────
    py::class_<MCTSSearch>(m, "MCTSSearch")
        .def(py::init<MCTSConfig>(), py::arg("config") = MCTSConfig())
        .def("search", &MCTSSearch::search,
             py::arg("board"), py::arg("color"), py::arg("eval_fn"),
             py::arg("temperature") = 1.0)
        .def("search_batch", &MCTSSearch::search_batch,
             py::arg("boards"), py::arg("colors"), py::arg("batch_eval_fn"),
             py::arg("temperature") = 1.0);

    // ── Constants ──────────────────────────────────────────────────
    m.attr("EMPTY") = EMPTY;
    m.attr("BLACK") = BLACK;
    m.attr("WHITE") = WHITE;
}
