// Standard headers BEFORE pybind11 to avoid cmath/random ambiguity on GCC 11
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "board.h"
#include "mcts.h"
#include "mcts_node.h"
#include "eval_thread.h"
#include "go_features.h"

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

    // Zero-copy variant: write features into a pre-allocated output buffer.
    // Avoids numpy array allocation in the hot MCTS eval loop.
    m.def("board_to_features_into", [](const Board& board, uint8_t color,
                                        py::array_t<float> out, int offset) {
        auto f = board_to_features(board, color, pass_move());
        int s = board.size();
        auto buf = out.mutable_unchecked<4>(); // shape: (batch, C, H, W)
        for (int ch = 0; ch < NUM_FEATURES; ch++)
            for (int r = 0; r < s; r++)
                for (int c = 0; c < s; c++)
                    buf(offset, ch, r, c) = f.data[ch][r][c];
    }, py::arg("board"), py::arg("color"), py::arg("out"), py::arg("offset"));

    // ── MCTS Config ────────────────────────────────────────────────
    py::class_<MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("num_simulations", &MCTSConfig::num_simulations)
        .def_readwrite("c_puct", &MCTSConfig::c_puct)
        .def_readwrite("dirichlet_alpha", &MCTSConfig::dirichlet_alpha)
        .def_readwrite("dirichlet_weight", &MCTSConfig::dirichlet_weight)
        .def_readwrite("num_threads", &MCTSConfig::num_threads)
        .def_readwrite("min_batch_size", &MCTSConfig::min_batch_size)
        .def_readwrite("max_batch_size", &MCTSConfig::max_batch_size)
        .def_readwrite("batch_timeout_us", &MCTSConfig::batch_timeout_us)
        .def_readwrite("virtual_loss_value", &MCTSConfig::virtual_loss_value);

    // ── Parallel Stats ─────────────────────────────────────────────
    py::class_<ParallelStats>(m, "ParallelStats")
        .def_readonly("sims_per_thread", &ParallelStats::sims_per_thread)
        .def_readonly("num_batches", &ParallelStats::num_batches)
        .def_readonly("total_batch_items", &ParallelStats::total_batch_items);

    // ── MCTS Result ────────────────────────────────────────────────
    py::class_<MCTSResult>(m, "MCTSResult")
        .def_readonly("best_move", &MCTSResult::best_move)
        .def_property_readonly("policy_vec", [](const MCTSResult& r) {
            return py::array_t<float>(r.policy_vec.size(), r.policy_vec.data());
        })
        .def_readonly("total_visits", &MCTSResult::total_visits)
        .def_readonly("parallel_stats", &MCTSResult::parallel_stats);

    // ── NetOutput ──────────────────────────────────────────────────
    py::class_<NetOutput>(m, "NetOutput")
        .def(py::init<>())
        .def_readwrite("policy", &NetOutput::policy)
        .def_readwrite("value", &NetOutput::value);

    // ── Shared Evaluator ─────────────────────────────────────────
    py::class_<SharedEvaluator>(m, "SharedEvaluator")
        .def(py::init<int, int, int>(),
             py::arg("min_batch"), py::arg("max_batch"), py::arg("timeout_us") = 100)
        .def("start", [](SharedEvaluator& self, const NetBatchEvalFn& fn) {
            self.start(fn);
        }, py::arg("batch_eval_fn"))
        .def("stop", [](SharedEvaluator& self) {
            py::gil_scoped_release release;
            self.stop();
        })
        .def("num_batches", &SharedEvaluator::num_batches)
        .def("total_items", &SharedEvaluator::total_items);

    // ── MCTS Search ────────────────────────────────────────────────
    py::class_<MCTSSearch>(m, "MCTSSearch")
        .def(py::init<MCTSConfig>(), py::arg("config") = MCTSConfig())
        .def("search", &MCTSSearch::search,
             py::arg("board"), py::arg("color"), py::arg("eval_fn"),
             py::arg("temperature") = 1.0)
        .def("search_batch", &MCTSSearch::search_batch,
             py::arg("boards"), py::arg("colors"), py::arg("batch_eval_fn"),
             py::arg("temperature") = 1.0)
        .def("search_parallel", [](MCTSSearch& self, const Board& board,
             uint8_t color, const NetBatchEvalFn& fn, double temp) {
            py::gil_scoped_release release;
            return self.search_parallel(board, color, fn, temp);
        }, py::arg("board"), py::arg("color"), py::arg("batch_eval_fn"),
           py::arg("temperature") = 1.0)
        .def("search_parallel_shared", [](MCTSSearch& self, const Board& board,
             uint8_t color, SharedEvaluator& shared_eval, double temp) -> MCTSResult {
            py::gil_scoped_release release;
            return self.search_parallel_shared(board, color, shared_eval.queue(), temp);
        }, py::arg("board"), py::arg("color"), py::arg("shared_eval"),
           py::arg("temperature") = 1.0);

    // ── Constants ──────────────────────────────────────────────────
    m.attr("EMPTY") = EMPTY;
    m.attr("BLACK") = BLACK;
    m.attr("WHITE") = WHITE;
}
