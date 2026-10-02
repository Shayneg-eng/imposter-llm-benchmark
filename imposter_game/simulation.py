"""
Simulation runner — sweeps parameter combinations and saves one JSON per game.

Usage:
    from imposter_game.simulation import run_sweep
    run_sweep(output_dir="imposter_game/results")

Tweakable parameters in DEFAULT_SWEEP:
    n_players      : total players
    n_imposters    : how many are imposters (don't know the word)
    clue_rounds    : how many rounds of clues before a vote
    allow_guess    : whether a caught imposter gets a word-guess attempt
    word_category  : restrict word bank (None = any; options in words.CATEGORIES)
    model          : LLM model string
    temperature    : LLM sampling temperature
"""

import json
import os
import time
import itertools
from typing import Any
from imposter_game.game import Game
from imposter_game.words import CATEGORIES


# ── Default sweep config ──────────────────────────────────────────────────────
# Edit any list to add/remove values. All valid combinations are tested.

DEFAULT_SWEEP: dict[str, list[Any]] = {
    "n_players":     [4, 6, 8],
    "n_imposters":   [1, 2],
    "clue_rounds":   [1, 2, 3],
    "allow_guess":   [True],
    "word_category": [None],           # None = random from full bank; or e.g. ["animal","food"]
    "model":         ["GPT-5.4-Nano"],
    "temperature":   [0.7],
}


def _valid_combo(p: dict) -> bool:
    return 1 <= p["n_imposters"] < p["n_players"]


def run_sweep(
    sweep_params: dict[str, list[Any]] | None = None,
    games_per_config: int = 3,
    output_dir: str = "imposter_game/results",
    verbose: bool = True,
) -> list[dict]:
    """Run all valid parameter combinations `games_per_config` times each."""
    params = sweep_params or DEFAULT_SWEEP
    os.makedirs(output_dir, exist_ok=True)

    keys = list(params.keys())
    combos = [
        dict(zip(keys, vals))
        for vals in itertools.product(*params.values())
        if _valid_combo(dict(zip(keys, vals)))
    ]

    total = len(combos) * games_per_config
    print(f"\nRunning {len(combos)} configs × {games_per_config} games = {total} total\n")

    all_results, counter = [], 0
    for combo in combos:
        for run_idx in range(games_per_config):
            counter += 1
            cat_tag = combo["word_category"] or "any"
            game_id = (
                f"n{combo['n_players']}"
                f"_k{combo['n_imposters']}"
                f"_r{combo['clue_rounds']}"
                f"_{cat_tag}"
                f"_run{run_idx}"
                f"_{int(time.time())}"
            )
            print(f"[{counter}/{total}] {game_id}")
            try:
                g = Game(
                    n_players=combo["n_players"],
                    n_imposters=combo["n_imposters"],
                    clue_rounds=combo["clue_rounds"],
                    allow_guess=combo["allow_guess"],
                    word_category=combo["word_category"],
                    model=combo["model"],
                    temperature=combo["temperature"],
                    game_id=game_id,
                )
                result = g.play(verbose=verbose)
                out_path = os.path.join(output_dir, f"{game_id}.json")
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                all_results.append(result)
            except Exception as exc:
                print(f"  ERROR: {exc}")

    print(f"\nDone. {len(all_results)}/{total} games completed.")
    return all_results


def run_single(
    n_players: int = 5,
    n_imposters: int = 1,
    clue_rounds: int = 2,
    allow_guess: bool = True,
    word_category: str | None = None,
    model: str = "GPT-5.4-Nano",
    temperature: float = 0.7,
    output_dir: str = "imposter_game/results",
    verbose: bool = True,
) -> dict:
    """Run a single game and save it."""
    os.makedirs(output_dir, exist_ok=True)
    game_id = f"single_n{n_players}_k{n_imposters}_r{clue_rounds}_{int(time.time())}"
    g = Game(
        n_players=n_players,
        n_imposters=n_imposters,
        clue_rounds=clue_rounds,
        allow_guess=allow_guess,
        word_category=word_category,
        model=model,
        temperature=temperature,
        game_id=game_id,
    )
    result = g.play(verbose=verbose)
    out_path = os.path.join(output_dir, f"{game_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out_path}")
    return result
