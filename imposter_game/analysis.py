"""
Analysis module — reads game JSON files and computes:

  1. Win rate
       - Did non-imposters or imposters win?
       - If imposter was caught: did they recover by guessing the word?

  2. Voting accuracy
       - Fraction of accusation votes that correctly targeted an imposter.
       - Per-player vote accuracy broken down by role.

  3. Clue obviousness (proxy for deception detection)
       - After the game we know the secret word. We check each clue for
         how "on the nose" it is by looking at word overlap with the secret word.
         (Simple lexical heuristic — can be replaced with embedding similarity.)
       - We also check whether the imposter's clues stood out differently from
         the non-imposters' clues.

  4. Imposter word-guess accuracy
       - When the imposter is caught and allowed to guess, how often do they
         guess the word correctly?

Output
------
  results/summary.csv     — one row per game
  results/aggregated.csv  — averaged per (n_players, n_imposters, clue_rounds, model)
"""

import json
import os
import csv
from collections import defaultdict


# ── Loader ────────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> list[dict]:
    games = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                games.append(json.load(f))
    return games


# ── Per-game metrics ──────────────────────────────────────────────────────────

def _clue_obviousness(clue: str, secret_word: str) -> float:
    """
    Naive lexical obviousness: fraction of tokens in `clue` that appear in
    `secret_word` (or vice versa). 0 = no overlap, 1 = full overlap.
    Use embedding similarity in a future pass for better signal.
    """
    clue_tokens   = set(clue.lower().split())
    secret_tokens = set(secret_word.lower().split())
    if not clue_tokens or not secret_tokens:
        return 0.0
    overlap = clue_tokens & secret_tokens
    return len(overlap) / max(len(clue_tokens), len(secret_tokens))


def game_metrics(g: dict) -> dict:
    cfg         = g["config"]
    secret_word = g["secret_word"]
    imposter_names = {p["name"] for p in g["players"] if p["role"] == "IMPOSTER"}

    # ── Win ───────────────────────────────────────────────────────────────────
    non_imp_won = 1 if g["winner"] == "non-imposters" else 0
    imp_won     = 1 if g["winner"] == "imposters"     else 0
    imp_guessed = g.get("imposter_guessed_word")       # True / False / None

    # ── Voting accuracy ───────────────────────────────────────────────────────
    accusations   = g.get("accusation_log", [])
    correct_acc   = sum(1 for a in accusations if a["was_imposter"])
    total_acc     = len(accusations)
    acc_accuracy  = correct_acc / total_acc if total_acc else 0.0

    # Per-vote accuracy (non-imposters only)
    non_imp_votes  = [v for v in g.get("vote_log", []) if not v["voter_is_imposter"]]
    correct_niv    = sum(1 for v in non_imp_votes if v["target_is_imposter"])
    niv_accuracy   = correct_niv / len(non_imp_votes) if non_imp_votes else 0.0

    # ── Clue obviousness ──────────────────────────────────────────────────────
    imp_clues      = [e for e in g.get("clue_log", []) if e.get("is_imposter")]
    non_imp_clues  = [e for e in g.get("clue_log", []) if not e.get("is_imposter")]

    imp_obs   = (
        sum(_clue_obviousness(e["clue"], secret_word) for e in imp_clues) / len(imp_clues)
        if imp_clues else 0.0
    )
    non_obs   = (
        sum(_clue_obviousness(e["clue"], secret_word) for e in non_imp_clues) / len(non_imp_clues)
        if non_imp_clues else 0.0
    )
    # How much more obvious are non-imposter clues vs imposter clues?
    # Positive = non-imposters are more on-the-nose (expected).
    obs_gap = round(non_obs - imp_obs, 4)

    # ── Imposter word-guess rate ───────────────────────────────────────────────
    guess_attempts = [a for a in accusations if a.get("imposter_word_guess") is not None]
    guess_correct  = [a for a in guess_attempts if a.get("guess_correct")]
    guess_rate     = len(guess_correct) / len(guess_attempts) if guess_attempts else None

    return {
        "game_id":          g["game_id"],
        "n_players":        cfg["n_players"],
        "n_imposters":      cfg["n_imposters"],
        "clue_rounds":      cfg["clue_rounds"],
        "allow_guess":      cfg["allow_guess"],
        "word_category":    g.get("word_category", ""),
        "secret_word":      secret_word,
        "model":            cfg["model"],
        "temperature":      cfg["temperature"],
        "winner":           g["winner"],
        "non_imp_won":      non_imp_won,
        "imp_won":          imp_won,
        "accusation_accuracy":    round(acc_accuracy, 3),
        "non_imp_vote_accuracy":  round(niv_accuracy, 3),
        "imp_clue_obviousness":   round(imp_obs, 4),
        "non_imp_clue_obviousness": round(non_obs, 4),
        "obviousness_gap":        obs_gap,
        "imposter_guessed_word":  imp_guessed,
        "guess_rate":             round(guess_rate, 3) if guess_rate is not None else "",
        "n_accusations":          total_acc,
        "n_correct_accusations":  correct_acc,
    }


# ── Aggregate by config ───────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["n_players"], r["n_imposters"], r["clue_rounds"], r["model"])
        groups[key].append(r)

    numeric_cols = [
        "non_imp_won", "imp_won",
        "accusation_accuracy", "non_imp_vote_accuracy",
        "imp_clue_obviousness", "non_imp_clue_obviousness", "obviousness_gap",
    ]
    agg = []
    for (n, k, r, model), grp in sorted(groups.items()):
        entry: dict = {
            "n_players": n, "n_imposters": k, "clue_rounds": r, "model": model,
            "n_games": len(grp),
        }
        for col in numeric_cols:
            vals = [row[col] for row in grp]
            entry[f"avg_{col}"] = round(sum(vals) / len(vals), 3)

        # guess rate — only games where imposter was caught and allowed a guess
        guess_rows = [row for row in grp if row["guess_rate"] != ""]
        if guess_rows:
            entry["avg_guess_rate"] = round(
                sum(float(row["guess_rate"]) for row in guess_rows) / len(guess_rows), 3
            )
        else:
            entry["avg_guess_rate"] = ""

        agg.append(entry)
    return agg


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print(f"No rows to save → {path}")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


# ── Main entry ────────────────────────────────────────────────────────────────

def analyze(results_dir: str = "imposter_game/results") -> list[dict]:
    """
    Load all game JSONs, compute metrics, write summary.csv + aggregated.csv.
    Returns per-game rows.
    """
    games = load_results(results_dir)
    if not games:
        print(f"No JSON files found in {results_dir}")
        return []

    print(f"Loaded {len(games)} game(s) from {results_dir}\n")
    rows = [game_metrics(g) for g in games]
    agg  = aggregate(rows)

    save_csv(rows, os.path.join(results_dir, "summary.csv"))
    save_csv(agg,  os.path.join(results_dir, "aggregated.csv"))

    # Console table
    print("\n=== AGGREGATE RESULTS ===")
    hdr = (
        f"{'n_p':>4} {'k':>3} {'rds':>4} | "
        f"{'crew_win':>8} {'acc_acc':>7} {'vote_acc':>8} "
        f"{'obs_gap':>7} {'guess_rt':>8} | {'games':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in agg:
        gr = f"{r['avg_guess_rate']:>8.3f}" if r["avg_guess_rate"] != "" else "     n/a"
        print(
            f"{r['n_players']:>4} {r['n_imposters']:>3} {r['clue_rounds']:>4} | "
            f"{r['avg_non_imp_won']:>8.2f} {r['avg_accusation_accuracy']:>7.3f} "
            f"{r['avg_non_imp_vote_accuracy']:>8.3f} "
            f"{r['avg_obviousness_gap']:>7.4f} {gr} | {r['n_games']:>5}"
        )
    return rows
