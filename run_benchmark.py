"""
Imposter Benchmark — main entry point.

THE GAME
--------
A secret word is shown to everyone except the imposter(s).
Players take turns giving short clues that signal they know the word —
without making it so obvious the imposter can piggyback.
After `clue_rounds` rounds of clues, everyone votes on who the imposter is.
If the imposter is caught, they get one chance to guess the word (comeback win).

QUICK START
-----------
1. Set your Poe API key in imposter_game/llm_client.py.
2. pip install openai

Run a single test game:
    python run_benchmark.py --mode single --n_players 5 --n_imposters 1 --clue_rounds 2

Run the full parameter sweep:
    python run_benchmark.py --mode sweep --games_per_config 3

Analyze existing results (no API calls):
    python run_benchmark.py --mode analyze

PARAMETERS
----------
  --n_players       Total players                          (default: 5)
  --n_imposters     How many don't know the word           (default: 1)
  --clue_rounds     Rounds of clues before a vote          (default: 2)
  --allow_guess     Caught imposter can guess the word     (default: True)
  --word_category   Restrict words: animal food place ...  (default: any)
  --model           Poe model string                       (default: GPT-5.4-Nano)
  --temperature     LLM temperature 0–1                    (default: 0.7)

SWEEP MODE
----------
Edit DEFAULT_SWEEP in imposter_game/simulation.py to change combinations.
Results saved as JSON to imposter_game/results/.

ANALYSIS MODE
-------------
Reads all JSONs in imposter_game/results/ and writes:
  - imposter_game/results/summary.csv      (one row per game)
  - imposter_game/results/aggregated.csv   (averaged per config)
"""

import argparse
from imposter_game.simulation import run_single, run_sweep
from imposter_game.analysis import analyze
from imposter_game.words import CATEGORIES

RESULTS_DIR = "imposter_game/results"


def main():
    parser = argparse.ArgumentParser(description="LLM Imposter Word-Clue Benchmark")
    parser.add_argument(
        "--mode", choices=["single", "sweep", "analyze"], default="single",
    )
    # Game params
    parser.add_argument("--n_players",       type=int,   default=5)
    parser.add_argument("--n_imposters",     type=int,   default=1)
    parser.add_argument("--clue_rounds",     type=int,   default=2)
    parser.add_argument("--allow_guess",     type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--word_category",   type=str,   default=None,
                        choices=[None] + CATEGORIES, metavar="CATEGORY")
    parser.add_argument("--model",           type=str,   default="GPT-5.4-Nano")
    parser.add_argument("--temperature",     type=float, default=0.7)
    # Sweep params
    parser.add_argument("--games_per_config", type=int,  default=3)
    parser.add_argument("--quiet",           action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    if args.mode == "single":
        run_single(
            n_players=args.n_players,
            n_imposters=args.n_imposters,
            clue_rounds=args.clue_rounds,
            allow_guess=args.allow_guess,
            word_category=args.word_category,
            model=args.model,
            temperature=args.temperature,
            output_dir=RESULTS_DIR,
            verbose=verbose,
        )

    elif args.mode == "sweep":
        run_sweep(
            games_per_config=args.games_per_config,
            output_dir=RESULTS_DIR,
            verbose=verbose,
        )

    elif args.mode == "analyze":
        analyze(RESULTS_DIR)


if __name__ == "__main__":
    main()
