"""
Imposter word-clue game engine.

RULES
-----
- A secret word is revealed to all players EXCEPT the imposter(s).
- Each round every player gives ONE short clue (word or phrase):
    - Non-imposters: hint at the word without making it too obvious
      (the imposter is listening and will try to blend in).
    - Imposter(s): bluff — say something plausible without knowing the word.
- After `clue_rounds` rounds of clues, everyone votes on who they think
  the imposter is.
- The player with the most votes is "accused".
    - If the accused IS an imposter → non-imposters score a point.
    - After each accusation the imposter (if caught) may attempt to guess
      the secret word. A correct guess is an imposter comeback win.
- Win conditions (tracked separately):
    - CREWMATES WIN: all imposters are voted out before imposters = crewmates.
    - IMPOSTERS WIN: imposter survives the vote OR guesses the word correctly.

Tweakable parameters
--------------------
  n_players      : total number of players
  n_imposters    : how many are secretly imposters
  clue_rounds    : how many rounds of clues before a vote
  allow_guess    : whether a caught imposter gets a word-guess attempt
  word_category  : restrict the word bank to a category (None = any)
  model          : LLM model string
  temperature    : LLM sampling temperature
"""

import random
import json
from dataclasses import dataclass, field
from typing import Optional
from imposter_game.llm_client import chat
from imposter_game.words import pick_word


# ── Player ─────────────────────────────────────────────────────────────────────

@dataclass
class Player:
    name: str
    is_imposter: bool
    active: bool = True   # still in the game (not yet voted out)

    def role_label(self) -> str:
        return "IMPOSTER" if self.is_imposter else "NON-IMPOSTER"


# ── System prompts ─────────────────────────────────────────────────────────────

def _system_non_imposter(player: Player, secret_word: str, n_players: int, n_imposters: int) -> str:
    return (
        f"You are {player.name}, playing a social deduction word game.\n"
        f"The secret word is: \"{secret_word}\".\n"
        f"There are {n_players} players total. {n_imposters} of them are IMPOSTERS "
        f"who do NOT know the secret word.\n\n"
        "YOUR GOAL: Give clues each round that prove to other non-imposters that "
        "you know the secret word — but don't make your clues so obvious that the "
        "imposter can figure out the word and blend in.\n"
        "Balance: too vague = you look like the imposter; too obvious = imposter learns the word.\n"
        "At the end of the clue rounds, everyone votes on who they think the imposter is.\n"
        "Do NOT reveal you know the secret word explicitly. Do NOT say the word itself."
    )


def _system_imposter(player: Player, n_players: int, n_imposters: int) -> str:
    allies = []  # filled in Game.__init__
    return (
        f"You are {player.name}, playing a social deduction word game.\n"
        f"You are an IMPOSTER. You do NOT know the secret word.\n"
        f"There are {n_players} players total. You are one of {n_imposters} imposters.\n\n"
        "YOUR GOAL: Listen to the other players' clues and blend in. "
        "Give clues that sound plausible but are vague enough that you could be consistent "
        "with many different words. Try not to get voted out.\n"
        "If you think you have figured out the word from context, you can give a more "
        "specific clue — but be careful not to overcommit and expose yourself.\n"
        "Do NOT say 'I don't know the word' or give any hint that you are the imposter."
    )


# ── Turn prompts ───────────────────────────────────────────────────────────────

def _clue_prompt(
    player: Player,
    secret_word: Optional[str],
    clue_log: list[dict],   # [{round, name, clue}]
    active_players: list[Player],
    current_round: int,
) -> str:
    lines = [f"=== ROUND {current_round} — Your turn to give a clue ===\n"]
    lines.append(f"Players still in game: {', '.join(p.name for p in active_players)}")

    if clue_log:
        lines.append("\n--- Clues given so far ---")
        cur = None
        for entry in clue_log:
            if entry["round"] != cur:
                cur = entry["round"]
                lines.append(f"[Round {cur}]")
            lines.append(f"  {entry['name']}: \"{entry['clue']}\"")

    lines.append(f"\n--- Round {current_round} clues so far ---")
    current_clues = [e for e in clue_log if e["round"] == current_round]
    if current_clues:
        for entry in current_clues:
            lines.append(f"  {entry['name']}: \"{entry['clue']}\"")
    else:
        lines.append("  (You are first to speak this round.)")

    if secret_word:
        lines.append(f"\nRemember: the secret word is \"{secret_word}\".")

    lines.append(
        "\nGive your clue now. Respond with ONE SINGLE WORD only — no phrases, no punctuation, "
        "no explanation. Do not say the secret word itself. Just one word."
    )
    return "\n".join(lines)


def _vote_prompt(
    player: Player,
    clue_log: list[dict],
    active_players: list[Player],
    current_round: int,
) -> str:
    candidates = [p.name for p in active_players if p.name != player.name]
    lines = [f"=== VOTE PHASE (after Round {current_round}) ===\n"]
    lines.append(f"Players: {', '.join(p.name for p in active_players)}")
    lines.append("\n--- All clues given ---")
    cur = None
    for entry in clue_log:
        if entry["round"] != cur:
            cur = entry["round"]
            lines.append(f"[Round {cur}]")
        lines.append(f"  {entry['name']}: \"{entry['clue']}\"")

    lines.append(
        f"\nVote for who you think the imposter is. "
        f"Choose from: {', '.join(candidates)}.\n"
        "Respond with ONLY the player's exact name — nothing else."
    )
    return "\n".join(lines)


def _guess_prompt(secret_word: str) -> str:
    # This prompt is sent to the imposter after being voted out
    return (
        "You were voted out as the suspected imposter.\n"
        "You now have ONE chance to guess the secret word. "
        "If you guess correctly, you win!\n"
        "Respond with ONLY your guess — a single word or short phrase, nothing else."
    )


# ── Game ───────────────────────────────────────────────────────────────────────

class Game:
    def __init__(
        self,
        n_players: int = 5,
        n_imposters: int = 1,
        clue_rounds: int = 2,
        allow_guess: bool = True,
        word_category: Optional[str] = None,
        model: str = "GPT-5.4-Nano",
        temperature: float = 0.7,
        game_id: Optional[str] = None,
    ):
        assert 1 <= n_imposters < n_players, "Need at least 1 non-imposter and 1 imposter"
        assert clue_rounds >= 1

        self.n_players = n_players
        self.n_imposters = n_imposters
        self.clue_rounds = clue_rounds
        self.allow_guess = allow_guess
        self.word_category = word_category
        self.model = model
        self.temperature = temperature
        self.game_id = game_id or f"game_{random.randint(100000, 999999)}"

        # Pick secret word
        self.secret_word, self.word_category_actual = pick_word(word_category)

        # Assign players and roles
        names = [f"Player_{chr(65 + i)}" for i in range(n_players)]
        random.shuffle(names)
        imposter_set = set(names[:n_imposters])
        self.players: list[Player] = [
            Player(name=n, is_imposter=(n in imposter_set)) for n in names
        ]

        # Build per-player system prompts
        self._sys: dict[str, str] = {}
        for p in self.players:
            if p.is_imposter:
                self._sys[p.name] = _system_imposter(p, n_players, n_imposters)
            else:
                self._sys[p.name] = _system_non_imposter(
                    p, self.secret_word, n_players, n_imposters
                )

        # Logs
        self.clue_log: list[dict] = []
        self.vote_log: list[dict] = []
        self.accusation_log: list[dict] = []
        self.winner: Optional[str] = None
        self.imposter_guessed_word: Optional[bool] = None
        self.round_num = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _active(self) -> list[Player]:
        return [p for p in self.players if p.active]

    def _active_imposters(self) -> list[Player]:
        return [p for p in self._active() if p.is_imposter]

    def _active_non_imposters(self) -> list[Player]:
        return [p for p in self._active() if not p.is_imposter]

    def _check_win(self) -> Optional[str]:
        n_imp  = len(self._active_imposters())
        n_non  = len(self._active_non_imposters())
        if n_imp == 0:
            return "non-imposters"
        if n_imp >= n_non:
            return "imposters"
        return None

    def _resolve_vote(self, votes: dict[str, str]) -> str:
        tally: dict[str, int] = {}
        for target in votes.values():
            tally[target] = tally.get(target, 0) + 1
        max_v = max(tally.values())
        top = [name for name, cnt in tally.items() if cnt == max_v]
        return random.choice(top)

    def _parse_vote(self, raw: str, voter: Player) -> str:
        candidates = [p.name for p in self._active() if p.name != voter.name]
        for name in candidates:
            if name in raw:
                return name
        return random.choice(candidates)

    def _parse_guess(self, raw: str) -> str:
        return raw.strip().lower().split("\n")[0].strip("\"'.,!? ")

    # ── API calls ─────────────────────────────────────────────────────────────

    def _get_clue(self, player: Player) -> str:
        word = self.secret_word if not player.is_imposter else None
        usr = _clue_prompt(player, word, self.clue_log, self._active(), self.round_num)
        raw = chat(self._sys[player.name], usr, model=self.model, temperature=self.temperature)
        # Enforce single word — take only the first token, strip punctuation
        return raw.strip().split()[0].strip("\"'.,!?:;") if raw.strip() else "pass"

    def _get_vote(self, player: Player) -> str:
        usr = _vote_prompt(player, self.clue_log, self._active(), self.round_num)
        raw = chat(self._sys[player.name], usr, model=self.model, temperature=self.temperature)
        return self._parse_vote(raw, player)

    def _get_word_guess(self, imposter: Player) -> str:
        usr = _guess_prompt(self.secret_word)
        raw = chat(self._sys[imposter.name], usr, model=self.model, temperature=self.temperature)
        return self._parse_guess(raw)

    # ── main loop ─────────────────────────────────────────────────────────────

    def play(self, verbose: bool = True) -> dict:
        imp_names = [p.name for p in self.players if p.is_imposter]
        if verbose:
            print(f"\n{'='*55}")
            print(f"Game {self.game_id}")
            print(f"Secret word : \"{self.secret_word}\" ({self.word_category_actual})")
            print(f"Players     : {self.n_players}  |  Imposters: {self.n_imposters}")
            print(f"Imposter(s) : {imp_names}")
            print(f"Clue rounds : {self.clue_rounds}  |  Allow guess: {self.allow_guess}")
            print(f"{'='*55}")

        while True:
            self.round_num += 1
            active = self._active()

            if verbose:
                print(f"\n--- Clue round {self.round_num} ---")

            # ── Clue phase ────────────────────────────────────────────────────
            for player in active:
                clue = self._get_clue(player)
                entry = {
                    "round": self.round_num,
                    "name": player.name,
                    "is_imposter": player.is_imposter,
                    "clue": clue,
                }
                self.clue_log.append(entry)
                if verbose:
                    tag = "[IMP] " if player.is_imposter else "[NON] "
                    print(f"  {tag}{player.name}: \"{clue}\"")

            # Only vote after `clue_rounds` rounds of clues
            if self.round_num < self.clue_rounds:
                continue

            # ── Vote phase ────────────────────────────────────────────────────
            if verbose:
                print(f"\n--- Vote phase ---")

            votes: dict[str, str] = {}
            for player in active:
                target = self._get_vote(player)
                votes[player.name] = target
                self.vote_log.append({
                    "round": self.round_num,
                    "voter": player.name,
                    "voter_is_imposter": player.is_imposter,
                    "target": target,
                    "target_is_imposter": next(
                        p.is_imposter for p in self.players if p.name == target
                    ),
                })
                if verbose:
                    tag = "[IMP] " if player.is_imposter else "[NON] "
                    print(f"  {tag}{player.name} votes → {target}")

            accused_name = self._resolve_vote(votes)
            accused = next(p for p in self.players if p.name == accused_name)
            accused.active = False

            acc_entry = {
                "round": self.round_num,
                "accused": accused_name,
                "role": accused.role_label(),
                "was_imposter": accused.is_imposter,
                "votes_received": sum(1 for t in votes.values() if t == accused_name),
                "total_votes": len(votes),
                "imposter_word_guess": None,
                "guess_correct": None,
            }

            if verbose:
                print(f"\n  >> ACCUSED: {accused_name} ({accused.role_label()})")

            # ── Imposter word-guess attempt ───────────────────────────────────
            if accused.is_imposter and self.allow_guess:
                guess = self._get_word_guess(accused)
                correct = guess == self.secret_word.lower()
                acc_entry["imposter_word_guess"] = guess
                acc_entry["guess_correct"] = correct
                self.imposter_guessed_word = correct
                if verbose:
                    result = "CORRECT ✓" if correct else f"WRONG (was \"{self.secret_word}\")"
                    print(f"  >> Imposter guesses: \"{guess}\" → {result}")
                if correct:
                    self.winner = "imposters"
                    self.accusation_log.append(acc_entry)
                    if verbose:
                        print("\n  *** IMPOSTERS WIN (correct word guess)! ***\n")
                    break

            self.accusation_log.append(acc_entry)

            # ── Win check ─────────────────────────────────────────────────────
            result = self._check_win()
            if result:
                self.winner = result
                if verbose:
                    print(f"\n  *** {result.upper()} WIN! ***\n")
                break

            # Still going — reset round counter for next clue block
            self.round_num = 0  # will be incremented to 1 at top of loop

        return self.to_dict()

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "config": {
                "n_players":      self.n_players,
                "n_imposters":    self.n_imposters,
                "clue_rounds":    self.clue_rounds,
                "allow_guess":    self.allow_guess,
                "word_category":  self.word_category,
                "model":          self.model,
                "temperature":    self.temperature,
            },
            "secret_word":    self.secret_word,
            "word_category":  self.word_category_actual,
            "players": [
                {"name": p.name, "role": p.role_label(), "survived": p.active}
                for p in self.players
            ],
            "winner":               self.winner,
            "imposter_guessed_word": self.imposter_guessed_word,
            "clue_log":         self.clue_log,
            "vote_log":         self.vote_log,
            "accusation_log":   self.accusation_log,
        }
