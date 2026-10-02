# Imposter LLM Benchmark

A multi-agent "imposter" game used as a benchmark for large language models — several model
agents play a social-deduction game, and their performance (deception, detection, reasoning) is
scored.

## Layout
| File | Role |
|---|---|
| `game.py` | Game engine / rules |
| `llm_client.py` | Model interface |
| `simulation.py` | Runs matches |
| `analysis.py` | Scores and summarizes results |
| `words.py` | Word/role content |

## Setup
Set your key first (see `.env.example`), then:
```bash
python -m pip install openai
export DEEPSEEK_API_KEY="sk-..."
python simulation.py
```
