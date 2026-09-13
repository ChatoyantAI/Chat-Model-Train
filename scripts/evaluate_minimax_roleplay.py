"""Run MiniMax public-seed self-play with versioned local simulator/judge prompts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.roleplay_benchmarks.entry import entry

if __name__ == "__main__":
    raise SystemExit(entry("minimax"))
