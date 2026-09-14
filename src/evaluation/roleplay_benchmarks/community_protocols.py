"""Explicit public-data protocols: fixed RMT user scripts and dynamic MiniMax users."""
import json
import math
import random
import re

from .sources import read_json

RMT_FIELDS = ("character_understanding", "character_maintenance", "implicit_motivations", "user_preference")
MINI_DIMS = ("basics", "logic", "knowledge", "diversity", "content_logic", "interaction")


def rmt_turns(row, field, block_index):
    block = row[field][block_index]
    chat = block["chat"] if field == "user_preference" else block
    if not chat or not all(isinstance(x, str) and x.strip() for x in chat):
        raise ValueError("Invalid RMT user block")
    turns = [{"user": text, "field": field, "block_index": block_index, "block_turn": i}
             for i, text in enumerate(chat)]
    if field == "user_preference":
        questions, answers = block["question"], block["answer"]
        questions = [questions] if isinstance(questions, str) else questions
        answers = [answers] if isinstance(answers, str) else answers
        if not questions or len(questions) != len(answers) or len(answers) > len(turns):
            raise ValueError("UPA question/answer alignment invalid")
        # Paper-version final two turns / updated-version final one turn; retain
        # both the abstract reference question and actual user wording for audit.
        for turn, question, answer in zip(turns[-len(answers):], questions, answers):
            turn.update(reference_question=question, reference_answer=answer)
    return turns


def build_cases(config, base):
    cases = []
    for lang in config["languages"]:
        if config["benchmark"] == "rmtbench":
            prefix = "rmtbench_claude" if config["version"] == "claude35" else "rmtbench_claude45"
            for row_index, row in enumerate(read_json(base / f"{prefix}_{lang}.json")):
                if config["layout"] == "single_blocks":
                    layouts = [[(field, i)] for field in RMT_FIELDS for i in range(len(row[field]))]
                else:
                    layouts = []
                    for repeat in range(config["repeats"]):
                        rng = random.Random(f"{config['seed']}:{row_index}:{repeat}")
                        blocks = [(field, rng.randrange(len(row[field]))) for field in RMT_FIELDS]
                        rng.shuffle(blocks)
                        layouts.append(blocks)
                for index, layout in enumerate(layouts):
                    turns = [turn for field, bi in layout for turn in rmt_turns(row, field, bi)]
                    cases.append({"id": f"{lang}-{row_index:03}-{index:03}", "language": lang,
                                  "source_id": row["idx"], "name": row["name"], "profile": row["description"],
                                  "layout": layout, "turns": turns})
        else:
            rows = [json.loads(line) for line in (base / f"data/{lang}/seeds.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            for row_index, scene in enumerate(rows):
                for key in ("ai_name", "ai_setting", "user_name", "user_setting", "ai_prologue", "initial_user_input"):
                    if not isinstance(scene.get(key), str):
                        raise ValueError(f"Invalid MiniMax scene field: {key}")
                for repeat in range(config["repeats"]):
                    cases.append({"id": f"{lang}-{row_index:03}-{repeat:03}", "language": lang,
                                  "source_id": scene["id"], "repeat": repeat, "scene": scene})
    if config.get("limit_cases") is not None:
        # Deterministic shuffled subset, rather than taking only first language.
        random.Random(config["seed"]).shuffle(cases)
        cases = cases[:config["limit_cases"]]
    return cases


def parse_rmt(text, dimension):
    # A complete bold wrapper around the final score line is presentation,
    # not a different rating. Do not strip arbitrary trailing prose/markers.
    lines = text.rstrip().splitlines()
    if lines:
        last = lines[-1].strip()
        if last.startswith("**") and last.endswith("**") and last.count("**") == 2:
            lines[-1] = last[2:-2]
            text = "\n".join(lines)
    match = re.search(r"\[\[(Yes|No|[1-5])\]\]\s*[.!。]?\s*$", text, re.I)
    if not match:
        raise ValueError("Missing final [[rating]]")
    value = match.group(1).lower()
    if dimension in ("CM", "UPA"):
        if value not in ("yes", "no"):
            raise ValueError("Binary rating expected")
        return 1 if value == ("no" if dimension == "CM" else "yes") else 0
    if value not in "12345" or len(value) != 1:
        raise ValueError("Integer 1-5 expected")
    return int(value)


def parse_mini(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    result = json.loads(text, object_pairs_hook=unique)
    if not isinstance(result, dict) or set(result) != set(MINI_DIMS):
        raise ValueError("Expected six MiniMax dimensions")
    for item in result.values():
        if not isinstance(item, dict) or set(item) != {"score", "reason"}:
            raise ValueError("Expected score/reason")
        score = item["score"]
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 100 or not isinstance(item["reason"], str):
            raise ValueError("Invalid score/reason")
    return result


def actor_system(case, prompts):
    language = "English" if case["language"] == "en" else "Chinese"
    if "scene" in case:
        return prompts["actor"].format(language=language, **case["scene"])
    return prompts["actor"].format(language=language, name=case["name"], profile=case["profile"])


def expected_judgments(case, config):
    if "scene" in case:
        return math.ceil(config["generated_messages"] / config["chunk_messages"])
    return sum(4 + (turn["field"] == "character_maintenance") + ("reference_answer" in turn) for turn in case["turns"])


def rmt_judge_jobs(case, trajectory, prompts):
    for i, turn in enumerate(case["turns"]):
        response = trajectory[2 * i + 1]["content"]
        params = {"profile": case["profile"], "query": turn["user"], "response": response}
        for dimension in ("EE", "EC", "PA", "CU"):
            prompt = prompts["quality"].format(**params, dimension1=prompts["dimension_names"][dimension], dimension=prompts["dimensions"][dimension])
            yield f"turn-{i:04}-{dimension}", dimension, i, prompt
        if turn["field"] == "character_maintenance":
            yield f"turn-{i:04}-CM", "CM", i, prompts["CM"].format(**params)
        if "reference_answer" in turn:
            yield f"turn-{i:04}-UPA", "UPA", i, prompts["UPA"].format(**params, answer=turn["reference_answer"])


def mini_chunks(trajectory, size):
    # First two messages are source opening + initial user input, not scored.
    for start in range(2, len(trajectory), size):
        chunk = trajectory[start:start + size]
        if any(message["role"] == "assistant" for message in chunk):
            yield start, trajectory[:start], chunk
