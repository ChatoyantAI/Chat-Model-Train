"""Conversation adapter for the same frozen platform builder as single Judge/Battle."""

from copy import deepcopy
import hashlib
import json

try:
    from ..roleplay_judge_dataset import context_builder as platform
except ImportError:  # Standalone deployment places both packages in app/.
    from roleplay_judge_dataset import context_builder as platform

from .llm_judge import _platform_output_requirements

CONTEXT_VERSION = "platform-assembly-v1"
ROLE_FIELDS = (
    "name",
    "intro",
    "description",
    "gender",
    "age",
    "personality",
    "scenario",
    "qa_examples",
    "example_reply",
)


def clean_role_profile(profile):
    return {k: deepcopy(profile[k]) for k in ROLE_FIELDS if k in profile}


def assemble_platform_context(packet, runtime):
    """Use factual fields + shared templates; never interpolate stored DB prompt."""
    packet = deepcopy(packet)
    packet["role_profile"] = clean_role_profile(packet.get("role_profile", {}))
    persona = packet.get("persona")
    if persona and isinstance(persona.get("pinned_memories"), str):
        try:
            persona["pinned_memories"] = json.loads(persona["pinned_memories"])
        except (ValueError, TypeError):
            persona["pinned_memories"] = (
                [persona["pinned_memories"]] if persona["pinned_memories"] else []
            )
    settings = {
        k: runtime.get(k)
        for k in (
            "chat_mod",
            "user_gender",
            "third_person_perspective",
            "is_test_account",
            "prompt_variant",
        )
    }
    # Defaults match normal platform settings, and are explicitly identified.
    defaulted = [
        k
        for k in ("chat_mod", "third_person_perspective", "is_test_account")
        if runtime.get(k) is None
    ]
    maximum = int(runtime.get("chat_mod") or 250)
    character = {
        **packet["role_profile"],
        "greeting": packet["greeting"].get(
            "source_content", packet["greeting"]["content"]
        ),
        "third_person_perspective": bool(runtime.get("third_person_perspective")),
        "card_prompt_description": runtime.get("card_prompt_description"),
        "card_lorebook": runtime.get("card_lorebook"),
    }
    if isinstance(character["card_lorebook"], str):
        character["card_lorebook"] = json.loads(character["card_lorebook"])
    systems, prior, current_query = [], [], ""
    first_language = None
    for message in packet["messages"]:
        if message["role"] == "user":
            current_query = message["content"]
        if message["role"] == "assistant" and message["scope"] == "candidate":
            requirements = message.get("platform_output_requirements") or {}
            language = requirements.get(
                "response_language"
            ) or platform._output_language_name(
                message.get("generation_language") or packet["language"]
            )
            first_language = first_language or language
            content, template = platform.build_platform_system_prompt(
                character,
                persona,
                query=current_query,
                message_list=prior[:-1]
                if prior and prior[-1]["role"] == "user"
                else prior,
                language=language,
                max_tokens=maximum,
                user_gender=runtime.get("user_gender"),
                is_test_account=bool(runtime.get("is_test_account")),
                prompt_variant=runtime.get("prompt_variant"),
            )
            entry = next(
                (
                    s
                    for s in systems
                    if s["content"] == content and s["response_language"] == language
                ),
                None,
            )
            if entry is None:
                entry = {
                    "id": f"S{len(systems) + 1}",
                    "role": "system",
                    "content": content,
                    "template": template,
                    "response_language": language,
                    "applies_to": [],
                }
                systems.append(entry)
            entry["applies_to"].append(message["id"])
            mode = platform.select_mode_by_tokens(maximum)
            generated = (
                _platform_output_requirements(
                    [platform._length_guideline(mode, maximum, language)]
                )
                or {}
            )
            generated.update(requirements)
            generated["response_language"] = language
            generated["source"] = (
                "saved_guideline_or_dataset_language_and_current_platform_settings"
            )
            generated["description"] = (
                "这是平台输出要求，不是用户说的话。目标语言以本字段为准，不按用户正文语言推断。"
                "长度仅供了解生成模式，不在任何评分项检查字数或 Token 是否达标。模式参数为当前配置快照，不是历史逐轮记录。"
            )
            message["platform_output_requirements"] = generated
        prior.append({"role": message["role"], "content": message["content"]})
    raw_greeting = character["greeting"]
    if raw_greeting:
        formatted = platform._format_character_prompt(
            raw_greeting,
            character=character,
            persona=persona,
            query="",
            message_list=[],
            language=first_language
            or platform._output_language_name(packet["language"]),
            max_tokens=maximum,
            mode=platform.select_mode_by_tokens(maximum),
            user_gender=runtime.get("user_gender"),
        )
        packet["greeting"]["source_content"] = raw_greeting
        packet["greeting"]["content"] = platform._remove_markdown_links(formatted)
    packet["system_contexts"] = systems
    packet["platform_assembly"] = {
        "version": CONTEXT_VERSION,
        "production_code_commit": platform.PRODUCTION_CODE_COMMIT,
        "settings": settings,
        "effective_chat_mod": maximum,
        "defaulted_settings": defaulted,
        "settings_source": "current_readonly_snapshot",
        "historical_request_replay": False,
        "excluded_fields": ["character_descriptions.prompt", "characters.prompt"],
    }
    basis = packet["context_basis"]
    basis["assembly_version"] = CONTEXT_VERSION
    basis["notes"] = [
        n
        for n in basis.get("notes", [])
        if "平台要求只从" not in n and "Greeting" not in n and "开场" not in n
    ]
    basis["notes"].extend(
        [
            "System 使用与单条 Judge / Battle 共用的平台模板、选择分支和格式化函数；不读取数据库的整段 prompt。",
            "目标语言来自候选数据或已保存的平台要求，不来自用户正文。聊天模式、视角及实验配置为当前只读快照，不能证明历史逐轮参数未变。",
            "Greeting 取角色资料原文，仅按平台规则替换角色/玩家占位符并移除图片链接；未生成或续写。页面可查看角色开场原文。",
        ]
    )
    return packet


def context_fingerprint(packet):
    raw = {k: v for k, v in packet.items() if k not in ("seq", "context_hash")}
    return hashlib.sha256(
        json.dumps(
            raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
