import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional

MAX_SKILL_RECORDS = 1000
MIN_SKILL_RECORDS = 100

NO_SKILL_TARGET_MESSAGE = '请在消息里 @ 要生成 skill 的群友'
SKILL_RECORDS_NOT_ENOUGH_MESSAGE = '这位群友的聊天记录不足 100 条，暂时无法生成 skill'
SKILL_INTENT_FAILED_MESSAGE = '没有识别出要为哪位群友生成 skill'


def has_skill_generation_keywords(text: str) -> bool:
    compact_text = re.sub(r'\s+', '', text or '').lower()
    return '生成' in compact_text and 'skill' in compact_text


def build_skill_intent_messages(text: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    candidate_lines = [
        f'- user_id={candidate["user_id"]}, name={candidate.get("name") or ""}, source={candidate.get("source") or ""}'
        for candidate in candidates
    ]
    return [
        {
            'role': 'system',
            'content': (
                '你是群友 skill 生成请求的意图识别器，只输出 JSON，不要输出解释。\n'
                '如果用户想让机器人为某位群友生成 skill，则输出：'
                '{"is_skill": true, "target_user_id": 123}。\n'
                '候选用户只来自用户消息中显式 @ 的群友。'
                'target_user_id 必须从候选用户中选择；如果无法确认目标，target_user_id=null。\n'
                '如果用户不是这个意图，输出 {"is_skill": false, "target_user_id": null}。'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'用户消息：{text}\n'
                '候选用户：\n'
                + '\n'.join(candidate_lines)
            ),
        },
    ]


def parse_skill_intent_result(content: str, candidate_user_ids: Iterable[int]) -> Optional[int]:
    data = _load_json_object(content)
    if not data or data.get('is_skill') is False:
        return None

    candidate_set = {int(user_id) for user_id in candidate_user_ids}
    target_user_id = data.get('target_user_id') or data.get('user_id')
    try:
        target_user_id = int(target_user_id)
    except (TypeError, ValueError):
        return None

    return target_user_id if target_user_id in candidate_set else None


def resolve_skill_target(content: str, candidate_user_ids: Iterable[int]) -> Optional[int]:
    candidates = list(dict.fromkeys(int(user_id) for user_id in candidate_user_ids))
    if len(candidates) == 1:
        return candidates[0]
    return parse_skill_intent_result(content, candidates)


def fetch_user_skill_records(
        collection,
        group_id: int,
        user_id: int,
        use_rpc: bool,
        limit: int = MAX_SKILL_RECORDS) -> List[Dict]:
    query = {
        'group_id': group_id,
        'user_id': user_id,
    }

    if use_rpc:
        records = list(collection.find(query))
        records = sorted(records, key=lambda item: item.get('time', 0), reverse=True)[:limit]
    else:
        records = list(collection.find(query).sort('time', -1).limit(limit))

    return sorted(records, key=lambda item: item.get('time', 0))


def format_skill_records(records: Iterable[Dict]) -> str:
    lines = []
    for record in sorted(records, key=lambda item: item.get('time', 0)):
        text = (record.get('plain_text') or '').strip()
        if not text:
            continue
        timestamp = time.strftime('%Y-%m-%d %H:%M', time.localtime(record.get('time', 0)))
        lines.append(f'{timestamp}: {text}')

    return '\n'.join(lines)


def build_skill_generation_messages(target_name: str, history: str) -> List[Dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': (
                '你是群友 skill 生成器。请只根据提供的聊天记录分析，不要编造没有依据的事实。'
                '输出中文 Markdown，语气可以幽默，但要尊重被分析的人。'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'请基于下面的聊天记录，为群友「{target_name}」生成一份 .skill 风格的能力卡。\n'
                '请包含：名称、被动技能、主动技能、触发条件、常见口癖/话题、使用限制、简短评价。\n'
                '如果证据不足，请在对应条目里写“样本不足”。\n\n'
                f'{history}'
            ),
        },
    ]


def _load_json_object(content: str) -> Optional[Dict[str, Any]]:
    text = (content or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        data = json.loads(text)
    except ValueError:
        match = re.search(r'\{.*\}', text, flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return None

    return data if isinstance(data, dict) else None
