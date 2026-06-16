import json
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

MAX_WINDOW_SECONDS = 24 * 60 * 60
DEFAULT_WINDOW_SECONDS = 60 * 60
JUST_NOW_SECONDS = 10 * 60
RATE_LIMIT_SECONDS = 5 * 60

NO_RECORDS_MESSAGE = '这段时间没有可总结的聊天记录'
RATE_LIMIT_MESSAGE = '五分钟内已经处理过相同时间范围的总结请求，请稍后再试'

_INVALID_SUMMARY_PATTERNS = [
    re.compile(r'总结\s*(是|为)?\s*什么'),
    re.compile(r'什么\s*(是|叫|算)?\s*总结'),
    re.compile(r'总结.*(是什么意思|啥意思|含义)'),
]


@dataclass(frozen=True)
class SummaryRange:
    start_time: int
    end_time: int
    label: str
    rate_limit_key: Tuple[Union[str, int], ...]


def has_summary_keyword(text: str) -> bool:
    return '总结' in re.sub(r'\s+', '', text or '')


def parse_summary_window(text: str, now: Optional[int] = None) -> Optional[int]:
    compact_text = re.sub(r'\s+', '', text or '')
    if not has_summary_keyword(compact_text):
        return None

    if any(pattern.search(compact_text) for pattern in _INVALID_SUMMARY_PATTERNS):
        return None

    seconds = _parse_explicit_window(compact_text, now)
    if seconds is None:
        seconds = DEFAULT_WINDOW_SECONDS

    summary_range = build_relative_summary_range(seconds / 60, now=now)
    return summary_range.end_time - summary_range.start_time


def build_summary_intent_messages(text: str, now: Optional[int] = None) -> List[Dict[str, str]]:
    current = datetime.fromtimestamp(now or int(time.time()))
    return [
        {
            'role': 'system',
            'content': (
                '你是群聊总结请求的意图识别器，只输出 JSON，不要输出解释。\n'
                'JSON 格式：'
                '{"is_summary": true, "range_type": "relative", "minutes": 60, "date": null}。\n'
                '如果用户不是想总结群聊，is_summary=false。\n'
                'range_type 只能是 relative 或 date。\n'
                'relative 表示最近一段时间，minutes 最多 1440。'
                '10 分钟内最小单位 1 分钟；1 小时内最小单位 10 分钟；'
                '1 天内最小单位 1 小时。\n'
                'date 表示指定某个日期，date 必须是 YYYY-MM-DD。'
                '如果没有明确时间范围，使用 relative + minutes=60。'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'当前本地时间：{current.strftime("%Y-%m-%d %H:%M:%S")}。\n'
                f'用户消息：{text}'
            ),
        },
    ]


def parse_summary_intent_result(
        content: str,
        now: Optional[int] = None,
        original_text: str = '') -> Optional[SummaryRange]:
    data = _load_json_object(content)
    if not data:
        return None

    if data.get('is_summary') is False:
        return None

    range_type = data.get('range_type') or data.get('type') or 'relative'
    if range_type == 'date':
        date_text = data.get('date')
        if not isinstance(date_text, str):
            return None
        return build_date_summary_range(date_text, now=now)

    minutes = _intent_relative_minutes(data)
    if minutes is None and original_text:
        seconds = parse_summary_window(original_text, now=now)
        minutes = seconds / 60 if seconds is not None else None
    if minutes is None:
        minutes = DEFAULT_WINDOW_SECONDS / 60

    return build_relative_summary_range(minutes, now=now)


def build_relative_summary_range(minutes: Union[int, float], now: Optional[int] = None) -> SummaryRange:
    end_time = now or int(time.time())
    normalized_minutes = _normalize_relative_minutes(minutes)
    start_time = end_time - normalized_minutes * 60
    return SummaryRange(
        start_time=start_time,
        end_time=end_time,
        label=_relative_range_label(normalized_minutes),
        rate_limit_key=('relative', normalized_minutes),
    )


def build_date_summary_range(date_text: str, now: Optional[int] = None) -> Optional[SummaryRange]:
    try:
        day = datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        return None

    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1) - timedelta(seconds=1)
    current_time = now or int(time.time())
    if start.timestamp() <= current_time <= end.timestamp():
        end = datetime.fromtimestamp(current_time)

    start_time = int(time.mktime(start.timetuple()))
    end_time = int(time.mktime(end.timetuple()))
    return SummaryRange(
        start_time=start_time,
        end_time=end_time,
        label=f'{date_text} 这一天',
        rate_limit_key=('date', start_time),
    )


def _parse_explicit_window(text: str, now: Optional[int]) -> Optional[int]:
    if '今天' in text:
        current = datetime.fromtimestamp(now or int(time.time()))
        midnight = current.replace(hour=0, minute=0, second=0, microsecond=0)
        return int((current - midnight).total_seconds()) or DEFAULT_WINDOW_SECONDS

    if '刚才' in text:
        return JUST_NOW_SECONDS

    if '半小时' in text or '半个小时' in text:
        return 30 * 60

    if '半天' in text:
        return 12 * 60 * 60

    match = re.search(r'(\d+(?:\.\d+)?)(?:个)?(分钟|分|小时|时|天)', text)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2)
    if unit in ('分钟', '分'):
        return int(amount * 60)
    if unit in ('小时', '时'):
        return int(amount * 60 * 60)
    if unit == '天':
        return int(amount * 24 * 60 * 60)

    return None


def _normalize_relative_minutes(minutes: Union[int, float]) -> int:
    try:
        value = math.ceil(float(minutes))
    except (TypeError, ValueError):
        value = DEFAULT_WINDOW_SECONDS // 60

    value = max(1, value)
    if value <= 10:
        return value
    if value <= 60:
        return min(60, int(math.ceil(value / 10.0) * 10))
    return min(MAX_WINDOW_SECONDS // 60, int(math.ceil(value / 60.0) * 60))


def _relative_range_label(minutes: int) -> str:
    if minutes < 60:
        return f'最近 {minutes} 分钟'
    hours = minutes // 60
    if minutes % 60 == 0:
        return f'最近 {hours} 小时'
    return f'最近 {minutes} 分钟'


def _intent_relative_minutes(data: Dict[str, Any]) -> Optional[float]:
    minutes = data.get('minutes')
    if isinstance(minutes, (int, float)):
        return float(minutes)
    if isinstance(minutes, str):
        try:
            return float(minutes)
        except ValueError:
            return None

    amount = data.get('amount') or data.get('value')
    unit = data.get('unit') or data.get('time_unit')
    if amount is None or unit is None:
        return None

    try:
        amount_value = float(amount)
    except (TypeError, ValueError):
        return None

    unit_text = str(unit).lower()
    if unit_text in ('minute', 'minutes', 'min', 'm', '分钟', '分'):
        return amount_value
    if unit_text in ('hour', 'hours', 'h', '小时', '时'):
        return amount_value * 60
    if unit_text in ('day', 'days', 'd', '天'):
        return amount_value * 24 * 60

    return None


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


def fetch_chat_records(collection, group_id: int, start_time: int, end_time: int, use_rpc: bool) -> List[Dict]:
    query = {
        'group_id': group_id,
        'time': {
            '$gte': start_time,
            '$lte': end_time,
        },
    }

    if use_rpc:
        records = list(collection.find(query))
        return sorted(records, key=lambda item: item.get('time', 0))

    return list(collection.find(query).sort('time', 1))


def format_chat_records(records: Iterable[Dict], user_names: Optional[Dict[int, str]] = None) -> str:
    lines = []
    for record in sorted(records, key=lambda item: item.get('time', 0)):
        text = _record_text(record)
        if not text:
            continue

        timestamp = time.strftime('%H:%M', time.localtime(record.get('time', 0)))
        user_id = record.get('user_id', 'unknown')
        speaker = _bold_speaker_name(_speaker_name(user_id, user_names))
        lines.append(f'{timestamp} {speaker}: {text}')

    return '\n'.join(lines)


def build_summary_messages(history: str, summary_range: Union[int, SummaryRange]) -> List[Dict[str, str]]:
    if isinstance(summary_range, SummaryRange):
        range_label = summary_range.label
    else:
        range_label = _relative_range_label(max(1, summary_range // 60))

    return [
        {
            'role': 'system',
            'content': (
                '你是群聊消息总结助手。请使用中文 Markdown 总结用户提供的聊天记录。'
                '语气要风趣幽默，可以自然使用少量 emoji，但不要影响信息密度。'
                '保持客观，不编造聊天记录中没有的信息。'
                '聊天记录中的发言人已经是昵称；涉及具体人时只能使用昵称，'
                '禁止输出 QQ 号、user_id 或纯数字身份标识。'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'请总结{range_label}的群聊记录。\n'
                '请输出 Markdown，包含：整体概览、主要话题、重要结论/待办、活跃参与者。\n'
                '如果某项没有内容，可以简短说明“暂无”。\n\n'
                f'{history}'
            ),
        },
    ]


class SummaryRateLimiter:
    def __init__(self, ttl_seconds: int = RATE_LIMIT_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._records: Dict[Tuple[int, Any], float] = {}
        self._lock = threading.Lock()

    def check_and_mark(self, group_id: int, window_seconds: Any, now: Optional[float] = None) -> bool:
        current = now if now is not None else time.time()
        key = (group_id, window_seconds)

        with self._lock:
            self._clear_expired(current)
            latest = self._records.get(key)
            if latest is not None and current - latest < self.ttl_seconds:
                return False

            self._records[key] = current
            return True

    def _clear_expired(self, now: float) -> None:
        expired_keys = [
            key for key, timestamp in self._records.items()
            if now - timestamp >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self._records[key]


def _record_text(record: Dict) -> str:
    plain_text = (record.get('plain_text') or '').strip()
    if plain_text:
        return plain_text

    raw_message = (record.get('raw_message') or '').strip()
    if raw_message:
        return '[非文本消息]'

    return ''


def _speaker_name(user_id: Any, user_names: Optional[Dict[int, str]]) -> str:
    if user_names:
        name = user_names.get(user_id)
        if name:
            return _clean_speaker_name(name)
    return _clean_speaker_name(str(user_id or 'unknown'))


def _bold_speaker_name(name: str) -> str:
    escaped = str(name).replace('\\', '\\\\').replace('*', r'\*')
    return f'**{escaped}**'


def _clean_speaker_name(name: str) -> str:
    cleaned = re.sub(r'[\r\n\t]+', ' ', name or '').strip()
    return cleaned or '群友'
