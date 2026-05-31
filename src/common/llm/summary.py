import re
import threading
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

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


def parse_summary_window(text: str, now: Optional[int] = None) -> Optional[int]:
    compact_text = re.sub(r'\s+', '', text or '')
    if '总结' not in compact_text:
        return None

    if any(pattern.search(compact_text) for pattern in _INVALID_SUMMARY_PATTERNS):
        return None

    seconds = _parse_explicit_window(compact_text, now)
    if seconds is None:
        seconds = DEFAULT_WINDOW_SECONDS

    return min(max(1, seconds), MAX_WINDOW_SECONDS)


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


def format_chat_records(records: Iterable[Dict]) -> str:
    lines = []
    for record in sorted(records, key=lambda item: item.get('time', 0)):
        text = _record_text(record)
        if not text:
            continue

        timestamp = time.strftime('%H:%M', time.localtime(record.get('time', 0)))
        user_id = record.get('user_id', 'unknown')
        lines.append(f'{timestamp} {user_id}: {text}')

    return '\n'.join(lines)


def build_summary_messages(history: str, window_seconds: int) -> List[Dict[str, str]]:
    minutes = max(1, window_seconds // 60)
    return [
        {
            'role': 'system',
            'content': (
                '你是群聊消息总结助手。请使用中文总结用户提供的聊天记录，'
                '保持客观，不编造聊天记录中没有的信息。'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'请总结最近 {minutes} 分钟内的群聊记录。\n'
                '输出包含：整体概览、主要话题、重要结论/待办、活跃参与者。\n\n'
                f'{history}'
            ),
        },
    ]


class SummaryRateLimiter:
    def __init__(self, ttl_seconds: int = RATE_LIMIT_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._records: Dict[Tuple[int, int], float] = {}
        self._lock = threading.Lock()

    def check_and_mark(self, group_id: int, window_seconds: int, now: Optional[float] = None) -> bool:
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
    if record.get('is_plain_text'):
        return plain_text

    raw_message = (record.get('raw_message') or '').strip()
    if raw_message:
        return '[非文本消息]'

    return ''
