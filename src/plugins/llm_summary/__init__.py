import time

from asyncer import asyncify
from nonebot import logger, on_message
from nonebot.adapters import Bot
from nonebot.adapters.onebot.v11 import GroupMessageEvent, permission
from nonebot.rule import Rule, to_me
from nonebot.typing import T_State

from src.common.config import plugin_config
from src.common.llm import LLMClient, LLMConfigError, LLMError
from src.plugins.repeater.model import Chat

from src.common.llm.summary import (
    NO_RECORDS_MESSAGE,
    RATE_LIMIT_MESSAGE,
    SummaryRateLimiter,
    build_summary_messages,
    fetch_chat_records,
    format_chat_records,
    parse_summary_window,
)

if plugin_config.use_rpc:
    from src.common.utils.rpc import MongoClient
else:
    from pymongo import MongoClient

_rate_limiter = SummaryRateLimiter()


def _message_collection():
    mongo_client = MongoClient(
        plugin_config.mongo_host,
        plugin_config.mongo_port,
        unicode_decode_error_handler='ignore')
    return mongo_client['PallasBot']['message']


async def is_summary_request(bot: Bot, event: GroupMessageEvent, state: T_State) -> bool:
    window_seconds = parse_summary_window(event.get_plaintext(), now=event.time)
    if window_seconds is None:
        return False

    state['summary_window_seconds'] = window_seconds
    return True


summary_msg = on_message(
    rule=to_me() & Rule(is_summary_request),
    priority=12,
    block=True,
    permission=permission.GROUP,
)


@summary_msg.handle()
async def _(bot: Bot, event: GroupMessageEvent, state: T_State):
    window_seconds = state['summary_window_seconds']
    if not _rate_limiter.check_and_mark(event.group_id, window_seconds):
        await summary_msg.finish(RATE_LIMIT_MESSAGE)

    end_time = event.time or int(time.time())
    start_time = end_time - window_seconds

    await asyncify(Chat.sync)()

    collection = _message_collection()
    records = await asyncify(fetch_chat_records)(
        collection,
        event.group_id,
        start_time,
        end_time,
        plugin_config.use_rpc,
    )
    history = format_chat_records(records)
    if not history:
        await summary_msg.finish(NO_RECORDS_MESSAGE)

    try:
        client = LLMClient.from_config(plugin_config)
        summary = await asyncify(client.chat)(
            build_summary_messages(history, window_seconds))
    except LLMConfigError as error:
        logger.warning(f'LLM summary config error: {error}')
        await summary_msg.finish(f'LLM 配置不完整：{error}')
    except LLMError as error:
        logger.warning(f'LLM summary failed: {error}')
        await summary_msg.finish(f'总结失败：{error}')

    await summary_msg.finish(summary)
