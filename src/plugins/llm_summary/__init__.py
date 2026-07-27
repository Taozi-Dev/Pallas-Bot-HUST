import time

from asyncer import asyncify
from nonebot import logger, on_message
from nonebot.adapters import Bot
from nonebot.adapters.onebot.v11 import GroupMessageEvent, MessageSegment, permission
from nonebot.rule import Rule, to_me
from nonebot.typing import T_State

from src.common.config import plugin_config
from src.common.llm import LLMClient, LLMConfigError, LLMError
from src.common.llm.skill import (
    MIN_SKILL_RECORDS,
    NO_SKILL_TARGET_MESSAGE,
    SKILL_INTENT_FAILED_MESSAGE,
    SKILL_RECORDS_NOT_ENOUGH_MESSAGE,
    build_skill_generation_messages,
    build_skill_intent_messages,
    fetch_user_skill_records,
    format_skill_records,
    has_skill_generation_keywords,
    resolve_skill_target,
)
from src.common.utils.markdown_image import render_markdown_to_png
from src.plugins.repeater.model import Chat

from src.common.llm.summary import (
    NO_RECORDS_MESSAGE,
    RATE_LIMIT_MESSAGE,
    SummaryRateLimiter,
    build_summary_intent_messages,
    build_summary_messages,
    fetch_chat_records,
    format_chat_records,
    has_summary_keyword,
    parse_summary_intent_result,
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
    text = event.get_plaintext()
    if not has_summary_keyword(text):
        return False

    state['summary_request_text'] = text
    return True


summary_msg = on_message(
    rule=to_me() & Rule(is_summary_request),
    priority=12,
    block=True,
    permission=permission.GROUP,
)


async def is_skill_request(bot: Bot, event: GroupMessageEvent, state: T_State) -> bool:
    text = event.get_plaintext()
    if not has_skill_generation_keywords(text):
        return False

    state['skill_request_text'] = text
    return True


skill_msg = on_message(
    rule=to_me() & Rule(is_skill_request),
    priority=12,
    block=True,
    permission=permission.GROUP,
)


@summary_msg.handle()
async def _(bot: Bot, event: GroupMessageEvent, state: T_State):
    request_text = state['summary_request_text']
    try:
        client = LLMClient.from_config(plugin_config)
        intent_result = await asyncify(client.chat)(
            build_summary_intent_messages(request_text, now=event.time),
            temperature=0,
        )
        summary_range = parse_summary_intent_result(
            intent_result,
            now=event.time,
            original_text=request_text,
        )
    except LLMConfigError as error:
        logger.warning(f'LLM summary config error: {error}')
        await summary_msg.finish(f'LLM 配置不完整：{error}')
    except LLMError as error:
        logger.warning(f'LLM summary intent failed: {error}')
        await summary_msg.finish(f'总结意图识别失败：{error}')

    if summary_range is None:
        await summary_msg.finish('我听见“总结”了，但没判断出要总结哪段群聊。可以试试“@我 总结最近30分钟”。')

    if not _rate_limiter.check_and_mark(event.group_id, summary_range.rate_limit_key):
        await summary_msg.finish(RATE_LIMIT_MESSAGE)

    end_time = summary_range.end_time or event.time or int(time.time())
    start_time = summary_range.start_time

    await asyncify(Chat.sync)()

    collection = _message_collection()
    records = await asyncify(fetch_chat_records)(
        collection,
        event.group_id,
        start_time,
        end_time,
        plugin_config.use_rpc,
    )
    user_names = await _group_member_names(bot, event.group_id, records)
    history = format_chat_records(records, user_names)
    if not history:
        await summary_msg.finish(NO_RECORDS_MESSAGE)

    try:
        summary = await asyncify(client.chat)(
            build_summary_messages(history, summary_range),
            temperature=0.7,
        )
    except LLMError as error:
        logger.warning(f'LLM summary failed: {error}')
        await summary_msg.finish(f'总结失败：{error}')

    try:
        image = await asyncify(render_markdown_to_png)(summary)
    except Exception as error:
        logger.warning(f'Markdown summary render failed: {error}')
        await summary_msg.finish(summary)

    await summary_msg.finish(MessageSegment.image(file=image))


@skill_msg.handle()
async def _(bot: Bot, event: GroupMessageEvent, state: T_State):
    request_text = state['skill_request_text']
    candidates = await _skill_target_candidates(bot, event)
    if not candidates:
        await skill_msg.finish(NO_SKILL_TARGET_MESSAGE)

    try:
        client = LLMClient.from_config(plugin_config)
    except LLMConfigError as error:
        logger.warning(f'LLM skill config error: {error}')
        await skill_msg.finish(f'LLM 配置不完整：{error}')

    candidate_user_ids = [candidate['user_id'] for candidate in candidates]
    intent_result = ''
    if len(candidate_user_ids) > 1:
        try:
            intent_result = await asyncify(client.chat)(
                build_skill_intent_messages(request_text, candidates),
                temperature=0,
            )
        except LLMError as error:
            logger.warning(f'LLM skill intent failed: {error}')
            await skill_msg.finish(f'skill 意图识别失败：{error}')

    target_user_id = resolve_skill_target(intent_result, candidate_user_ids)
    if target_user_id is None:
        logger.warning(
            f'LLM skill target unresolved: response={intent_result!r}, '
            f'candidates={candidate_user_ids!r}')
        await skill_msg.finish(SKILL_INTENT_FAILED_MESSAGE)

    target_name = _candidate_name(candidates, target_user_id)

    await asyncify(Chat.sync)()

    collection = _message_collection()
    records = await asyncify(fetch_user_skill_records)(
        collection,
        event.group_id,
        target_user_id,
        plugin_config.use_rpc,
    )
    if len(records) < MIN_SKILL_RECORDS:
        await skill_msg.finish(SKILL_RECORDS_NOT_ENOUGH_MESSAGE)

    history = format_skill_records(records)
    if not history:
        await skill_msg.finish(SKILL_RECORDS_NOT_ENOUGH_MESSAGE)

    try:
        skill_markdown = await asyncify(client.chat)(
            build_skill_generation_messages(target_name, history),
            temperature=0.8,
        )
    except LLMError as error:
        logger.warning(f'LLM skill generation failed: {error}')
        await skill_msg.finish(f'skill 生成失败：{error}')

    try:
        image = await asyncify(render_markdown_to_png)(skill_markdown)
    except Exception as error:
        logger.warning(f'Markdown skill render failed: {error}')
        await skill_msg.finish(skill_markdown)

    await skill_msg.finish(MessageSegment.image(file=image))


async def _group_member_names(bot: Bot, group_id: int, records):
    user_ids = sorted({
        record.get('user_id') for record in records
        if record.get('user_id') is not None
    })
    names = {}
    fallback_index = 1
    for user_id in user_ids:
        name = ''
        try:
            info = await bot.call_api(
                'get_group_member_info',
                group_id=group_id,
                user_id=user_id,
                no_cache=False,
            )
            name = (info.get('card') or info.get('nickname') or '').strip()
        except Exception as error:
            logger.warning(f'Failed to fetch group member info: {error}')

        if not name:
            name = f'群友{fallback_index}'
            fallback_index += 1
        names[user_id] = name

    return names


async def _skill_target_candidates(bot: Bot, event: GroupMessageEvent):
    candidate_sources = {}
    for seg in event.message:
        if seg.type != 'at':
            continue
        user_id = _safe_int(seg.data.get('qq'))
        if user_id is None or user_id == event.self_id:
            continue
        candidate_sources.setdefault(user_id, 'mention')

    candidates = []
    for user_id, source in candidate_sources.items():
        candidates.append({
            'user_id': user_id,
            'name': await _group_member_name(bot, event.group_id, user_id),
            'source': source,
        })

    return candidates


async def _group_member_name(bot: Bot, group_id: int, user_id: int) -> str:
    try:
        info = await bot.call_api(
            'get_group_member_info',
            group_id=group_id,
            user_id=user_id,
            no_cache=False,
        )
        name = (info.get('card') or info.get('nickname') or '').strip()
    except Exception as error:
        logger.warning(f'Failed to fetch group member info: {error}')
        name = ''

    return name or f'群友{user_id}'


def _candidate_name(candidates, target_user_id: int) -> str:
    for candidate in candidates:
        if candidate['user_id'] == target_user_id:
            return candidate.get('name') or f'群友{target_user_id}'
    return f'群友{target_user_id}'


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
