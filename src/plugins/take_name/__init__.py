import random
from nonebot import require, logger, get_bot, on_notice
from nonebot.rule import Rule
from nonebot.typing import T_State
from nonebot.adapters import Bot, Event
from nonebot.exception import ActionFailed
from nonebot.adapters.onebot.v11 import Message, NoticeEvent

from src.plugins.repeater.model import Chat
from src.common.config import BotConfig
from src.common.utils import is_bot_admin

change_name_sched = require('nonebot_plugin_apscheduler').scheduler


@change_name_sched.scheduled_job('cron', minute='*/1')
async def change_name():
    rand_messages = Chat.get_random_message_from_each_group()
    if not rand_messages:
        return

    for group_id, target_msg in rand_messages.items():
        if random.random() > 0.002:  # 期望约每8个多小时改一次
            continue

        bot_id = target_msg['bot_id']
        config = BotConfig(bot_id, group_id)
        if config.is_sleep():
            continue

        target_user_id = target_msg['user_id']
        logger.info(
            'bot [{}] ready to change name by using [{}] in group [{}]'.format(
                bot_id, target_user_id, group_id))

        bot = get_bot(str(bot_id))
        if not bot:
            logger.error("no bot: " + str(bot_id))
            continue

        try:
            # 获取群友昵称
            info = await bot.call_api('get_group_member_info', **{
                'group_id': group_id,
                'user_id': target_user_id,
                'no_cache': True
            })
        except ActionFailed:
            # 可能这人退群了
            continue

        card = info['card'] if info['card'] else info['nickname']
        logger.info(
            'bot [{}] ready to change name to[{}] in group [{}]'.format(
                bot_id, card, group_id))
        try:
            # 改牛牛自己的群名片
            await bot.call_api('set_group_card', **{
                'group_id': group_id,
                'user_id': bot_id,
                'card': card
            })

            # 酒后夺舍！改群友的！
            if config.drunkenness() and await is_bot_admin(bot_id, group_id, True):
                await bot.call_api('set_group_card', **{
                    'group_id': group_id,
                    'user_id': target_user_id,
                    'card': random.choice(['帕拉斯', '牛牛', '牛牛喝酒', '牛牛干杯', '牛牛继续喝'])
                })

            # # 戳一戳
            # await bot.call_api('send_group_msg', **{
            #     'message': Message('[CQ:poke,qq={}]'.format(target_user_id)),
            #     'group_id': group_id
            # })

            config.update_taken_name(target_user_id)

        except ActionFailed:
            # 可能牛牛退群了
            continue


async def is_change_name_notice(bot: Bot, event: NoticeEvent, state: T_State) -> bool:
    config = BotConfig(event.self_id, event.group_id)
    if event.notice_type == 'group_card' and event.user_id == config.taken_name():
        return True
    return False


watch_name = on_notice(rule=Rule(is_change_name_notice), priority=4)


@watch_name.handle()
async def watch_name_handle(bot: Bot, event: NoticeEvent, state: T_State):
    group_id = event.group_id
    user_id = event.user_id
    bot_id = event.self_id

    try:
        info = await bot.call_api('get_group_member_info', **{
            'group_id': group_id,
            'user_id': user_id,
            'no_cache': True
        })
    except ActionFailed:
        return

    card = info['card'] if info['card'] else info['nickname']

    logger.info(
        'bot [{}] watch name change by [{}] in group [{}]'.format(
            bot.self_id, user_id, group_id))

    config = BotConfig(bot.self_id, group_id)

    try:
        await bot.call_api('set_group_card', **{
            'group_id': group_id,
            'user_id': bot_id,
            'card': card
        })

        config.update_taken_name(user_id)

    except ActionFailed:
        return
