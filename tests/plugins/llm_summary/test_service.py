import time
import unittest
from datetime import datetime

from src.common.llm.summary import (
    MAX_WINDOW_SECONDS,
    SummaryRateLimiter,
    format_chat_records,
    has_summary_keyword,
    parse_summary_intent_result,
    parse_summary_window,
)


class TestSummaryIntentParser(unittest.TestCase):
    def test_default_window(self):
        self.assertEqual(60 * 60, parse_summary_window('总结一下'))

    def test_detects_summary_keyword(self):
        self.assertTrue(has_summary_keyword(' 帮我 总结 一下'))
        self.assertFalse(has_summary_keyword('今天吃什么'))

    def test_minutes_window(self):
        self.assertEqual(30 * 60, parse_summary_window('帮我总结最近30分钟内的群聊'))

    def test_small_minutes_keep_minute_granularity(self):
        self.assertEqual(8 * 60, parse_summary_window('总结最近8分钟'))

    def test_hours_window(self):
        self.assertEqual(3 * 60 * 60, parse_summary_window('总结3小时'))

    def test_caps_at_24_hours(self):
        self.assertEqual(MAX_WINDOW_SECONDS, parse_summary_window('总结48小时'))

    def test_rejects_non_summary_intent(self):
        self.assertIsNone(parse_summary_window('总结是什么'))
        self.assertIsNone(parse_summary_window('今天吃什么'))

    def test_parses_llm_relative_range(self):
        result = parse_summary_intent_result(
            '{"is_summary": true, "range_type": "relative", "minutes": 45}',
            now=10000,
        )

        self.assertEqual('最近 50 分钟', result.label)
        self.assertEqual(10000, result.end_time)
        self.assertEqual(10000 - 50 * 60, result.start_time)

    def test_parses_llm_date_range(self):
        now = int(time.mktime(datetime(2026, 6, 13, 12, 0, 0).timetuple()))
        result = parse_summary_intent_result(
            '{"is_summary": true, "range_type": "date", "date": "2026-06-12"}',
            now=now,
        )
        expected_start = int(time.mktime(datetime(2026, 6, 12, 0, 0, 0).timetuple()))

        self.assertEqual('2026-06-12 这一天', result.label)
        self.assertEqual(expected_start, result.start_time)
        self.assertEqual(expected_start + 24 * 60 * 60 - 1, result.end_time)

    def test_rejects_llm_non_summary_intent(self):
        self.assertIsNone(parse_summary_intent_result('{"is_summary": false}', now=10000))


class TestFormatChatRecords(unittest.TestCase):
    def test_formats_sorted_records(self):
        records = [
            {
                'time': 200,
                'user_id': 2,
                'is_plain_text': True,
                'plain_text': 'second',
                'raw_message': 'second',
            },
            {
                'time': 100,
                'user_id': 1,
                'is_plain_text': True,
                'plain_text': 'first',
                'raw_message': 'first',
            },
        ]

        result = format_chat_records(records)

        self.assertLess(result.index('**1**: first'), result.index('**2**: second'))

    def test_formats_with_bold_nicknames(self):
        records = [
            {
                'time': 100,
                'user_id': 1,
                'is_plain_text': True,
                'plain_text': 'first',
                'raw_message': 'first',
            },
        ]

        result = format_chat_records(records, {1: '小明'})

        self.assertIn('**小明**: first', result)
        self.assertNotIn('1: first', result)

    def test_escapes_markdown_in_nickname(self):
        result = format_chat_records([{
            'time': 100,
            'user_id': 1,
            'is_plain_text': True,
            'plain_text': 'hello',
            'raw_message': 'hello',
        }], {1: 'a*b'})

        self.assertIn(r'**a\*b**: hello', result)

    def test_reply_message_keeps_plain_text(self):
        result = format_chat_records([{
            'time': 100,
            'user_id': 1,
            'is_plain_text': False,
            'plain_text': 'reply text',
            'raw_message': '[CQ:reply,id=123]reply text',
        }])

        self.assertIn('**1**: reply text', result)
        self.assertNotIn('[CQ:reply', result)

    def test_skips_empty_text(self):
        result = format_chat_records([{
            'time': 100,
            'user_id': 1,
            'is_plain_text': True,
            'plain_text': '  ',
            'raw_message': '',
        }])

        self.assertEqual('', result)

    def test_non_text_placeholder(self):
        result = format_chat_records([{
            'time': 100,
            'user_id': 1,
            'is_plain_text': False,
            'plain_text': '',
            'raw_message': '[CQ:image]',
        }])

        self.assertIn('[非文本消息]', result)


class TestSummaryRateLimiter(unittest.TestCase):
    def test_blocks_same_group_and_window_inside_ttl(self):
        limiter = SummaryRateLimiter(ttl_seconds=300)

        self.assertTrue(limiter.check_and_mark(group_id=1, window_seconds=3600, now=1000))
        self.assertFalse(limiter.check_and_mark(group_id=1, window_seconds=3600, now=1100))

    def test_allows_different_group_or_window(self):
        limiter = SummaryRateLimiter(ttl_seconds=300)

        self.assertTrue(limiter.check_and_mark(group_id=1, window_seconds=3600, now=1000))
        self.assertTrue(limiter.check_and_mark(group_id=2, window_seconds=3600, now=1100))
        self.assertTrue(limiter.check_and_mark(group_id=1, window_seconds=1800, now=1100))

    def test_allows_after_ttl(self):
        limiter = SummaryRateLimiter(ttl_seconds=300)

        self.assertTrue(limiter.check_and_mark(group_id=1, window_seconds=3600, now=1000))
        self.assertTrue(limiter.check_and_mark(group_id=1, window_seconds=3600, now=1300))


if __name__ == '__main__':
    unittest.main()
