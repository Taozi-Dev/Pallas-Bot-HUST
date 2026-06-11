import unittest

from src.common.llm.summary import (
    MAX_WINDOW_SECONDS,
    SummaryRateLimiter,
    format_chat_records,
    parse_summary_window,
)


class TestSummaryIntentParser(unittest.TestCase):
    def test_default_window(self):
        self.assertEqual(60 * 60, parse_summary_window('总结一下'))

    def test_minutes_window(self):
        self.assertEqual(30 * 60, parse_summary_window('帮我总结最近30分钟内的群聊'))

    def test_hours_window(self):
        self.assertEqual(3 * 60 * 60, parse_summary_window('总结3小时'))

    def test_caps_at_24_hours(self):
        self.assertEqual(MAX_WINDOW_SECONDS, parse_summary_window('总结48小时'))

    def test_rejects_non_summary_intent(self):
        self.assertIsNone(parse_summary_window('总结是什么'))
        self.assertIsNone(parse_summary_window('今天吃什么'))


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

        self.assertLess(result.index('1: first'), result.index('2: second'))

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
