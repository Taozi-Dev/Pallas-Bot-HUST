import unittest

from src.common.llm.skill import (
    MAX_SKILL_RECORDS,
    build_skill_generation_messages,
    build_skill_intent_messages,
    fetch_user_skill_records,
    format_skill_records,
    has_skill_generation_keywords,
    parse_skill_intent_result,
    resolve_skill_target,
)


class FakeCursor:
    def __init__(self, records):
        self.records = list(records)

    def sort(self, key, direction):
        reverse = direction < 0
        self.records.sort(key=lambda item: item.get(key, 0), reverse=reverse)
        return self

    def limit(self, limit):
        self.records = self.records[:limit]
        return self

    def __iter__(self):
        return iter(self.records)


class FakeCollection:
    def __init__(self, records):
        self.records = records
        self.last_query = None

    def find(self, query):
        self.last_query = query
        filtered = [
            record for record in self.records
            if record.get('group_id') == query.get('group_id')
            and record.get('user_id') == query.get('user_id')
        ]
        return FakeCursor(filtered)


class TestSkillIntent(unittest.TestCase):
    def test_detects_skill_generation_keywords(self):
        self.assertTrue(has_skill_generation_keywords('帮我生成 @小明 的 skill'))
        self.assertTrue(has_skill_generation_keywords('生成Skill'))
        self.assertFalse(has_skill_generation_keywords('看看小明的 skill'))
        self.assertFalse(has_skill_generation_keywords('生成总结'))

    def test_builds_intent_prompt_with_candidates(self):
        messages = build_skill_intent_messages('生成 skill', [{
            'user_id': 1,
            'name': '小明',
            'source': 'mention',
        }])

        self.assertIn('target_user_id', messages[0]['content'])
        self.assertIn('user_id=1', messages[1]['content'])
        self.assertIn('小明', messages[1]['content'])

    def test_parses_target_from_candidate_set(self):
        result = parse_skill_intent_result(
            '{"is_skill": true, "target_user_id": 2}',
            candidate_user_ids=[1, 2],
        )

        self.assertEqual(2, result)

    def test_rejects_unknown_target(self):
        result = parse_skill_intent_result(
            '{"is_skill": true, "target_user_id": 3}',
            candidate_user_ids=[1, 2],
        )

        self.assertIsNone(result)

    def test_rejects_non_skill_intent(self):
        self.assertIsNone(parse_skill_intent_result('{"is_skill": false}', [1]))

    def test_resolves_single_candidate_without_llm_result(self):
        self.assertEqual(123, resolve_skill_target('', [123]))
        self.assertEqual(
            123,
            resolve_skill_target('{"is_skill": false, "target_user_id": null}', [123]),
        )

    def test_resolves_multiple_candidates_from_llm_result(self):
        result = resolve_skill_target(
            '{"is_skill": true, "target_user_id": 2}',
            [1, 2],
        )

        self.assertEqual(2, result)


class TestSkillRecords(unittest.TestCase):
    def test_fetches_latest_records_in_chronological_order(self):
        records = [
            {'group_id': 1, 'user_id': 2, 'time': index}
            for index in range(MAX_SKILL_RECORDS + 5)
        ]
        collection = FakeCollection(records)

        result = fetch_user_skill_records(collection, group_id=1, user_id=2, use_rpc=False)

        self.assertEqual(MAX_SKILL_RECORDS, len(result))
        self.assertEqual(5, result[0]['time'])
        self.assertEqual(MAX_SKILL_RECORDS + 4, result[-1]['time'])
        self.assertEqual({'group_id': 1, 'user_id': 2}, collection.last_query)

    def test_formats_text_records_only(self):
        result = format_skill_records([
            {'time': 100, 'plain_text': ' hello '},
            {'time': 200, 'plain_text': ''},
        ])

        self.assertIn('hello', result)
        self.assertNotIn('200', result)

    def test_builds_generation_prompt(self):
        messages = build_skill_generation_messages('小明', 'hello')

        self.assertIn('小明', messages[1]['content'])
        self.assertIn('能力卡', messages[1]['content'])
        self.assertIn('hello', messages[1]['content'])


if __name__ == '__main__':
    unittest.main()
