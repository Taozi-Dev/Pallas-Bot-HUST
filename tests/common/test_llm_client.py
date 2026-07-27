import sys
import types
import unittest

from src.common.llm import LLMClient, LLMConfigError, LLMError


class FakeTimeout(Exception):
    pass


class FakeRequestException(Exception):
    pass


class FakeResponse:
    def __init__(
            self,
            status_code=200,
            data=None,
            text='',
            headers=None,
            url='https://example.com/v1/chat/completions'):
        self.status_code = status_code
        self._data = data
        self.text = text
        self.headers = headers or {}
        self.url = url

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.last_request = None

    def post(self, *args, **kwargs):
        self.last_request = (args, kwargs)
        if self.error:
            raise self.error
        return self.response


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self._requests_module = sys.modules.get('requests')
        sys.modules['requests'] = types.SimpleNamespace(
            Timeout=FakeTimeout,
            RequestException=FakeRequestException,
            Session=lambda: FakeSession(),
        )

    def tearDown(self):
        if self._requests_module is None:
            sys.modules.pop('requests', None)
        else:
            sys.modules['requests'] = self._requests_module

    def test_chat_success(self):
        session = FakeSession(FakeResponse(data={
            'choices': [{'message': {'content': ' 总结内容 '}}],
        }))
        client = LLMClient(
            api_key='key',
            base_url='https://example.com/v1/',
            model='model',
            session=session,
        )

        result = client.chat([{'role': 'user', 'content': 'hello'}])

        self.assertEqual('总结内容', result)
        self.assertEqual('https://example.com/v1/chat/completions', session.last_request[0][0])
        self.assertEqual('Bearer key', session.last_request[1]['headers']['Authorization'])
        self.assertIs(False, session.last_request[1]['json']['stream'])
        self.assertNotIn('max_tokens', session.last_request[1]['json'])

    def test_missing_config(self):
        with self.assertRaises(LLMConfigError):
            LLMClient(api_key='', base_url='https://example.com/v1', model='model')

        with self.assertRaises(LLMConfigError):
            LLMClient(api_key='key', base_url='https://example.com/v1', model='')

    def test_timeout(self):
        client = LLMClient(
            api_key='key',
            base_url='https://example.com/v1',
            model='model',
            session=FakeSession(error=FakeTimeout()),
        )

        with self.assertRaisesRegex(LLMError, 'timed out'):
            client.chat([{'role': 'user', 'content': 'hello'}])

    def test_http_error(self):
        response = FakeResponse(
            status_code=401,
            data={'error': {'message': 'bad key'}},
            text='{"error":{"message":"bad key"}}',
            headers={'Content-Type': 'application/json'},
        )
        client = LLMClient(
            api_key='key',
            base_url='https://example.com/v1',
            model='model',
            session=FakeSession(response),
        )

        with self.assertLogs('src.common.llm', level='ERROR') as logs:
            with self.assertRaisesRegex(LLMError, 'HTTP 401: bad key'):
                client.chat([{'role': 'user', 'content': 'hello'}])

        self.assertIn(response.text, logs.output[0])

    def test_empty_choices(self):
        response = FakeResponse(data={'choices': []}, text='{"choices":[]}')
        client = LLMClient(
            api_key='key',
            base_url='https://example.com/v1',
            model='model',
            session=FakeSession(response),
        )

        with self.assertLogs('src.common.llm', level='ERROR') as logs:
            with self.assertRaisesRegex(LLMError, 'no choices'):
                client.chat([{'role': 'user', 'content': 'hello'}])

        self.assertIn(response.text, logs.output[0])

    def test_invalid_json_logs_raw_response(self):
        response = FakeResponse(
            data=ValueError('invalid json'),
            text='<html>gateway error</html>',
            headers={'Content-Type': 'text/html'},
        )
        client = LLMClient(
            api_key='key',
            base_url='https://example.com/v1',
            model='model',
            session=FakeSession(response),
        )

        with self.assertLogs('src.common.llm', level='ERROR') as logs:
            with self.assertRaisesRegex(LLMError, 'not valid JSON'):
                client.chat([{'role': 'user', 'content': 'hello'}])

        self.assertIn('status=200', logs.output[0])
        self.assertIn('content_type=text/html', logs.output[0])
        self.assertIn(response.text, logs.output[0])

    def test_empty_content_logs_raw_response(self):
        response = FakeResponse(
            data={
                'choices': [{
                    'message': {
                        'content': '',
                        'reasoning_content': 'reasoning used all output tokens',
                    },
                    'finish_reason': 'length',
                }],
            },
            text='{"choices":[{"message":{"content":"","reasoning_content":"reasoning used all output tokens"},"finish_reason":"length"}]}',
            headers={'Content-Type': 'application/json'},
        )
        client = LLMClient(
            api_key='key',
            base_url='https://example.com/v1',
            model='model',
            session=FakeSession(response),
        )

        with self.assertLogs('src.common.llm', level='ERROR') as logs:
            with self.assertRaisesRegex(LLMError, 'content is empty'):
                client.chat([{'role': 'user', 'content': 'hello'}])

        self.assertIn(response.text, logs.output[0])


if __name__ == '__main__':
    unittest.main()
