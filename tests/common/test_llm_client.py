import sys
import types
import unittest

from src.common.llm import LLMClient, LLMConfigError, LLMError


class FakeTimeout(Exception):
    pass


class FakeRequestException(Exception):
    pass


class FakeResponse:
    def __init__(self, status_code=200, data=None, text=''):
        self.status_code = status_code
        self._data = data
        self.text = text

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
        client = LLMClient(
            api_key='key',
            base_url='https://example.com/v1',
            model='model',
            session=FakeSession(FakeResponse(
                status_code=401,
                data={'error': {'message': 'bad key'}},
            )),
        )

        with self.assertRaisesRegex(LLMError, 'HTTP 401: bad key'):
            client.chat([{'role': 'user', 'content': 'hello'}])

    def test_empty_choices(self):
        client = LLMClient(
            api_key='key',
            base_url='https://example.com/v1',
            model='model',
            session=FakeSession(FakeResponse(data={'choices': []})),
        )

        with self.assertRaisesRegex(LLMError, 'no choices'):
            client.chat([{'role': 'user', 'content': 'hello'}])


if __name__ == '__main__':
    unittest.main()
