import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMConfigError(LLMError):
    pass


class LLMClient:
    def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str,
            timeout: int = 60,
            session: Optional[Any] = None) -> None:
        if not api_key:
            raise LLMConfigError('llm_api_key is required')
        if not model:
            raise LLMConfigError('llm_model is required')

        requests = _requests()
        self.api_key = api_key
        self.base_url = (base_url or 'https://api.openai.com/v1').rstrip('/')
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()

    @classmethod
    def from_config(cls, config: Any) -> 'LLMClient':
        return cls(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            timeout=config.llm_timeout,
        )

    def chat(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.2,
            max_tokens: Optional[int] = None) -> str:
        payload: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'stream': False,
        }
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens

        try:
            response = self.session.post(
                f'{self.base_url}/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
                json=payload,
                timeout=self.timeout,
            )
        except _requests().Timeout as error:
            raise LLMError('LLM request timed out') from error
        except _requests().RequestException as error:
            raise LLMError(f'LLM request failed: {error}') from error

        try:
            if response.status_code >= 400:
                raise LLMError(self._format_http_error(response))

            try:
                data = response.json()
            except ValueError as error:
                raise LLMError('LLM response is not valid JSON') from error

            choices = data.get('choices')
            if not choices:
                raise LLMError('LLM response has no choices')

            message = choices[0].get('message') or {}
            content = (message.get('content') or '').strip()
            if not content:
                raise LLMError('LLM response content is empty')

            return content
        except Exception:
            self._log_raw_response(response)
            raise

    @staticmethod
    def _log_raw_response(response: Any) -> None:
        headers = getattr(response, 'headers', {}) or {}
        content_type = headers.get('Content-Type', '') if hasattr(headers, 'get') else ''
        logger.error(
            'LLM raw response on error: status=%s url=%s content_type=%s body=%r',
            getattr(response, 'status_code', ''),
            getattr(response, 'url', ''),
            content_type,
            getattr(response, 'text', ''),
        )

    @staticmethod
    def _format_http_error(response: Any) -> str:
        try:
            data = response.json()
        except ValueError:
            detail = response.text
        else:
            error = data.get('error')
            if isinstance(error, dict):
                detail = error.get('message') or str(error)
            else:
                detail = str(data)

        return f'LLM request failed with HTTP {response.status_code}: {detail}'


def _requests():
    try:
        import requests
    except ImportError as error:
        raise LLMConfigError('requests is required for LLM requests') from error

    return requests
