"""Offline provider contract and UI regressions; no credentials required."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from llm_providers import PROVIDER_MODELS, call_llm, validate_api_key

MESSAGES = [
    {"role": "system", "content": "Return JSON."},
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"},
    {"role": "user", "content": "Follow up"},
]


class ProviderTests(unittest.TestCase):
    def test_openai_catalog_requests(self):
        with patch('openai.OpenAI') as factory:
            create = factory.return_value.chat.completions.create
            create.return_value = NS(choices=[NS(message=NS(content='{}'))])
            for model in PROVIDER_MODELS['OpenAI']:
                for temperature in (0.0, 0.7, 0.9):
                    with self.subTest(model=model, temperature=temperature):
                        self.assertEqual(call_llm('OpenAI', model, 'test', MESSAGES, True, temperature), '{}')
                        args = create.call_args.kwargs
                        self.assertEqual(args['model'], model)
                        self.assertNotIn('max_tokens', args)
                        self.assertGreater(args['max_completion_tokens'], 4096)
                        self.assertEqual(args['response_format'], {'type': 'json_object'})
                        if model.startswith('gpt-5'):
                            self.assertNotIn('temperature', args)
                        else:
                            self.assertEqual(args['temperature'], temperature)

    def test_claude_catalog_and_mixed_content(self):
        with patch('anthropic.Anthropic') as factory:
            create = factory.return_value.messages.create
            create.return_value = NS(content=[NS(type='thinking'), NS(type='text', text='A'), NS(type='text', text='B')])
            for model in PROVIDER_MODELS['Claude']:
                for temperature in (0.0, 0.7, 0.9):
                    with self.subTest(model=model, temperature=temperature):
                        self.assertEqual(call_llm('Anthropic', model, 'test', MESSAGES, temperature=temperature), 'AB')
                        args = create.call_args.kwargs
                        self.assertEqual(args['model'], model)
                        self.assertEqual(args['messages'], MESSAGES[1:])
                        self.assertEqual(args['system'][0]['text'], 'Return JSON.')
                        if model in ('claude-haiku-4-5-20251001', 'claude-sonnet-4-6'):
                            self.assertEqual(args['temperature'], temperature)
                        else:
                            self.assertNotIn('temperature', args)

    def test_gemini_catalog_history_and_json(self):
        with patch('google.genai.Client') as factory:
            create = factory.return_value.models.generate_content
            create.return_value = NS(text='{}')
            for model in PROVIDER_MODELS['Gemini']:
                for temperature in (0.0, 0.7, 0.9):
                    with self.subTest(model=model, temperature=temperature):
                        self.assertEqual(call_llm('Gemini', 'models/' + model, 'test', MESSAGES, True, temperature), '{}')
                        factory.assert_called_with(api_key='test', vertexai=False)
                        args = create.call_args.kwargs
                        self.assertEqual(args['model'], model)
                        self.assertEqual([m.role for m in args['contents']], ['user', 'model', 'user'])
                        config = args['config'].model_dump(exclude_none=True)
                        self.assertNotIn('temperature', config)
                        self.assertEqual(config['response_mime_type'], 'application/json')
                        self.assertEqual(config['system_instruction'], 'Return JSON.')
            self.assertEqual(factory.return_value.close.call_count, 9)

    def test_gemini_errors_are_not_all_bad_keys(self):
        from google.genai.errors import ClientError
        with patch('google.genai.Client') as factory:
            for code, message, expected in [
                (400, 'Invalid argument: temperature', 'temperature'),
                (400, 'API key not valid', 'invalid API key'),
                (403, 'Permission denied', 'permission denied'),
                (404, 'Not found', 'unavailable'),
                (429, 'Quota exceeded', 'quota'),
            ]:
                with self.subTest(code=code, message=message):
                    factory.return_value.models.generate_content.side_effect = ClientError(code, {'error': {'message': message}})
                    ok, result = validate_api_key('Gemini', 'gemini-3.8-flash', 'test')
                    self.assertFalse(ok)
                    self.assertIn(expected, result)

    def test_empty_response_fails_validation(self):
        with patch('openai.OpenAI') as factory:
            factory.return_value.chat.completions.create.return_value = NS(choices=[NS(message=NS(content=None))])
            ok, message = validate_api_key('OpenAI', 'gpt-5-mini', 'test')
            self.assertFalse(ok)
            self.assertIn('no text', message)

    def test_sidebar_validation_resets(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file('app.py').run()
        self.assertFalse(app.exception)
        app.sidebar.text_input[0].set_value('test').run()
        with patch('llm_providers.validate_api_key', return_value=(True, 'OK')):
            app.sidebar.button[0].click().run()
        self.assertTrue(app.session_state['api_key_valid'])
        app.sidebar.selectbox[1].select('gpt-5-mini').run()
        self.assertFalse(app.session_state['api_key_valid'])
        app.session_state['api_key_valid'] = True
        app.sidebar.text_input[0].set_value('different').run()
        self.assertFalse(app.session_state['api_key_valid'])
        app.session_state['api_key_valid'] = True
        app.sidebar.selectbox[0].select('Gemini').run()
        self.assertFalse(app.session_state['api_key_valid'])
        self.assertEqual(app.sidebar.selectbox[1].value, PROVIDER_MODELS['Gemini'][0])
        self.assertEqual(app.sidebar.text_input[0].value, '')
        self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
