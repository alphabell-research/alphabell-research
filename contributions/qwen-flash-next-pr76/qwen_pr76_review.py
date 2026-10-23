"""Independent offline review of Qwen3.8-Flash-Next template PR 76.

Fetches four public, hash-checked source blobs, then renders bounded synthetic
conversations. No tokenizer/model loading, inference or tool execution.
"""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import platform
import sys
import argparse
import tempfile
from urllib.request import urlopen

import jinja2
from jinja2 import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def main(ROOT, output_path=None):
    BASE_SHA = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
    PR_SHA = '4381e064c58fa0d453f0630939b04140100ce0f7'
    HASHES = {
        'base-chat_template.jinja': 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041',
        'pr76-chat_template.jinja': '8d5e067223749842f5233b23efd5eafda763bf670edeb8d0a62eca3b9b47340d',
        'base-tokenizer_config.json': 'b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27',
        'pr76-tokenizer_config.json': 'f6e749346e1d8ad055f68c8e0aaf30469d3291d2829ac40635eb26061ec8f60c',
    }
    for name, digest in HASHES.items():
        source_name, source_file = name.split('-', 1)
        revision = BASE_SHA if source_name == 'base' else PR_SHA
        url = f'https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{revision}/{source_file}'
        with urlopen(url, timeout=30) as response:
            blob = response.read(1_000_001)
        if len(blob) > 1_000_000 or hashlib.sha256(blob).hexdigest() != digest:
            raise RuntimeError(f'Source integrity check failed: {name}')
        (ROOT / name).write_bytes(blob)

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = raise_exception
    templates = {name: env.from_string((ROOT / f'{name}-chat_template.jinja').read_text()) for name in ['base', 'pr76']}
    config = {name: json.loads((ROOT / f'{name}-tokenizer_config.json').read_text()) for name in ['base', 'pr76']}
    static_checks = []
    for name in ['base', 'pr76']:
        assert config[name]['chat_template'] == (ROOT / f'{name}-chat_template.jinja').read_text()
        static_checks.append(f'{name} standalone template exactly equals tokenizer_config.chat_template')
    a, b = copy.deepcopy(config['base']), copy.deepcopy(config['pr76'])
    a.pop('chat_template'); b.pop('chat_template'); assert a == b
    static_checks.append('all non-template tokenizer_config values unchanged')

    TOOLS = [{'type': 'function', 'function': {'name': 'read', 'description': 'Read a UTF-8 file, 保留文字.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'options': {'type': 'object'}}, 'required': ['path']}}}]
    USER = {'role': 'user', 'content': 'Read a.txt, then explain briefly.'}
    CALL = {'role': 'assistant', 'content': '', 'reasoning_content': 'Need to inspect the actual file.', 'tool_calls': [{'type': 'function', 'function': {'name': 'read', 'arguments': {'path': 'a.txt', 'options': {'lines': [1, 3], 'literal': '雪 🪴'}}}}]}
    HISTORIES = {
        'simple user': [USER],
        'tool result': [USER, CALL, {'role': 'tool', 'content': 'hello\n雪'}],
        'two tool results and follow-up': [USER, CALL, {'role': 'tool', 'content': 'first'}, {'role': 'tool', 'content': [{'type': 'text', 'text': 'second'}]}, {'role': 'assistant', 'content': 'Read both.', 'reasoning_content': 'Both results are available.'}, {'role': 'user', 'content': 'Now summarize.'}],
        'image and text user': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Inspect these. '}, {'type': 'image_url', 'image_url': {'url': 'unused.png'}}, {'type': 'text', 'text': ' Preserve text.'}]}],
    }
    OPTIONS = [
        ('default', {}),
        ('nonthinking', {'enable_thinking': False}),
        ('medium effort', {'reasoning_effort': 'medium'}),
        ('low effort', {'reasoning_effort': 'low'}),
        ('no preserved thinking', {'preserve_thinking': False}),
        ('no generation header', {'add_generation_prompt': False}),
        ('numbered vision', {'add_vision_id': True}),
        ('combined controls', {'enable_thinking': False, 'preserve_thinking': False, 'add_generation_prompt': False, 'add_vision_id': True}),
    ]
    PREFIXES = {
        'no system': [],
        'one plain system': [{'role': 'system', 'content': 'Be concise.'}],
        'one whitespace system': [{'role': 'system', 'content': ' \n\t '}],
        'one null system': [{'role': 'system', 'content': None}],
        'one empty-list system': [{'role': 'system', 'content': []}],
        'one text-list system': [{'role': 'system', 'content': [{'type': 'text', 'text': 'First.\n'}, {'type': 'text', 'text': 'Second.'}]}],
        'one Unicode system': [{'role': 'system', 'content': '  中文 تعليمات 🪴\nKeep exact quotes: "x".  '}],
    }
    MULTIPLE = {
        'two plain': ['Alpha rule.', 'Beta rule.'],
        'three plain': ['Alpha rule.', 'Beta rule.', 'Gamma rule.'],
        'blank first': ['  ', 'Beta rule.'],
        'blank middle': ['Alpha rule.', '\n\t', 'Gamma rule.'],
        'blank last': ['Alpha rule.', ' '],
        'all blank': [' ', '\n', '\t'],
        'null first': [None, 'Beta rule.'],
        'empty list first': [[], 'Beta rule.'],
        'two text lists': [[{'type': 'text', 'text': ' Alpha.\n'}, {'type': 'text', 'text': 'Piece. '}], [{'type': 'text', 'text': ' Beta. '}]],
        'Unicode and multiline': [' 中文 تعليمات 🪴\n"Alpha". ', ' خط ثان\n第二行. '],
    }

    def render(which, messages, tools, kwargs):
        options = {'add_generation_prompt': True, 'add_vision_id': False, **kwargs}
        try:
            result = templates[which].render(messages=copy.deepcopy(messages), tools=copy.deepcopy(tools), **options)
            return {'ok': True, 'value': result}
        except (TemplateError, TypeError, ValueError) as error:
            return {'ok': False, 'error_type': type(error).__name__, 'error': str(error)}

    def content_text(value):
        if value is None:
            return ''
        if isinstance(value, str):
            return value.strip()
        return ''.join(item['text'] for item in value).strip()

    records = []; failures = []; category_counts = {}
    def record(category, label, passed, details=None):
        category_counts[category] = category_counts.get(category, 0) + 1
        entry = {'category': category, 'label': label, 'passed': bool(passed)}
        if details:
            entry['details'] = details
        records.append(entry)
        if not passed:
            failures.append(entry)

    for (prefix_name, prefix), (history_name, history), (option_name, options), (tool_name, tools) in itertools.product(PREFIXES.items(), HISTORIES.items(), OPTIONS, [('without tools', None), ('with tools', TOOLS)]):
        label = ' / '.join([prefix_name, history_name, option_name, tool_name])
        original, revised = render('base', prefix+history, tools, options), render('pr76', prefix+history, tools, options)
        record('accepted-input byte identity', label, original['ok'] and revised == original, None if revised == original else {'base': original, 'pr76': revised})

    for (prefix_name, contents), (history_name, history), (option_name, options), (tool_name, tools) in itertools.product(MULTIPLE.items(), HISTORIES.items(), OPTIONS, [('without tools', None), ('with tools', TOOLS)]):
        prefix = [{'role': 'system', 'content': value} for value in contents]
        joined = '\n\n'.join(value for value in map(content_text, contents) if value)
        merged_control = [{'role': 'system', 'content': joined}] + history
        original, revised, control = render('base', prefix+history, tools, options), render('pr76', prefix+history, tools, options), render('base', merged_control, tools, options)
        passed = not original['ok'] and original['error'] == 'System message must be at the beginning.' and revised['ok'] and revised == control
        label = ' / '.join([prefix_name, history_name, option_name, tool_name])
        record('new leading-system rendering equals single-merged control', label, passed, None if passed else {'base': original, 'pr76': revised, 'merged_control': control})

    late_histories = {
        'after user': [USER, {'role':'system','content':'Late instruction.'}, {'role':'user','content':'Continue.'}],
        'after assistant': [USER, {'role':'assistant','content':'Ready.'}, {'role':'system','content':'Late instruction.'}, {'role':'user','content':'Continue.'}],
        'after tool': [USER, CALL, {'role':'tool','content':'File data.'}, {'role':'system','content':'Late instruction.'}, {'role':'user','content':'Continue.'}],
    }
    for (prefix_name, prefix), (history_name, history), (option_name, options), (tool_name, tools) in itertools.product(list(PREFIXES.items())[:2], late_histories.items(), OPTIONS, [('without tools', None), ('with tools', TOOLS)]):
        original, revised = render('base', prefix+history, tools, options), render('pr76', prefix+history, tools, options)
        passed = not original['ok'] and revised == original and original['error'] == 'System message must be at the beginning.'
        record('late-system rejection unchanged', ' / '.join([prefix_name, history_name, option_name, tool_name]), passed, None if passed else {'base': original,'pr76':revised})

    invalid = {
        'empty messages': [],
        'system without any user': [{'role':'system','content':'Only system.'}],
        'invalid system content type': [{'role':'system','content':42}, USER],
        'image in first system': [{'role':'system','content':[{'type':'image_url','image_url':{'url':'unused.png'}}]}, USER],
        'video in first system': [{'role':'system','content':[{'type':'video','video':'unused.mp4'}]}, USER],
        'unknown content item in first system': [{'role':'system','content':[{'type':'audio','audio':'unused.wav'}]}, USER],
    }
    for (case_name, messages), (tool_name, tools) in itertools.product(invalid.items(), [('without tools', None), ('with tools', TOOLS)]):
        original, revised = render('base', messages, tools, {}), render('pr76', messages, tools, {})
        record('existing invalid-input rejection unchanged', f'{case_name} / {tool_name}', not original['ok'] and revised == original, None if revised == original else {'base':original,'pr76':revised})

    summary = {
        'base_revision': BASE_SHA, 'pr_head_revision': PR_SHA,
        'python': platform.python_version(), 'jinja2': jinja2.__version__,
        'matrix_cases': len(records), 'passed': sum(x['passed'] for x in records), 'failed': len(failures),
        'category_counts': category_counts, 'static_checks': static_checks,
        'scope':'Offline Jinja text rendering and template/config consistency only. No tokenizer tokenization, inference server, model, application compatibility or generation quality tested.',
    }
    if output_path:
        output_path.write_text(json.dumps({'summary':summary,'failures':failures,'records':records},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if failures:
        print(json.dumps(failures[:4],ensure_ascii=False,indent=2));sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON result file with all case labels')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='qwen-template-review-') as directory:
        main(Path(directory), args.output)
