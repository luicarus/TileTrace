import unittest
from pathlib import Path

from tiletrace import analyze, inspect_transform


def source(body, args='N: tl.constexpr = 2'):
    return 'import triton\nimport triton.language as tl\n@triton.jit\ndef demo(' + args + '):\n' + '\n'.join('    ' + line for line in body.splitlines())


def named(result, name):
    return next(n for n in reversed(result['nodes']) if n['name'] == name)


class AttentionTests(unittest.TestCase):
    def test_dot_coordinates_and_accumulator(self):
        result = analyze(source('a = tl.zeros((2, 4), tl.float16)\nb = tl.zeros((4, 8), tl.float16)\nc = tl.zeros((2, 8), tl.float32)\ny = tl.dot(input=a, other=b, acc=c)'))
        y = named(result, 'y')
        self.assertEqual(y['shape'], [2, 8])
        mapping = inspect_transform(result, y['id'], [1, 6], limit=2)
        self.assertEqual(mapping['status'], 'exact')
        self.assertEqual([o['indices'] for o in mapping['origins']], [[[1, 0], [1, 1]], [[0, 6], [1, 6]], [[1, 6]]])
        self.assertEqual([o['total'] for o in mapping['origins']], [4, 4, 1])
        self.assertTrue(mapping['origins'][0]['truncated'])

    def test_dot_rejects_incompatible_ranks_dimensions_and_accumulator(self):
        for shape, acc in [('(2, 8)', 'None'), ('(2, 4, 8)', 'None'), ('(4, 8)', 'tl.zeros((1, 8), tl.float32)')]:
            result = analyze(source(f'a = tl.zeros((2, 4), tl.float16)\nb = tl.zeros({shape}, tl.float16)\ny = tl.dot(a, b, {acc})'))
            self.assertEqual(named(result, 'y')['status'], 'unsupported')

    def test_large_dot_mapping_is_bounded(self):
        result = analyze(source('a = tl.zeros((2, 1048576), tl.float16)\nb = tl.zeros((1048576, 8), tl.float16)\ny = tl.dot(a, b)'))
        origin = inspect_transform(result, named(result, 'y')['id'], [1, 6], limit=3)['origins'][0]
        self.assertEqual(origin['total'], 1048576)
        self.assertEqual(origin['indices'], [[1, 0], [1, 1], [1, 2]])

    def test_cast_preserves_coordinates(self):
        result = analyze(source('a = tl.zeros((2, 4), tl.float32)\ny = a.to(tl.float16)'))
        y = named(result, 'y')
        self.assertEqual(y['shape'], [2, 4])
        self.assertEqual(y['dtype'], 'float16')
        self.assertEqual(inspect_transform(result, y['id'], [1, 3])['origins'][0]['indices'], [[1, 3]])

    def test_optional_arguments_cannot_hide_binding_changes(self):
        for call in ['a.to(tl.float32, bitcast=(flag := True))',
                     'tl.dot(a, b, input_precision=(flag := "ieee"))',
                     'tl.dot(a, b, out_dtype=(flag := tl.float32))']:
            result = analyze(source('flag = False\na = tl.zeros((2, 4), tl.float32)\nb = tl.zeros((4, 8), tl.float32)\nc = ' + call + '\nif flag:\n    y = tl.zeros((8,), tl.float32)\nelse:\n    y = tl.zeros((4,), tl.float32)'))
            self.assertEqual(named(result, 'c')['status'], 'unsupported', call)
            self.assertEqual(named(result, 'y')['status'], 'unsupported', call)

    def test_nested_loops_share_budget_and_loop_context(self):
        result = analyze(source('a = tl.zeros((2, 4), tl.float32)\nfor i in tl.static_range(2):\n    for j in tl.static_range(2):\n        a = a + i + j\ny = a.to(tl.float16)'))
        additions = [n for n in result['nodes'] if n['op'] == 'add']
        self.assertEqual([l['value'] for l in additions[-1]['attrs']['loops']], [1, 1])
        self.assertEqual(named(result, 'y')['shape'], [2, 4])
        too_large = analyze(source('a = tl.zeros((2, 4), tl.float32)\nfor i in tl.static_range(8):\n    for j in tl.static_range(8):\n        a = a + j\ny = a.to(tl.float16)'))
        self.assertEqual(named(too_large, 'y')['status'], 'unsupported')
        self.assertLess(len(too_large['nodes']), 100)

    def test_static_loop_ssa_and_iteration_labels(self):
        result = analyze(source('a = tl.zeros((2, 4), tl.float32)\nfor start in tl.static_range(0, N * 4, 4):\n    a = a + start\ny = a.to(tl.float16)'))
        additions = [n for n in result['nodes'] if n['op'] == 'add']
        self.assertEqual(len(additions), 2)
        self.assertEqual([n['attrs']['loops'][-1]['value'] for n in additions], [0, 4])
        self.assertIn(additions[0]['id'], additions[1]['inputs'])
        self.assertNotEqual(named(result, 'y')['status'], 'unsupported')

    def test_static_loop_unknown_over_budget_and_control_transfer_are_conservative(self):
        for bounds, body in [('N', 'a = a + i'), ('1000000000', 'a = a + i'), ('2', 'break\n    a = a + i'), ('2', 'return\n    a = a + i'), ('2', 'continue\n    a = a + i')]:
            result = analyze(source(f'a = tl.zeros((2, 4), tl.float32)\nfor i in tl.static_range({bounds}):\n    {body}\ny = a.to(tl.float16)', 'N: tl.constexpr'))
            self.assertEqual(named(result, 'y')['status'], 'unsupported', bounds + body)
            self.assertLess(len(result['nodes']), 30)

    def test_static_empty_and_negative_step(self):
        for bounds, expected in [('0', []), ('4, 0, -2', [4, 2])]:
            result = analyze(source(f'a = tl.zeros((2, 4), tl.float32)\nfor i in tl.static_range({bounds}):\n    a = a + i\ny = a.to(tl.float16)'))
            self.assertEqual([n['attrs']['loops'][-1]['value'] for n in result['nodes'] if n['op'] == 'add'], expected)
            self.assertNotEqual(named(result, 'y')['status'], 'unsupported')

    def test_only_public_demo_is_flash_attention_and_all_shapes_are_available(self):
        examples = Path(__file__).resolve().parents[1] / 'examples'
        self.assertEqual([p.name for p in examples.glob('*.py')], ['flash_attention.py'])
        text = (examples / 'flash_attention.py').read_text(encoding='utf-8')
        for params in ({}, {'CAUSAL': False}, {'N_CTX': 48}):
            result = analyze(text, parameters=params, program_ids=[1])
            self.assertEqual(result['kernels'], ['flash_attention_forward'])
            self.assertEqual(result['missing_parameters'], [])
            self.assertFalse([n for n in result['nodes'] if n['status'] == 'unsupported'])
            self.assertEqual(named(result, 'q')['shape'], [16, 32])
            self.assertEqual(named(result, 'output')['shape'], [16, 32])
            self.assertEqual(len([n for n in result['nodes'] if n['op'] == 'dot']), 4)
            self.assertEqual([n['attrs']['loops'][-1]['value'] for n in result['nodes'] if n['op'] == 'dot'], [0, 0, 32, 32])


if __name__ == '__main__':
    unittest.main()
