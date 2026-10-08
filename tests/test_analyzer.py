import json
import unittest

from tiletrace.analyzer import analyze, inspect_transform


def kernel(body, args='BLOCK: tl.constexpr = 8'):
    return 'import triton\nimport triton.language as tl\n@triton.jit\ndef demo(' + args + '):\n' + ''.join('    ' + line + '\n' for line in body.splitlines())


def named(result, name):
    return [node for node in result['nodes'] if node['name'] == name][-1]


class AnalyzerTests(unittest.TestCase):
    def test_default_source_ranges_survive_parameter_overrides(self):
        source = ('import triton\nimport triton.language as tl\n@triton.jit\n'
                  'def demo(\n    A: tl.constexpr=4,\n    B: tl.constexpr=\n        4 * 2\n):\n'
                  '    y = tl.arange(0, B)\n')
        for parameters in ({}, {'B': 16}):
            result = analyze(source, parameters=parameters)
            parameter = named(result, 'B')
            self.assertEqual(parameter['attrs']['default_source'],
                             {'start_line': 7, 'start_col': 8, 'end_line': 7, 'end_col': 13})
            self.assertEqual(named(result, 'y')['shape'], [parameters.get('B', 8)])

    def test_broadcast_and_immediate_coordinates(self):
        result = analyze(kernel('a = tl.zeros((2, 1), tl.float32)\nb = tl.zeros((1, 4), tl.float32)\nc = a + b'))
        node = named(result, 'c')
        self.assertEqual(node['shape'], [2, 4])
        mapping = inspect_transform(result, node['id'], [1, 3])
        self.assertEqual(mapping['status'], 'exact')
        self.assertEqual([o['indices'] for o in mapping['origins']], [[[1, 0]], [[0, 3]]])

    def test_reshape_coordinates_and_reorder(self):
        result = analyze(kernel('a = tl.arange(0, BLOCK)\nb = a.reshape(2, 4)\nc = tl.reshape(a, (2, 4), can_reorder=True)'))
        self.assertEqual(named(result, 'b')['shape'], [2, 4])
        self.assertEqual(inspect_transform(result, named(result, 'b')['id'], [1, 2])['origins'][0]['indices'], [[6]])
        self.assertEqual(inspect_transform(result, named(result, 'c')['id'], [1, 2])['status'], 'unavailable')

    def test_default_transpose_last_two_dimensions(self):
        result = analyze(kernel('a = tl.zeros((2, 4, 8), tl.float32)\nb = a.trans()\nc = tl.permute(a, (2, 0, 1))'))
        self.assertEqual(named(result, 'b')['shape'], [2, 8, 4])
        self.assertEqual(inspect_transform(result, named(result, 'b')['id'], [1, 6, 3])['origins'][0]['indices'], [[1, 3, 6]])
        self.assertEqual(named(result, 'c')['shape'], [8, 2, 4])

    def test_reductions(self):
        result = analyze(kernel('a = tl.zeros((2, 4, 8), tl.float32)\nb = a.sum(-1, keep_dims=True)\nc = tl.max(a, axis=1)\nd = tl.sum(a)\ne = tl.sum(a, axis=None, keep_dims=True)'))
        self.assertEqual(named(result, 'b')['shape'], [2, 4, 1])
        self.assertEqual(named(result, 'c')['shape'], [2, 8])
        self.assertEqual(named(result, 'd')['shape'], [])
        self.assertEqual(named(result, 'e')['shape'], [1, 1, 1])
        self.assertEqual(inspect_transform(result, named(result, 'c')['id'], [1, 6])['origins'][0]['indices'], [[1, i, 6] for i in range(4)])

    def test_large_reduction_bounded(self):
        result = analyze(kernel('a = tl.zeros((1048576, 1048576), tl.float32)\nb = tl.sum(a)'))
        origin = inspect_transform(result, named(result, 'b')['id'], [], limit=3)['origins'][0]
        self.assertEqual(origin['total'], 1048576 ** 2)
        self.assertEqual(origin['indices'], [[0, 0], [0, 1], [0, 2]])
        self.assertTrue(origin['truncated'])

    def test_symbolic_parameters_and_known_arithmetic(self):
        result = analyze(kernel('a = tl.arange(0, BLOCK)\nb = a[:, None]', 'BLOCK: tl.constexpr'))
        self.assertEqual(result['missing_parameters'], ['BLOCK'])
        self.assertEqual(named(result, 'a')['shape'], ['BLOCK'])
        self.assertEqual(named(result, 'b')['shape'], ['BLOCK', 1])
        concrete = analyze(kernel('a = tl.arange(0, triton.next_power_of_2(triton.cdiv(BLOCK, 3)))'), parameters={'BLOCK': 16})
        self.assertEqual(named(concrete, 'a')['shape'], [8])

    def test_incompatible_and_invalid_shapes(self):
        for body in ['a = tl.zeros((2, 3), tl.float32)\nb = tl.zeros((4,), tl.float32)\nc = a + b', 'a = tl.arange(0, 7)', 'a = tl.arange(0, 8)\nc = a.reshape(3, 4)', 'a = tl.arange(0, 8)\nc = a.reshape(-1, 2)']:
            result = analyze(kernel(body))
            self.assertTrue(any(d['severity'] == 'error' for d in result['diagnostics']), body)
            self.assertEqual(result['nodes'][-1]['status'], 'unsupported')

    def test_unknown_branch_invalidates_old_binding(self):
        result = analyze(kernel('a = tl.arange(0, 8)\nif flag:\n    a = tl.arange(0, 16)\nb = a.reshape(2, 4)', 'flag'))
        self.assertEqual(named(result, 'b')['status'], 'unsupported')

    def test_known_branch(self):
        result = analyze(kernel('if BLOCK == 8:\n    a = tl.arange(0, 8)\nelse:\n    a = tl.arange(0, 16)\nb = a.reshape(2, 4)'))
        self.assertEqual(named(result, 'b')['shape'], [2, 4])

    def test_loop_invalidates_old_binding(self):
        result = analyze(kernel('a = tl.arange(0, 8)\nfor i in range(2):\n    a = unknown(a)\nb = a.trans()'))
        self.assertEqual(named(result, 'b')['status'], 'unsupported')

    def test_reassignment_preserves_inputs(self):
        result = analyze(kernel('a = tl.arange(0, 8)\na = a.reshape((2, 4))'))
        nodes = [n for n in result['nodes'] if n['name'] == 'a']
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[-1]['inputs'], [nodes[0]['id']])

    def test_unicode_columns_are_utf16(self):
        source = kernel('标记 = "😀"; a = tl.arange(0, 8)')
        result = analyze(source)
        node = named(result, 'a')
        line = source.splitlines()[4]
        expected = len(line[:line.index('tl.arange')].encode('utf-16-le')) // 2
        self.assertEqual(node['source']['start_col'], expected)

    def test_source_is_never_executed(self):
        result = analyze('raise RuntimeError("DO NOT EXECUTE")\n' + kernel('a = __import__("os").system("DO NOT EXECUTE")'))
        self.assertEqual(named(result, 'a')['status'], 'unsupported')
        json.dumps(result)

    def test_invalid_inspection_and_absent_index(self):
        result = analyze(kernel('a = tl.arange(0, 8)\nb = a.reshape((2, 4))'))
        node = named(result, 'b')
        for index in [None, [2, 0], [1], [True, 0], ['1', 0]]:
            mapping = inspect_transform(result, node['id'], index)
            self.assertEqual(mapping['status'], 'unavailable')
            self.assertEqual(mapping['origins'], [])

    def test_unknown_flags_and_calls_propagate(self):
        result = analyze(kernel('a = tl.arange(0, 8)\nb = tl.reshape(a, (2, 4), can_reorder=flag)\nc = tl.sum(a, keep_dims=flag)\nd = custom(a)\ne = d.reshape((2, 4))', 'flag'))
        self.assertEqual(inspect_transform(result, named(result, 'b')['id'], [0, 0])['status'], 'unavailable')
        self.assertNotEqual(named(result, 'c')['status'], 'resolved')
        self.assertEqual(named(result, 'e')['status'], 'unsupported')

    def test_untrusted_configuration_does_not_crash(self):
        for value in [{}, [], 'bad', True, None]:
            result = analyze(kernel('a = tl.arange(0, BLOCK)\nb = a.reshape((2, 4))'), parameters={'BLOCK': value})
            self.assertTrue(result['diagnostics'])
        self.assertTrue(analyze('not python !!!')['diagnostics'])

    def test_alias_discovery_and_named_function(self):
        source = 'import triton as tr\nimport triton.language as lang\n@tr.jit\ndef one():\n    a = lang.arange(0, 8)\ndef two():\n    b = lang.arange(0, 16)\n'
        self.assertEqual(analyze(source)['kernel'], 'one')
        self.assertEqual(named(analyze(source, kernel='two'), 'b')['shape'], [16])

    def test_load_pointer_shape_and_program_id(self):
        result = analyze(kernel('offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\na = tl.load(X + offsets)\ntl.store(X + offsets, a)', 'X, BLOCK: tl.constexpr = 8'), program_ids=[3])
        self.assertEqual(named(result, 'a')['shape'], [8])
        self.assertTrue(any('store' in d['message'] for d in result['diagnostics']))

    def test_pointer_arguments_are_opaque_not_required_scalar_values(self):
        result = analyze(kernel('offsets = tl.arange(0, BLOCK)\nptr = X + offsets\na = tl.load(ptr, other=-float("inf"))', 'X, BLOCK: tl.constexpr'))
        self.assertEqual(result['missing_parameters'], ['BLOCK'])
        self.assertTrue(named(result, 'X')['attrs']['opaque_pointer'])
        self.assertEqual(named(result, 'a')['shape'], ['BLOCK'])
        self.assertFalse(any(d['severity'] == 'error' for d in result['diagnostics']))
        concrete = analyze(kernel('a = tl.load(X + tl.arange(0, 8) * stride)', 'X, stride'))
        self.assertEqual(concrete['missing_parameters'], ['stride'])
        self.assertEqual(named(concrete, 'a')['shape'], [8])

    def test_slices_expand_dims_and_where(self):
        result = analyze(kernel('a = tl.arange(0, 8)\nb = a[None, :]\nc = a[:, None]\nd = tl.where(c > 2, b, c)\ne = tl.expand_dims(a, 0)'))
        self.assertEqual(named(result, 'd')['shape'], [8, 8])
        self.assertEqual(named(result, 'e')['shape'], [1, 8])
        self.assertEqual(inspect_transform(result, named(result, 'c')['id'], [5, 0])['origins'][0]['indices'], [[5]])

    def test_unsupported_arange_bound_stays_unsupported(self):
        result = analyze(kernel('a = tl.arange(0, custom())\nb = a.reshape((2, 4))'))
        self.assertEqual(named(result, 'a')['status'], 'unsupported')
        self.assertEqual(named(result, 'b')['status'], 'unsupported')

    def test_tuple_variables_and_full_shape_sources(self):
        result = analyze(kernel('dims = (2, 4)\np = (1, 0)\na = tl.full(dims, 0.5, tl.float32)\nb = tl.trans(a, p)\nc = b.reshape(dims)'))
        self.assertEqual(named(result, 'a')['shape'], [2, 4])
        self.assertEqual(named(result, 'b')['shape'], [4, 2])
        self.assertEqual(named(result, 'c')['shape'], [2, 4])
        self.assertEqual(named(result, 'a')['dtype'], 'float32')

    def test_alias_assignment_preserves_old_node_and_constant(self):
        result = analyze(kernel('a = tl.arange(0, 8)\nb = a\na = a.reshape((2, 4))\nc = b[:, None]'))
        self.assertEqual(named(result, 'b')['shape'], [8])
        self.assertEqual(named(result, 'c')['shape'], [8, 1])
        self.assertEqual(named(result, 'a')['shape'], [2, 4])

    def test_numeric_boolean_expressions_preserve_selected_value(self):
        result = analyze(kernel('size = 0 or BLOCK\na = tl.arange(0, size)'))
        self.assertEqual(named(result, 'a')['shape'], [8])

    def test_annotations_invalid_permutation_and_ellipsis(self):
        result = analyze(kernel('a: tl.tensor = tl.zeros((2, 4, 8), tl.float32)\nb = a[..., None]\nc = a.trans(2, 0, 1)\nd = a.trans((0, 0, 2))'))
        self.assertEqual(named(result, 'b')['shape'], [2, 4, 8, 1])
        self.assertEqual(named(result, 'c')['shape'], [8, 2, 4])
        self.assertEqual(named(result, 'd')['status'], 'unsupported')

    def test_num_programs_is_symbolic_and_program_values_validated(self):
        result = analyze(kernel('a = tl.num_programs(0)\nb = tl.program_id(1)'))
        self.assertEqual(named(result, 'a')['status'], 'symbolic')
        self.assertNotIn('value', named(result, 'a')['attrs'])
        self.assertEqual(named(analyze(kernel('b = tl.program_id(1)'), program_ids=[3, 7]), 'b')['attrs']['value'], 7)
        self.assertTrue(analyze(kernel('b = tl.program_id(1)'), program_ids=[True])['diagnostics'])

    def test_empty_limits_and_unknown_node(self):
        result = analyze(kernel('a = tl.arange(0, 8)\nb = tl.sum(a)'))
        mapping = inspect_transform(result, named(result, 'b')['id'], [], limit=0)
        self.assertEqual(mapping['status'], 'exact')
        self.assertEqual(mapping['origins'][0], {'node_id': named(result, 'a')['id'], 'indices': [], 'total': 8, 'truncated': True})
        self.assertEqual(inspect_transform(result, 'unknown')['status'], 'unavailable')

    def test_max_positional_flags_follow_triton_signature(self):
        result = analyze(kernel('a = tl.zeros((2, 4), tl.float32)\nb = a.max(1, False, True, True)\nc = a.max(1, True)'))
        self.assertEqual(named(result, 'b')['shape'], [2, 1])
        self.assertEqual(named(result, 'c')['status'], 'unsupported')

    def test_tensor_cdiv_preserves_shape_and_scalar_provenance(self):
        result = analyze(kernel('a = tl.arange(0, 8)\nx = tl.cdiv(a, 2)\nb = x[:, None]\nc = x.reshape((1,))'))
        self.assertEqual(named(result, 'x')['shape'], [8])
        self.assertEqual(named(result, 'b')['shape'], [8, 1])
        mapping = inspect_transform(result, named(result, 'x')['id'], [5])
        self.assertEqual(mapping['status'], 'exact')
        self.assertEqual([o['indices'] for o in mapping['origins']], [[[5]], [[]]])
        self.assertEqual(named(result, 'c')['status'], 'unsupported')
        self.assertEqual(inspect_transform(result, named(result, 'c')['id'], [0])['status'], 'unavailable')

    def test_tensor_tensor_cdiv_broadcast_and_host_tensor_rejection(self):
        result = analyze(kernel('a = tl.zeros((2, 1), tl.int32)\nb = tl.zeros((1, 4), tl.int32)\nc = tl.cdiv(a, b)\nd = triton.cdiv(a, 2)'))
        self.assertEqual(named(result, 'c')['shape'], [2, 4])
        mapping = inspect_transform(result, named(result, 'c')['id'], [1, 3])
        self.assertEqual([o['indices'] for o in mapping['origins']], [[[1, 0]], [[0, 3]]])
        self.assertEqual(named(result, 'd')['status'], 'unsupported')

    def test_unsupported_rebinding_never_reuses_prior_tensor(self):
        snippets = ['x = (a := custom())', 'del a', 'def a():\n    pass',
                    'import unknown as a', 'from unknown import a',
                    'class a:\n    pass', 'x = custom(value=(a := custom()))']
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                result = analyze(kernel('a = tl.arange(0, 8)\n' + snippet + '\nb = a.reshape((2, 4))'))
                self.assertEqual(named(result, 'b')['status'], 'unsupported')
                self.assertEqual(inspect_transform(result, named(result, 'b')['id'], [1, 3])['status'], 'unavailable')

    def test_nested_scopes_do_not_poison_outer_tensor_bindings(self):
        snippets = ['def helper():\n    a = custom()', 'class Helper:\n    a = custom()',
                    'x = lambda: (a := custom())', 'x = [a for a in items]']
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                result = analyze(kernel('a = tl.arange(0, 8)\n' + snippet + '\nb = a.reshape((2, 4))'))
                self.assertEqual(named(result, 'b')['status'], 'resolved')
                self.assertEqual(inspect_transform(result, named(result, 'b')['id'], [1, 3])['status'], 'exact')

    def test_executing_class_nonlocal_write_invalidates_outer_tensor(self):
        result = analyze(kernel('a = tl.arange(0, 8)\nclass Helper:\n    nonlocal a\n    a = custom()\nb = a.reshape((2, 4))', ''))
        self.assertEqual(named(result, 'b')['status'], 'unsupported')
        self.assertEqual(inspect_transform(result, named(result, 'b')['id'], [1, 3])['status'], 'unavailable')

    def test_class_nonlocal_delete_and_nested_class_write_invalidate(self):
        snippets = ['class Helper:\n    nonlocal a\n    del a',
                    'class Helper:\n    if flag:\n        nonlocal a\n        a = custom()',
                    'class Helper:\n    class Inner:\n        nonlocal a\n        a = custom()']
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                result = analyze(kernel('a = tl.arange(0, 8)\n' + snippet + '\nb = a.reshape((2, 4))', 'flag'))
                self.assertEqual(named(result, 'b')['status'], 'unsupported')
                self.assertEqual(inspect_transform(result, named(result, 'b')['id'], [1, 3])['status'], 'unavailable')

    def test_class_nonlocal_read_and_unexecuted_method_preserve_outer_tensor(self):
        snippets = ['class Helper:\n    nonlocal a\n    value = a',
                    'class Helper:\n    def method(self):\n        nonlocal a\n        a = custom()',
                    'class Helper:\n    nonlocal a\n    def method(self):\n        nonlocal a\n        del a']
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                result = analyze(kernel('a = tl.arange(0, 8)\n' + snippet + '\nb = a.reshape((2, 4))'))
                self.assertEqual(named(result, 'b')['status'], 'resolved')
                self.assertEqual(inspect_transform(result, named(result, 'b')['id'], [1, 3])['status'], 'exact')

    def test_unknown_branch_invalidates_definition_and_import_names(self):
        for statement in ['def a():\n        pass', 'import unknown as a', 'from unknown import a']:
            with self.subTest(statement=statement):
                result = analyze(kernel('a = tl.arange(0, 8)\nif flag:\n    ' + statement + '\nb = a.reshape((2, 4))', 'flag'))
                self.assertEqual(named(result, 'b')['status'], 'unsupported')

    def test_local_module_rebinding_blocks_recognition_of_old_api(self):
        for statement in ['import unknown as tl', 'tl = custom()', 'def tl():\n    pass']:
            with self.subTest(statement=statement):
                result = analyze(kernel(statement + '\na = tl.arange(0, 8)'))
                self.assertEqual(named(result, 'a')['status'], 'unsupported')

    def test_full_keeps_unsupported_fill_dependency(self):
        result = analyze(kernel('a = tl.full((8,), custom(), tl.float32)\nb = a.reshape((2, 4))'))
        node = named(result, 'a')
        self.assertEqual(node['shape'], [8])
        self.assertEqual(node['status'], 'unsupported')
        self.assertTrue(node['inputs'])
        self.assertEqual(named(result, 'b')['status'], 'unsupported')
        self.assertEqual(inspect_transform(result, named(result, 'b')['id'], [1, 3])['status'], 'unavailable')

    def test_full_rejects_tensor_fill_and_retains_symbolic_scalar(self):
        result = analyze(kernel('x = tl.arange(0, 8)\na = tl.full((8,), x, tl.float32)\nb = tl.full((8,), fill, tl.float32)', 'fill'))
        self.assertEqual(named(result, 'a')['status'], 'unsupported')
        self.assertEqual(named(result, 'a')['shape'], [8])
        self.assertEqual(named(result, 'b')['status'], 'symbolic')
        self.assertEqual(named(result, 'b')['inputs'], [named(result, 'fill')['id']])

    def test_keyword_and_mixed_supported_operands_match_positional(self):
        variants = [('tl.arange(0, 8)', 'tl.arange(start=0, end=8)', 'tl.arange(0, end=8)'),
                    ('tl.cdiv(8, 2)', 'tl.cdiv(x=8, div=2)', 'tl.cdiv(8, div=2)'),
                    ('triton.cdiv(8, 2)', 'triton.cdiv(x=8, y=2)', 'triton.cdiv(8, y=2)'),
                    ('tl.where(a > 2, a, 0)', 'tl.where(condition=a > 2, x=a, y=0)', 'tl.where(a > 2, x=a, y=0)'),
                    ('tl.maximum(a, 0)', 'tl.maximum(x=a, y=0)', 'tl.maximum(a, y=0)'),
                    ('tl.exp(a)', 'tl.exp(x=a)', 'tl.exp(a)'),
                    ('triton.next_power_of_2(8)', 'triton.next_power_of_2(n=8)', 'triton.next_power_of_2(8)')]
        for expressions in variants:
            with self.subTest(expressions=expressions):
                result = analyze(kernel('a = tl.arange(0, 8)\n' + '\n'.join('v' + str(i) + ' = ' + expression for i, expression in enumerate(expressions))))
                nodes = [named(result, 'v' + str(i)) for i in range(3)]
                self.assertTrue(all(n['status'] == 'resolved' for n in nodes))
                self.assertEqual([n['shape'] for n in nodes], [nodes[0]['shape']] * 3)
                self.assertEqual([n['attrs'].get('value') for n in nodes], [nodes[0]['attrs'].get('value')] * 3)
                if nodes[0]['shape']:
                    origins = [inspect_transform(result, n['id'], [3])['origins'] for n in nodes]
                    self.assertEqual([[o['indices'] for o in os] for os in origins], [[o['indices'] for o in origins[0]]] * 3)

    def test_duplicate_and_missing_operands_are_diagnosed(self):
        for expression in ['tl.arange(0, start=0, end=8)', 'tl.cdiv(8, x=8, div=2)',
                           'tl.where(a > 2, a, x=a, y=0)', 'tl.maximum(a, x=a, y=0)',
                           'tl.arange(start=0)', 'tl.cdiv(x=8)', 'tl.where(condition=a > 2, x=a)']:
            with self.subTest(expression=expression):
                result = analyze(kernel('a = tl.arange(0, 8)\nb = ' + expression))
                self.assertEqual(named(result, 'b')['status'], 'unsupported')
                self.assertTrue(any(d['severity'] == 'error' for d in result['diagnostics']))


if __name__ == '__main__':
    unittest.main()
