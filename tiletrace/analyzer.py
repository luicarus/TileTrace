"""Conservative AST-only Triton shape analysis and lazy input-coordinate maps.

Source is parsed, never imported, compiled, or executed. Scalar value knowledge
is separate from shapes: an unknown scalar-shaped symbol is not a constant.
Returned dictionaries contain JSON data, with no live AST state.
"""

import ast
import itertools
import math
import operator


_UNKNOWN = object()
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.LShift: operator.lshift,
           ast.RShift: operator.rshift, ast.BitAnd: operator.and_,
           ast.BitOr: operator.or_, ast.BitXor: operator.xor}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


def _integer(value):
    return type(value) is int


def _scalar(value):
    return type(value) in (bool, int, float) and (not isinstance(value, float) or math.isfinite(value))


def _concrete(shape):
    return all(_integer(d) and d > 0 for d in shape)


def _status(inputs, shape, forced=None):
    if forced == 'unsupported' or any(n['status'] == 'unsupported' for n in inputs):
        return 'unsupported'
    if forced == 'symbolic' or not _concrete(shape) or any(n['status'] == 'symbolic' for n in inputs):
        return 'symbolic'
    return 'resolved'


class _BindingCollector(ast.NodeVisitor):
    """Collect enclosing-scope bindings without entering nested bodies."""

    def __init__(self):
        self.bindings = {}
        self.wildcard_import = False
        self.nonlocal_names = set()
        self.class_nonlocal_bindings = {}

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bindings[node.id] = node

    def visit_FunctionDef(self, node):
        self.bindings[node.name] = node
        self._function_outer_expressions(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _function_outer_expressions(self, node):
        for expression in getattr(node, 'decorator_list', []):
            self.visit(expression)
        for expression in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(expression)
        for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if getattr(node, 'returns', None) is not None:
            self.visit(node.returns)

    def visit_Lambda(self, node):
        self._function_outer_expressions(node)

    def visit_ClassDef(self, node):
        self.bindings[node.name] = node
        for expression in node.decorator_list + node.bases + [k.value for k in node.keywords]:
            self.visit(expression)
        # Class blocks execute immediately. Their ordinary bindings are local,
        # but nonlocal writes/deletions affect the enclosing function. Function
        # bodies remain excluded by the child collector's normal scope rules.
        body = _BindingCollector()
        for statement in node.body:
            body.visit(statement)
        writes = dict(body.class_nonlocal_bindings)
        writes.update({name: body.bindings[name] for name in body.nonlocal_names if name in body.bindings})
        self.bindings.update(writes)
        self.class_nonlocal_bindings.update(writes)

    def visit_Nonlocal(self, node):
        self.nonlocal_names.update(node.names)

    def visit_Import(self, node):
        for alias in node.names:
            self.bindings[alias.asname or alias.name.split('.')[0]] = node

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name == '*':
                self.wildcard_import = True
            else:
                self.bindings[alias.asname or alias.name] = node

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.bindings[node.name] = node
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name is not None:
            self.bindings[node.name] = node
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest is not None:
            self.bindings[node.rest] = node
        self.generic_visit(node)

    def _comprehension(self, node, expressions):
        # Comprehension targets are local; walrus targets in their expressions
        # bind in the enclosing scope and are collected by the usual visitor.
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for expression in expressions:
            self.visit(expression)

    def visit_ListComp(self, node):
        self._comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self._comprehension(node, [node.key, node.value])


class _Analyzer:
    def __init__(self, source, result):
        self.source = source
        self.lines = source.splitlines()
        self.result = result
        self.env = {}
        self.aliases = {'tl': 'triton.language', 'triton': 'triton'}
        self.program_ids = []

    def range(self, expr):
        def col(line, offset):
            if not 0 < line <= len(self.lines):
                return 0
            prefix = self.lines[line - 1].encode('utf-8')[:offset].decode('utf-8', errors='ignore')
            return len(prefix.encode('utf-16-le')) // 2
        start = getattr(expr, 'lineno', 1)
        end = getattr(expr, 'end_lineno', start)
        return {'start_line': start, 'start_col': col(start, getattr(expr, 'col_offset', 0)),
                'end_line': end, 'end_col': col(end, getattr(expr, 'end_col_offset', 0))}

    def diagnostic(self, message, expr=None, severity='warning'):
        entry = {'severity': severity, 'message': message}
        if expr is not None:
            entry['source'] = self.range(expr)
        self.result['diagnostics'].append(entry)

    def node(self, expr, op, inputs=(), shape=(), attrs=None, status=None, dtype=None):
        inputs = list(inputs)
        node = {'id': 'n' + str(len(self.result['nodes'])), 'op': op,
                'name': ast.get_source_segment(self.source, expr) or op,
                'inputs': [n['id'] for n in inputs], 'shape': list(shape),
                'status': _status(inputs, shape, status), 'source': self.range(expr),
                'attrs': attrs or {}, 'dtype': dtype}
        self.result['nodes'].append(node)
        return node

    def unsupported(self, expr, message, inputs=(), shape=()):
        self.diagnostic(message, expr)
        self.poison_bindings(expr, message)
        return self.node(expr, 'unsupported', inputs, shape, {'reason': message}, 'unsupported')

    def value(self, node):
        return node['attrs'].get('value', _UNKNOWN)

    def symbol(self, expr):
        return ast.get_source_segment(self.source, expr) or '?'

    def path(self, expr):
        if isinstance(expr, ast.Name):
            if expr.id in self.env:
                return 'local:' + expr.id
            return self.aliases.get(expr.id, expr.id)
        if isinstance(expr, ast.Attribute):
            return self.path(expr.value) + '.' + expr.attr
        return ''

    def shape(self, expr):
        if not isinstance(expr, (ast.Tuple, ast.List)):
            n = self.expr(expr)
            value = self.value(n)
            if isinstance(value, list):
                if any(not _integer(d) or d <= 0 for d in value):
                    raise ValueError('Shape dimensions must be positive integers; -1 inference is unsupported.')
                return list(value)
        items = expr.elts if isinstance(expr, (ast.Tuple, ast.List)) else [expr]
        dimensions = []
        for item in items:
            n = self.expr(item)
            value = self.value(n)
            if value is _UNKNOWN:
                if n['status'] == 'unsupported' or n['shape']:
                    raise ValueError('Shape dimensions must be scalar integers.')
                dimensions.append(n['attrs'].get('symbol', self.symbol(item)))
            elif not _integer(value) or value <= 0:
                raise ValueError('Shape dimensions must be positive integers; -1 inference is unsupported.')
            else:
                dimensions.append(value)
        return dimensions

    def broadcast(self, nodes):
        result = []
        rank = max((len(n['shape']) for n in nodes), default=0)
        for i in range(rank):
            dimensions = [n['shape'][-1 - i] if len(n['shape']) > i else 1 for n in nodes]
            nonunit = [d for d in dimensions if d != 1]
            if not nonunit:
                dim = 1
            elif all(d == nonunit[0] for d in nonunit):
                dim = nonunit[0]
            elif all(_integer(d) for d in nonunit):
                raise ValueError('Incompatible shapes for broadcasting: ' + str([n['shape'] for n in nodes]))
            else:
                dim = 'broadcast(' + ', '.join(map(str, nonunit)) + ')'
            result.append(dim)
        return list(reversed(result))

    def scalar_operation(self, expr, inputs, operation):
        attrs = {'mapping': 'broadcast'}
        values = [self.value(n) for n in inputs]
        if all(v is not _UNKNOWN and _scalar(v) for v in values) and all(not n['shape'] for n in inputs):
            try:
                if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.LShift, ast.RShift)):
                    if not _integer(values[1]) or not 0 <= values[1] <= 4096:
                        raise ValueError('Shift amount exceeds static-analysis bound.')
                value = operation(*values)
                if not _scalar(value) or (type(value) is int and value.bit_length() > 4096):
                    raise ValueError('Scalar arithmetic exceeds static-analysis bound.')
                attrs['value'] = value
            except (ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
                self.diagnostic(str(error), expr, 'error')
                return self.node(expr, 'arithmetic', inputs, (), attrs, 'unsupported')
        else:
            attrs['symbol'] = self.symbol(expr)
        return self.node(expr, type(expr.op).__name__.lower() if hasattr(expr, 'op') else 'compare', inputs,
                         self.broadcast(inputs), attrs)

    def expr(self, expr):
        try:
            return self._expr(expr)
        except (ValueError, TypeError, IndexError, KeyError, OverflowError, ZeroDivisionError) as error:
            self.diagnostic(str(error), expr, 'error')
            self.poison_bindings(expr, str(error))
            return self.node(expr, 'unsupported', attrs={'reason': str(error)}, status='unsupported')

    def _expr(self, expr):
        if isinstance(expr, ast.Name):
            if expr.id in self.env:
                return self.env[expr.id]
            return self.node(expr, 'symbol', attrs={'symbol': expr.id}, status='symbolic')
        if isinstance(expr, ast.Constant):
            value = expr.value
            if value is None or _scalar(value) or type(value) is str:
                return self.node(expr, 'constant', attrs={'value': value})
            return self.unsupported(expr, 'Unsupported literal.')
        if isinstance(expr, (ast.Tuple, ast.List)):
            inputs = [self.expr(e) for e in expr.elts]
            values = [self.value(n) for n in inputs]
            attrs = {'value': values} if all(v is not _UNKNOWN for v in values) else {'symbol': self.symbol(expr)}
            return self.node(expr, 'tuple', inputs, attrs=attrs)
        if isinstance(expr, ast.BinOp):
            inputs = [self.expr(expr.left), self.expr(expr.right)]
            if type(expr.op) not in _BINARY:
                return self.unsupported(expr, 'Unsupported arithmetic operator.', inputs)
            return self.scalar_operation(expr, inputs, _BINARY[type(expr.op)])
        if isinstance(expr, ast.UnaryOp):
            inputs = [self.expr(expr.operand)]
            operations = {ast.UAdd: operator.pos, ast.USub: operator.neg,
                          ast.Not: operator.not_, ast.Invert: operator.invert}
            if type(expr.op) not in operations:
                return self.unsupported(expr, 'Unsupported unary operator.', inputs)
            return self.scalar_operation(expr, inputs, operations[type(expr.op)])
        if isinstance(expr, ast.Compare):
            inputs = [self.expr(expr.left)] + [self.expr(e) for e in expr.comparators]
            if any(type(op) not in _COMPARE for op in expr.ops):
                return self.unsupported(expr, 'Unsupported comparison operator.', inputs)
            def compare(*values):
                return all(_COMPARE[type(op)](values[i], values[i + 1]) for i, op in enumerate(expr.ops))
            return self.scalar_operation(expr, inputs, compare)
        if isinstance(expr, ast.BoolOp):
            inputs = [self.expr(e) for e in expr.values]
            def operation(*values):
                for value in values[:-1]:
                    if isinstance(expr.op, ast.And) and not value or isinstance(expr.op, ast.Or) and value:
                        return value
                return values[-1]
            return self.scalar_operation(expr, inputs, operation)
        if isinstance(expr, ast.Subscript):
            input_node = self.expr(expr.value)
            parts = expr.slice.elts if isinstance(expr.slice, ast.Tuple) else [expr.slice]
            return self.slice(expr, input_node, parts)
        if isinstance(expr, ast.Attribute) and expr.attr == 'T':
            input_node = self.expr(expr.value)
            if len(input_node['shape']) != 2:
                return self.unsupported(expr, '.T is supported only for rank-two tensors.', [input_node])
            return self.transpose(expr, input_node, [1, 0])
        if isinstance(expr, ast.Attribute) and self.path(expr).startswith('triton.language.'):
            return self.node(expr, 'dtype', attrs={'symbol': self.path(expr)}, status='symbolic')
        if isinstance(expr, ast.Call):
            return self.call(expr)
        return self.unsupported(expr, 'Unsupported expression: ' + type(expr).__name__)

    def slice(self, expr, input_node, parts):
        rank = len(input_node['shape'])
        ellipses = sum(isinstance(p, ast.Constant) and p.value is Ellipsis for p in parts)
        if ellipses > 1:
            raise ValueError('Only one ellipsis is supported.')
        consuming = sum(isinstance(p, ast.Slice) for p in parts)
        if consuming > rank:
            raise ValueError('Too many slice dimensions.')
        expanded = []
        for part in parts:
            if isinstance(part, ast.Constant) and part.value is Ellipsis:
                expanded.extend([ast.Slice()] * (rank - consuming))
            else:
                expanded.append(part)
        if not ellipses:
            expanded.extend([ast.Slice()] * (rank - consuming))
        shape, axes, consumed = [], [], 0
        for part in expanded:
            if isinstance(part, ast.Constant) and part.value is None:
                axes.append(len(shape))
                shape.append(1)
            elif isinstance(part, ast.Slice) and part.lower is None and part.upper is None and part.step is None:
                shape.append(input_node['shape'][consumed])
                consumed += 1
            else:
                return self.unsupported(expr, 'Only full slices, ellipsis, and new axes are supported.', [input_node])
        return self.node(expr, 'expand_dims', [input_node], shape, {'mapping': 'expand_dims', 'axes': axes})

    def transpose(self, expr, input_node, permutation):
        rank = len(input_node['shape'])
        if len(permutation) != rank or any(not _integer(i) for i in permutation) or sorted(permutation) != list(range(rank)):
            raise ValueError('Permutation must contain every input axis exactly once.')
        return self.node(expr, 'trans', [input_node], [input_node['shape'][i] for i in permutation],
                         {'mapping': 'transpose', 'permutation': permutation})

    def call(self, expr):
        path = self.path(expr.func)
        method = isinstance(expr.func, ast.Attribute) and not path.startswith(('triton.language.', 'triton.'))
        name = expr.func.attr if isinstance(expr.func, ast.Attribute) else path
        if path.startswith(('triton.language.', 'triton.')):
            name = path.rsplit('.', 1)[-1]
        args = list(expr.args)
        keywords = {k.arg: k.value for k in expr.keywords if k.arg is not None}
        if any(k.arg is None for k in expr.keywords) or any(isinstance(a, ast.Starred) for a in args):
            return self.unsupported(expr, 'Expanded call arguments are unsupported.')
        if len(keywords) != len(expr.keywords):
            raise ValueError('Duplicate keyword operand.')
        supported_methods = {'reshape', 'trans', 'permute', 'sum', 'max', 'expand_dims'}
        if method:
            if name not in supported_methods:
                inputs = [self.expr(expr.func.value)] + [self.expr(a) for a in args]
                return self.unsupported(expr, 'Unsupported method call: ' + name, inputs)
            input_node = self.expr(expr.func.value)
        elif path.startswith(('triton.language.', 'triton.')):
            input_node = None
        else:
            inputs = [self.expr(a) for a in args]
            return self.unsupported(expr, 'Unsupported call: ' + (path or self.symbol(expr.func)), inputs)

        argument_nodes = {}

        def evaluated(expression):
            if id(expression) not in argument_nodes:
                argument_nodes[id(expression)] = self.expr(expression)
            return argument_nodes[id(expression)]

        def arg(position, keyword, default=_UNKNOWN):
            if keyword in keywords and len(args) > position:
                raise ValueError('Duplicate operand: ' + keyword)
            if keyword in keywords:
                return self.value(evaluated(keywords[keyword]))
            if len(args) > position:
                return self.value(evaluated(args[position]))
            return default

        def bind(signature, required, keyword_only=()):
            """Bind AST operands using the published positional/keyword names."""
            if len(args) > len(signature):
                raise ValueError(name + ' received too many positional operands.')
            unknown = set(keywords) - set(signature) - set(keyword_only)
            if unknown:
                raise ValueError(name + ' received unknown operands: ' + ', '.join(sorted(unknown)))
            operands = dict(zip(signature, args))
            for keyword, expression in keywords.items():
                if keyword in operands:
                    raise ValueError('Duplicate operand: ' + keyword)
                operands[keyword] = expression
            missing = [operand for operand in required if operand not in operands]
            if missing:
                raise ValueError(name + ' requires operands: ' + ', '.join(missing))
            return operands

        def data():
            if method:
                return input_node
            if args and 'input' in keywords:
                raise ValueError('Duplicate operand: input')
            if not args:
                if 'input' in keywords:
                    return self.expr(keywords['input'])
                raise ValueError('Missing input tensor.')
            return self.expr(args.pop(0))

        if name in ('program_id', 'num_programs'):
            axis = arg(0, 'axis', 0)
            if not _integer(axis) or not 0 <= axis < 3:
                raise ValueError('Program axis must be an integer from zero to two.')
            attrs = {'axis': axis}
            if name == 'program_id' and axis < len(self.program_ids):
                attrs['value'] = self.program_ids[axis]
            else:
                attrs['symbol'] = name + '(' + str(axis) + ')'
            return self.node(expr, name, attrs=attrs, status='symbolic' if 'value' not in attrs else None)
        if name in ('cdiv', 'next_power_of_2'):
            tensor_cdiv = path == 'triton.language.cdiv'
            signature = ('x', 'div' if tensor_cdiv else 'y') if name == 'cdiv' else ('n',)
            operands = bind(signature, signature)
            inputs = [evaluated(operands[operand]) for operand in signature]
            values = [self.value(n) for n in inputs]
            attrs = {'symbol': self.symbol(expr)}
            if not tensor_cdiv and any(n['shape'] for n in inputs):
                return self.unsupported(expr, name + ' host arguments must be scalar-shaped.', inputs)
            shape = self.broadcast(inputs) if tensor_cdiv else []
            if tensor_cdiv:
                attrs['mapping'] = 'broadcast'
            if all(v is not _UNKNOWN for v in values) and all(not n['shape'] for n in inputs):
                if any(not _integer(v) for v in values):
                    raise ValueError(name + ' requires scalar integer arguments.')
                if name == 'cdiv':
                    attrs['value'] = (values[0] + values[1] - 1) // values[1]
                else:
                    if values[0] < 0 or values[0].bit_length() > 4096:
                        raise ValueError('next_power_of_2 argument is outside analysis bounds.')
                    attrs['value'] = 1 << max(0, (values[0] - 1).bit_length()) if values[0] > 1 else 1
            return self.node(expr, name, inputs, shape, attrs=attrs)
        if name == 'arange':
            operands = bind(('start', 'end'), ('start', 'end'))
            start_expr, end_expr = operands['start'], operands['end']
            bounds = [evaluated(start_expr), evaluated(end_expr)]
            start, end = [self.value(n) for n in bounds]
            if any(n['status'] == 'unsupported' or n['shape'] for n in bounds):
                return self.unsupported(expr, 'arange requires supported scalar bounds.', bounds)
            if start is not _UNKNOWN and not _integer(start) or end is not _UNKNOWN and not _integer(end):
                raise ValueError('arange bounds must be integers.')
            attrs = {}
            if start is not _UNKNOWN:
                attrs['start'] = start
            if end is not _UNKNOWN:
                attrs['end'] = end
            if start is not _UNKNOWN and end is not _UNKNOWN:
                length = end - start
                if start < 0 or (start != 0 and start & (start - 1)) or end <= start or end & (end - 1) or length & (length - 1) or length > 1048576:
                    raise ValueError('arange requires a valid power-of-two interval of at most 1048576 elements.')
                shape = [length]
            else:
                shape = [self.symbol(end_expr) if start == 0 else '(' + self.symbol(end_expr) + ' - ' + self.symbol(start_expr) + ')']
            return self.node(expr, 'arange', bounds, shape=shape, attrs=attrs)
        if name in ('zeros', 'full'):
            signature = ('shape', 'dtype') if name == 'zeros' else ('shape', 'value', 'dtype')
            operands = bind(signature, ('shape',) if name == 'zeros' else signature)
            shape_expr = operands['shape']
            shape = self.shape(shape_expr)
            dtype_expr = operands.get('dtype')
            dtype = self.path(dtype_expr).removeprefix('triton.language.') if dtype_expr is not None else None
            attrs = {}
            inputs = []
            if name == 'full':
                fill_expr = operands['value']
                fill = evaluated(fill_expr)
                inputs.append(fill)
                value = self.value(fill)
                if fill['shape'] or value is not _UNKNOWN and not _scalar(value):
                    return self.unsupported(expr, 'full fill value must be scalar-shaped and numeric.', inputs, shape)
                if value is not _UNKNOWN:
                    attrs['fill_value'] = value
                attrs['mapping'] = 'broadcast'
            return self.node(expr, name, inputs, shape=shape, attrs=attrs, dtype=dtype)
        if name in ('reshape', 'trans', 'permute', 'sum', 'max', 'expand_dims'):
            n = data()
            if n['status'] == 'unsupported':
                return self.unsupported(expr, 'Input to ' + name + ' is unsupported.', [n])
            if name == 'reshape':
                shape_exprs = args or ([keywords['shape']] if 'shape' in keywords else [])
                if not shape_exprs:
                    raise ValueError('reshape requires a target shape.')
                shape = self.shape(shape_exprs[0]) if len(shape_exprs) == 1 else [self.shape(a)[0] for a in shape_exprs]
                if _concrete(n['shape']) and _concrete(shape) and math.prod(n['shape']) != math.prod(shape):
                    raise ValueError('reshape must preserve the element count.')
                reorder = arg(len(args), 'can_reorder', False)
                if type(reorder) is not bool:
                    self.diagnostic('Unknown can_reorder flag prevents exact reshape mapping.', expr)
                attrs = {'mapping': 'reshape', 'can_reorder': reorder if type(reorder) is bool else None}
                return self.node(expr, name, [n], shape, attrs)
            if name in ('trans', 'permute'):
                dim_exprs = args or ([keywords['dims']] if 'dims' in keywords else [])
                if not dim_exprs:
                    if name == 'permute' or len(n['shape']) < 2:
                        raise ValueError(name + ' requires a permutation for this input rank.')
                    permutation = list(range(len(n['shape'])))
                    permutation[-2:] = reversed(permutation[-2:])
                else:
                    if len(dim_exprs) == 1 and isinstance(dim_exprs[0], (ast.Tuple, ast.List)):
                        dim_exprs = dim_exprs[0].elts
                    permutation = [self.value(self.expr(a)) for a in dim_exprs]
                    if len(permutation) == 1 and isinstance(permutation[0], list):
                        permutation = permutation[0]
                return self.transpose(expr, n, permutation)
            if name == 'expand_dims':
                axis = arg(0, 'axis')
                if not _integer(axis):
                    raise ValueError('expand_dims axis must be a known integer.')
                rank = len(n['shape']) + 1
                axis = axis + rank if axis < 0 else axis
                if not 0 <= axis < rank:
                    raise ValueError('expand_dims axis is outside the output rank.')
                shape = n['shape'][:axis] + [1] + n['shape'][axis:]
                return self.node(expr, name, [n], shape, {'mapping': 'expand_dims', 'axes': [axis]})
            if name == 'max':
                return_indices = arg(1, 'return_indices', False)
                if return_indices is not False:
                    return self.unsupported(expr, 'max with true or unknown return_indices produces an unsupported tuple result.', [n])
            axis = arg(0, 'axis', None)
            keep = arg(3 if name == 'max' else 1, 'keep_dims', False)
            rank = len(n['shape'])
            if type(keep) is not bool or axis is _UNKNOWN:
                return self.unsupported(expr, 'Reduction axis and keep_dims must be known; output rank is unknown.', [n])
            if axis is None:
                axes = list(range(rank))
            elif _integer(axis):
                axis = axis + rank if axis < 0 else axis
                if not 0 <= axis < rank:
                    raise ValueError('Reduction axis is outside the input rank.')
                axes = [axis]
            else:
                raise ValueError('Reduction axis must be an integer or None.')
            shape = [1 if i in axes else d for i, d in enumerate(n['shape'])] if keep else [d for i, d in enumerate(n['shape']) if i not in axes]
            return self.node(expr, name, [n], shape, {'mapping': 'reduction', 'axes': axes, 'keep_dims': keep})
        if name == 'load':
            if not args and 'pointer' not in keywords:
                raise ValueError('load requires a pointer expression.')
            n = self.expr(args[0] if args else keywords['pointer'])
            return self.node(expr, name, [n], n['shape'], {'mapping': 'identity', 'meaning': 'Pointer-expression shape; global memory addresses are not modeled.'})
        if name == 'store':
            self.diagnostic('tl.store is ignored; this tool analyzes expression shapes, not memory writes.', expr, 'info')
            return self.node(expr, name, attrs={'ignored': True})
        if name in ('exp', 'exp2', 'log', 'sqrt', 'abs', 'where', 'maximum', 'minimum'):
            signature = ('condition', 'x', 'y') if name == 'where' else ('x', 'y') if name in ('maximum', 'minimum') else ('x',)
            optional = ('propagate_nan',) if name in ('maximum', 'minimum') else ()
            operands = bind(signature + optional, signature)
            inputs = [evaluated(operands[operand]) for operand in signature]
            return self.node(expr, name, inputs, self.broadcast(inputs), {'mapping': 'broadcast'})
        inputs = [self.expr(a) for a in args]
        return self.unsupported(expr, 'Unsupported call: ' + path, inputs)

    def assign(self, target, node):
        if isinstance(target, ast.Name):
            # An alias is a new expression output; preserve the previous label.
            if any(n is node for n in self.env.values()):
                node = self.node(target, 'alias', [node], node['shape'], {'mapping': 'identity', **({'value': self.value(node)} if self.value(node) is not _UNKNOWN else {})})
            node['name'] = target.id
            self.env[target.id] = node
            self.aliases.pop(target.id, None)
        else:
            self.diagnostic('Only simple variable assignments are supported.', target)
            for name in ast.walk(target):
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                    n = self.unsupported(name, 'Unsupported assignment target.')
                    n['name'] = name.id
                    self.env[name.id] = n

    def invalidate(self, statement):
        self.diagnostic('Unsupported control flow; assigned variables are invalidated.', statement)
        self.poison_bindings(statement, 'Unsupported syntax or unknown control flow')

    def poison_bindings(self, syntax, reason):
        collector = _BindingCollector()
        collector.visit(syntax)
        if collector.wildcard_import:
            for name in set(self.env) | set(self.aliases):
                collector.bindings.setdefault(name, syntax)
        for name, location in collector.bindings.items():
            node = self.node(location, 'unsupported', attrs={'reason': reason}, status='unsupported')
            node['name'] = name
            self.env[name] = node
            self.aliases.pop(name, None)

    def pointer_parameters(self, function):
        """Recognize syntactic pointer bases without evaluating pointer values.

        Offset/scalar parameters appearing on the right side of pointer
        arithmetic are intentionally not classified as pointer addresses.
        """
        parameters = {a.arg for a in function.args.posonlyargs + function.args.args + function.args.kwonlyargs
                      if self.path(a.annotation) != 'triton.language.constexpr'}
        bindings = {}
        pointers = set()

        def base(expression, seen):
            if isinstance(expression, ast.Name):
                if expression.id in parameters:
                    return expression.id
                if expression.id in bindings and expression.id not in seen:
                    return base(bindings[expression.id], seen | {expression.id})
            elif isinstance(expression, ast.BinOp) and isinstance(expression.op, (ast.Add, ast.Sub)):
                left = base(expression.left, seen)
                return left or (base(expression.right, seen) if isinstance(expression.op, ast.Add) else None)
            elif isinstance(expression, ast.Subscript):
                return base(expression.value, seen)
            return None

        for statement in ast.walk(function):
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        bindings[target.id] = statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.value is not None:
                bindings[statement.target.id] = statement.value
        for expression in ast.walk(function):
            if isinstance(expression, ast.Call) and self.path(expression.func) in ('triton.language.load', 'triton.language.store'):
                pointer = expression.args[0] if expression.args else next((k.value for k in expression.keywords if k.arg == 'pointer'), None)
                if pointer is not None:
                    name = base(pointer, set())
                    if name:
                        pointers.add(name)
        return pointers

    def statements(self, statements):
        for statement in statements:
            if isinstance(statement, ast.Assign):
                node = self.expr(statement.value)
                for target in statement.targets:
                    self.assign(target, node)
            elif isinstance(statement, ast.AnnAssign):
                if statement.value is not None:
                    self.assign(statement.target, self.expr(statement.value))
            elif isinstance(statement, ast.AugAssign):
                expression = ast.BinOp(left=statement.target, op=statement.op, right=statement.value)
                ast.copy_location(expression, statement)
                self.assign(statement.target, self.expr(expression))
            elif isinstance(statement, ast.Expr):
                self.expr(statement.value)
            elif isinstance(statement, ast.If):
                condition = self.expr(statement.test)
                value = self.value(condition)
                if value is not _UNKNOWN and _scalar(value) and not condition['shape'] and condition['status'] != 'unsupported':
                    self.statements(statement.body if value else statement.orelse)
                else:
                    self.invalidate(statement)
            elif isinstance(statement, ast.Return):
                if statement.value is not None:
                    self.expr(statement.value)
                break
            elif isinstance(statement, ast.Pass):
                continue
            else:
                self.invalidate(statement)


def analyze(source: str, kernel: str | None = None, parameters: dict | None = None,
            input_shapes: dict | None = None, program_ids: list[int] | None = None,
            document_id: str = '', version: int = 0) -> dict:
    """Parse a kernel and derive conservative logical tensor shapes."""
    result = {'document_id': document_id, 'version': version, 'kernel': None,
              'kernels': [], 'nodes': [], 'diagnostics': [], 'missing_parameters': [], 'parameters': {}}
    if not isinstance(source, str):
        result['diagnostics'].append({'severity': 'error', 'message': 'Source must be a string.'})
        return result
    engine = _Analyzer(source, result)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as error:
        engine.diagnostic('Cannot parse source: ' + str(error), severity='error')
        return result
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                engine.aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                engine.aliases[alias.asname or alias.name] = (statement.module or '') + '.' + alias.name
    functions = [f for f in tree.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
    kernels = [f for f in functions if any(engine.path(d.func if isinstance(d, ast.Call) else d) == 'triton.jit' for d in f.decorator_list)]
    result['kernels'] = [f.name for f in kernels]
    selected = next((f for f in functions if f.name == kernel), None) if kernel else (kernels[0] if kernels else None)
    if selected is None:
        engine.diagnostic('Requested kernel was not found.' if kernel else 'No @triton.jit kernel found.', severity='error')
        return result
    result['kernel'] = selected.name
    if isinstance(selected, ast.AsyncFunctionDef):
        engine.diagnostic('Async kernels are unsupported.', selected, 'error')
        return result
    if parameters is not None and not isinstance(parameters, dict):
        engine.diagnostic('Parameters must be an object.', severity='error')
        parameters = {}
    parameters = parameters or {}
    if input_shapes is not None and not isinstance(input_shapes, dict):
        engine.diagnostic('Input shapes must be an object.', severity='error')
        input_shapes = {}
    input_shapes = input_shapes or {}
    if program_ids is not None:
        if not isinstance(program_ids, list) or len(program_ids) > 3 or any(not _integer(v) or v < 0 for v in program_ids):
            engine.diagnostic('Program IDs must be a list of at most three nonnegative integers.', severity='error')
        else:
            engine.program_ids = program_ids
    positional = selected.args.posonlyargs + selected.args.args
    defaults = dict(zip([a.arg for a in positional[len(positional) - len(selected.args.defaults):]], selected.args.defaults)) if selected.args.defaults else {}
    defaults.update({a.arg: d for a, d in zip(selected.args.kwonlyargs, selected.args.kw_defaults) if d is not None})
    all_args = positional + selected.args.kwonlyargs
    opaque_pointers = engine.pointer_parameters(selected)
    for argument in all_args:
        attrs = {'symbol': argument.arg, 'constexpr': engine.path(argument.annotation) == 'triton.language.constexpr'}
        if argument.arg in opaque_pointers:
            attrs['opaque_pointer'] = True
        value = _UNKNOWN
        if argument.arg in parameters:
            supplied = parameters[argument.arg]
            if not _scalar(supplied) or (type(supplied) is int and supplied.bit_length() > 4096):
                engine.diagnostic('Parameter ' + argument.arg + ' must be a finite scalar number or boolean.', argument, 'error')
            else:
                value = supplied
        elif argument.arg in defaults:
            value = engine.value(engine.expr(defaults[argument.arg]))
            if value is not _UNKNOWN and not _scalar(value):
                value = _UNKNOWN
        shape = input_shapes.get(argument.arg, [])
        if not isinstance(shape, (list, tuple)) or any(not ((_integer(d) and d > 0) or (type(d) is str and d)) for d in shape):
            engine.diagnostic('Input shape for ' + argument.arg + ' must contain positive integers or symbolic names.', argument, 'error')
            shape = []
        if value is _UNKNOWN:
            if not shape and argument.arg not in opaque_pointers:
                result['missing_parameters'].append(argument.arg)
        else:
            attrs['value'] = value
            result['parameters'][argument.arg] = value
        node = engine.node(argument, 'parameter', shape=shape, attrs=attrs, status='symbolic' if value is _UNKNOWN else None)
        node['name'] = argument.arg
        engine.env[argument.arg] = node
    if selected.args.vararg or selected.args.kwarg:
        engine.diagnostic('Variadic kernel parameters are unsupported.', selected)
    try:
        engine.statements(selected.body)
    except RecursionError:
        engine.diagnostic('Source nesting exceeds static-analysis limits.', severity='error')
    return result


def _unflatten(offset, shape):
    coordinates = [0] * len(shape)
    for i in range(len(shape) - 1, -1, -1):
        coordinates[i] = offset % shape[i]
        offset //= shape[i]
    return coordinates


def inspect_transform(analysis: dict, node_id: str, index: list[int] | None = None,
                      limit: int = 128) -> dict:
    """Map one output coordinate to immediate input coordinates, lazily.

    At most ``limit`` coordinates per input are emitted, capped at 4096. A
    missing index returns metadata only; entire tensors are never allocated.
    """
    nodes = {n['id']: n for n in analysis.get('nodes', [])}
    node = nodes.get(node_id)
    inputs = [nodes[i] for i in node['inputs'] if i in nodes] if node else []
    result = {'node': node, 'inputs': inputs, 'output_index': index if isinstance(index, list) else None,
              'status': 'unavailable', 'origins': []}

    def unavailable(message):
        result['message'] = message
        return result

    if node is None:
        return unavailable('Node not found.')
    if index is None:
        return unavailable('Select an output coordinate to inspect its immediate inputs.')
    if not isinstance(index, list) or any(not _integer(i) for i in index):
        return unavailable('Output coordinates must be a list of integers.')
    if not _integer(limit) or limit < 0:
        return unavailable('Origin limit must be a nonnegative integer.')
    limit = min(limit, 4096)
    if node['status'] == 'unsupported' or not _concrete(node['shape']) or any(n['status'] == 'unsupported' or not _concrete(n['shape']) for n in inputs):
        return unavailable('Unsupported or symbolic dimensions prevent a proven coordinate mapping.')
    if len(index) != len(node['shape']) or any(i < 0 or i >= d for i, d in zip(index, node['shape'])):
        return unavailable('Output coordinate rank or bounds are invalid.')
    mapping = node['attrs'].get('mapping')
    if not mapping or not inputs:
        return unavailable('This node has no modeled immediate input coordinate map.')
    origins = []

    def origin(input_node, coordinates, total=1):
        indices = list(itertools.islice(coordinates, limit))
        if any(len(coord) != len(input_node['shape']) or any(i < 0 or i >= d for i, d in zip(coord, input_node['shape'])) for coord in indices):
            raise ValueError('Derived input coordinate is out of bounds.')
        origins.append({'node_id': input_node['id'], 'indices': indices, 'total': total, 'truncated': total > len(indices)})

    try:
        if mapping == 'broadcast':
            for input_node in inputs:
                shape = input_node['shape']
                offset = len(index) - len(shape)
                if offset < 0 or any(d != 1 and d != node['shape'][offset + i] for i, d in enumerate(shape)):
                    return unavailable('Broadcast compatibility is not proven.')
                origin(input_node, [[0 if d == 1 else index[offset + i] for i, d in enumerate(shape)]])
        elif mapping == 'identity':
            if inputs[0]['shape'] != node['shape']:
                return unavailable('Identity shape compatibility is not proven.')
            origin(inputs[0], [list(index)])
        elif mapping == 'expand_dims':
            axes = node['attrs']['axes']
            origin(inputs[0], [[v for i, v in enumerate(index) if i not in axes]])
        elif mapping == 'transpose':
            permutation = node['attrs']['permutation']
            coordinates = [0] * len(index)
            for i, axis in enumerate(permutation):
                coordinates[axis] = index[i]
            origin(inputs[0], [coordinates])
        elif mapping == 'reshape':
            if node['attrs'].get('can_reorder') is not False:
                return unavailable('can_reorder is true or unknown; logical element order is not guaranteed.')
            if math.prod(inputs[0]['shape']) != math.prod(node['shape']):
                return unavailable('Reshape element count is not proven equal.')
            flat = 0
            for i, d in zip(index, node['shape']):
                flat = flat * d + i
            origin(inputs[0], [_unflatten(flat, inputs[0]['shape'])])
        elif mapping == 'reduction':
            input_node = inputs[0]
            axes = node['attrs']['axes']
            keep = node['attrs']['keep_dims']
            reduced_shape = [input_node['shape'][a] for a in axes]
            total = math.prod(reduced_shape)
            base = [0] * len(input_node['shape'])
            position = 0
            for axis in range(len(base)):
                if axis not in axes:
                    base[axis] = index[axis] if keep else index[position]
                    position += 1

            def coordinates():
                # Avoid itertools.product, which eagerly allocates axis pools.
                for flat in range(min(total, limit)):
                    coord = list(base)
                    for axis, value in zip(axes, _unflatten(flat, reduced_shape)):
                        coord[axis] = value
                    yield coord
            origin(input_node, coordinates(), total)
        else:
            return unavailable('This transform has no proven coordinate mapping.')
    except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
        return unavailable('Transform metadata does not define a valid coordinate mapping.')
    result['origins'] = origins
    result['status'] = 'exact'
    return result
