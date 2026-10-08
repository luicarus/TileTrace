/* Shared by the editor host and browser: presentation never removes graph nodes. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TileTraceOperations = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const metadata = new Set(['constant', 'parameter', 'symbol', 'dtype', 'tuple', 'alias', 'program_id', 'num_programs', 'store']);
  const semantic = new Set(['arange', 'zeros', 'full', 'load', 'expand_dims', 'reshape', 'trans', 'permute', 'sum', 'max', 'unsupported']);

  function classifier(nodes) {
    const lookup = new Map(nodes.map(node => [node.id, node]));
    const memo = new Map();
    const visiting = new Set();
    function dataRelated(node) {
      if (!node) return false;
      if (memo.has(node.id)) return memo.get(node.id);
      if (visiting.has(node.id)) return false;
      visiting.add(node.id);
      const related = node.status === 'unsupported' || node.shape.length > 0 || semantic.has(node.op) ||
        (!metadata.has(node.op) || node.op === 'alias') && node.inputs.some(id => dataRelated(lookup.get(id)));
      visiting.delete(node.id);
      memo.set(node.id, related);
      return related;
    }
    return {
      primary: node => node.status === 'unsupported' || !metadata.has(node.op) && (semantic.has(node.op) || dataRelated(node)),
      context: node => node.shape.length === 0 && !dataRelated(node)
    };
  }

  function operationNodes(nodes, showAll = false) {
    if (showAll) return nodes;
    const rules = classifier(nodes);
    return nodes.filter(rules.primary);
  }

  function contextInputs(node, nodes) {
    const rules = classifier(nodes);
    const ids = new Set(node.inputs);
    return nodes.filter(input => ids.has(input.id) && rules.context(input));
  }

  function contains(outer, inner) {
    const a = outer.source, b = inner.source;
    return (a.start_line < b.start_line || a.start_line === b.start_line && a.start_col <= b.start_col) &&
      (a.end_line > b.end_line || a.end_line === b.end_line && a.end_col >= b.end_col);
  }

  function operationId(nodes, id, previous) {
    const visible = operationNodes(nodes);
    const visibleIds = new Set(visible.map(node => node.id));
    const lookup = new Map(nodes.map(node => [node.id, node]));
    const fallback = () => visibleIds.has(previous) ? previous : visible[0]?.id;
    const raw = lookup.get(id);
    if (!raw) return fallback();
    if (visibleIds.has(id)) return id;
    // Inline bounds/axis literals belong to the smallest enclosing tensor step.
    const owners = visible.filter(node => contains(node, raw)).sort((a, b) =>
      a.source.end_line - a.source.start_line - (b.source.end_line - b.source.start_line) ||
      a.source.end_col - a.source.start_col - (b.source.end_col - b.source.start_col));
    if (owners.length) return owners[0].id;

    // AST-recorded default ranges identify ownership without comparing values
    // or guessing from line layout; overrides retain this declaration metadata.
    const parameters = nodes.filter(node => node.op === 'parameter' &&
      node.attrs.default_source && contains({source:node.attrs.default_source},raw));
    const start = parameters[0]?.id ?? id;
    const consumers = new Map();
    for (const node of nodes) for (const input of node.inputs) {
      if (!consumers.has(input)) consumers.set(input, []);
      consumers.get(input).push(node.id);
    }
    function nearest(edges) {
      const seen = new Set([start]);
      let frontier = [start];
      while (frontier.length) {
        const next = [];
        for (const item of frontier) for (const child of edges(item)) {
          if (seen.has(child)) continue;
          seen.add(child);
          if (visibleIds.has(child)) return child;
          next.push(child);
        }
        frontier = next;
      }
    }
    // Alias assignments lead back to their producer; preparation leads forward.
    const producers = () => nearest(item => lookup.get(item)?.inputs ?? []);
    if (raw.op === 'alias') return producers() ?? fallback();
    return nearest(item => consumers.get(item) ?? []) ?? producers() ?? fallback();
  }

  return {operationNodes, operationId, contextInputs};
});
