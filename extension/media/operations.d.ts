import {Node} from '../src/protocol';
export function operationNodes(nodes: Node[], showAll?: boolean): Node[];
export function operationId(nodes: Node[], id?: string, previous?: string): string | undefined;
export function contextInputs(node: Node, nodes: Node[]): Node[];
