import bz2
import xml.etree.ElementTree as ET

def strip_ns(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag

class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

def extract_full_graph(raw_xml_bytes):
    """Returns (nodes, edges) for the WHOLE network (all voltage levels).
    nodes: dict bus_id -> {'vl': voltageLevelId, 'nominal_v': float, 'substation': subId, 'region': str}
    edges: list of dicts {id, kind, bus1, bus2, r, x, b, connected}
    Only lines / twoWindingsTransformer / threeWindingsTransformer are considered branches.
    Bus id = f'{voltageLevelId}#{root_node}' (post switch-merge within a voltage level).
    """
    root = ET.fromstring(raw_xml_bytes)

    vl_nomv = {}
    vl_sub = {}
    vl_region = {}
    vl_uf = {}
    vl_busbar_nodes = {}

    for sub in root:
        if strip_ns(sub.tag) != 'substation':
            continue
        subid = sub.attrib['id']
        region = None
        for prop in sub:
            if strip_ns(prop.tag) == 'property' and prop.attrib.get('name') == 'regionCvg':
                region = prop.attrib.get('value')
        for vl in sub:
            if strip_ns(vl.tag) != 'voltageLevel':
                continue
            vlid = vl.attrib['id']
            nomv = float(vl.attrib.get('nominalV', 'nan'))
            vl_nomv[vlid] = nomv
            vl_sub[vlid] = subid
            vl_region[vlid] = region
            uf = UnionFind()
            busbar_nodes = set()
            topo_kind = vl.attrib.get('topologyKind')
            for topo in vl:
                ttag = strip_ns(topo.tag)
                if ttag == 'nodeBreakerTopology':
                    for el in topo:
                        tag = strip_ns(el.tag)
                        if tag == 'busbarSection':
                            node = int(el.attrib['node'])
                            busbar_nodes.add(node)
                            uf.find(node)
                        elif tag == 'switch':
                            n1 = int(el.attrib['node1']); n2 = int(el.attrib['node2'])
                            is_open = el.attrib.get('open', 'false') == 'true'
                            uf.find(n1); uf.find(n2)
                            if not is_open:
                                uf.union(n1, n2)
                elif ttag == 'busBreakerTopology':
                    # rare fallback: bus id IS the node identity directly
                    for el in topo:
                        tag = strip_ns(el.tag)
                        if tag == 'bus':
                            bid = el.attrib['id']
                            busbar_nodes.add(bid)
                            uf.find(bid)
                        elif tag == 'switch':
                            b1 = el.attrib.get('bus1'); b2 = el.attrib.get('bus2')
                            is_open = el.attrib.get('open', 'false') == 'true'
                            if b1 is not None: uf.find(b1)
                            if b2 is not None: uf.find(b2)
                            if not is_open and b1 is not None and b2 is not None:
                                uf.union(b1, b2)
            vl_uf[vlid] = uf
            vl_busbar_nodes[vlid] = busbar_nodes

    def component_has_busbar(vlid, node):
        uf = vl_uf[vlid]
        r = uf.find(node)
        for bn in vl_busbar_nodes[vlid]:
            if uf.find(bn) == r:
                return True
        return False

    def bus_label(vlid, node):
        uf = vl_uf[vlid]
        return f'{vlid}#{uf.find(node)}'

    # nodes = merged buses (components that contain >=1 busbar section)
    nodes = {}
    for vlid, uf in vl_uf.items():
        seen_roots = set()
        for bn in vl_busbar_nodes[vlid]:
            r = uf.find(bn)
            if r in seen_roots:
                continue
            seen_roots.add(r)
            bus_id = f'{vlid}#{r}'
            nodes[bus_id] = {
                'vl': vlid,
                'nominal_v': vl_nomv[vlid],
                'substation': vl_sub[vlid],
                'region': vl_region[vlid],
            }

    def terminal_bus(vlid, node_attr, bus_attr):
        # node-breaker: node_attr present; bus-breaker: bus_attr present (rare)
        if vlid not in vl_uf:
            return None, False
        if node_attr is not None:
            n = int(node_attr)
            connected = component_has_busbar(vlid, n)
            return (bus_label(vlid, n) if connected else None), connected
        elif bus_attr is not None:
            connected = component_has_busbar(vlid, bus_attr)
            return (bus_label(vlid, bus_attr) if connected else None), connected
        return None, False

    # generators: real presence/capacity (id, energySource, maxP) from RTE7000 --
    # NOT scrubbed, unlike actual dispatch (targetP is absent/unpopulated in this
    # dataset). Nested directly under <voltageLevel>, node-breaker connected like
    # busbar sections/loads.
    generators = []
    for sub in root:
        if strip_ns(sub.tag) != 'substation':
            continue
        for vl in sub:
            if strip_ns(vl.tag) != 'voltageLevel':
                continue
            vlid = vl.attrib['id']
            for el in vl:
                if strip_ns(el.tag) != 'generator':
                    continue
                bus, connected = terminal_bus(vlid, el.attrib.get('node'), el.attrib.get('bus'))
                try:
                    min_p = float(el.attrib.get('minP', '0') or 0)
                    max_p = float(el.attrib.get('maxP', '0') or 0)
                except ValueError:
                    min_p, max_p = 0.0, 0.0
                generators.append({
                    'id': el.attrib.get('id'), 'bus': bus, 'connected': connected,
                    'energy_source': el.attrib.get('energySource'),
                    'min_p': min_p, 'max_p': max_p,
                    'voltage_regulator_on': el.attrib.get('voltageRegulatorOn') == 'true',
                })

    def extract_current_limits(el):
        """Real RTE thermal ratings (Amps) from <operationalLimitsGroup1/2><currentLimits permanentLimit=.../>."""
        selected = {1: el.attrib.get('selectedOperationalLimitsGroupId1'),
                    2: el.attrib.get('selectedOperationalLimitsGroupId2')}
        groups = {1: [], 2: []}
        for child in el:
            ctag = strip_ns(child.tag)
            if ctag == 'operationalLimitsGroup1':
                groups[1].append(child)
            elif ctag == 'operationalLimitsGroup2':
                groups[2].append(child)
        limits = {1: None, 2: None}
        for side in (1, 2):
            chosen = None
            for g in groups[side]:
                if selected[side] is not None and g.attrib.get('id') == selected[side]:
                    chosen = g
                    break
            if chosen is None and groups[side]:
                chosen = groups[side][0]
            if chosen is not None:
                for gc in chosen:
                    if strip_ns(gc.tag) == 'currentLimits':
                        pl = gc.attrib.get('permanentLimit')
                        if pl is not None:
                            try:
                                limits[side] = float(pl)
                            except ValueError:
                                pass
        return limits[1], limits[2]

    def iter_all_branch_candidates(root):
        # lines are always top-level; transformers may be nested inside <substation>
        # (same-substation transformers) or top-level (rare cross-substation transformers)
        for el in root:
            tag = strip_ns(el.tag)
            if tag in ('line', 'twoWindingsTransformer', 'threeWindingsTransformer'):
                yield el
            elif tag == 'substation':
                for sub_el in el:
                    stag = strip_ns(sub_el.tag)
                    if stag in ('twoWindingsTransformer', 'threeWindingsTransformer'):
                        yield sub_el

    edges = []
    for el in iter_all_branch_candidates(root):
        tag = strip_ns(el.tag)
        if tag in ('line', 'twoWindingsTransformer'):
            vlid1 = el.attrib.get('voltageLevelId1')
            vlid2 = el.attrib.get('voltageLevelId2')
            if vlid1 not in vl_uf or vlid2 not in vl_uf:
                continue
            bus1, c1 = terminal_bus(vlid1, el.attrib.get('node1'), el.attrib.get('bus1'))
            bus2, c2 = terminal_bus(vlid2, el.attrib.get('node2'), el.attrib.get('bus2'))
            try:
                r = float(el.attrib.get('r', '0') or 0)
                x = float(el.attrib.get('x', '0') or 0)
            except ValueError:
                r, x = 0.0, 0.0
            b = 0.0
            for attr in ('b1', 'b2'):
                try:
                    b += float(el.attrib.get(attr, '0') or 0)
                except ValueError:
                    pass
            imax1, imax2 = extract_current_limits(el)
            edges.append({
                'id': el.attrib['id'], 'kind': tag,
                'bus1': bus1, 'bus2': bus2, 'connected1': c1, 'connected2': c2,
                'r': r, 'x': x, 'b': b, 'imax1': imax1, 'imax2': imax2,
            })
        elif tag == 'threeWindingsTransformer':
            # 3 legs: leg1/leg2/leg3 sub-elements with their own voltageLevelId/node
            legs = []
            for leg in el:
                ltag = strip_ns(leg.tag)
                if ltag.startswith('leg'):
                    vlid = leg.attrib.get('voltageLevelId')
                    if vlid not in vl_uf:
                        legs.append(None)
                        continue
                    bus, c = terminal_bus(vlid, leg.attrib.get('node'), leg.attrib.get('bus'))
                    try:
                        x = float(leg.attrib.get('x', '0') or 0)
                        r = float(leg.attrib.get('r', '0') or 0)
                    except ValueError:
                        r, x = 0.0, 0.0
                    legs.append({'bus': bus, 'connected': c, 'r': r, 'x': x})
            # connect legs pairwise (star-point implicit) as 3 branches with combined series x
            for i in range(len(legs)):
                for j in range(i+1, len(legs)):
                    a, bb = legs[i], legs[j]
                    if a is None or bb is None:
                        continue
                    edges.append({
                        'id': el.attrib['id'] + f'_leg{i+1}{j+1}', 'kind': 'threeWindingsTransformer',
                        'bus1': a['bus'], 'bus2': bb['bus'],
                        'connected1': a['connected'], 'connected2': bb['connected'],
                        'r': a['r'] + bb['r'], 'x': a['x'] + bb['x'], 'b': 0.0,
                    })

    return nodes, edges, generators

if __name__ == '__main__':
    path = r'C:\Users\lital\AppData\Local\Temp\claude\C--Users-lital-Downloads-Tuindorp---1-min-resolution\73478a93-7764-4684-8a81-786f8cb42119\scratchpad\rte7000_sample_raw\recollement-auto-20211020-0000-enrichi.xiidm.bz2'
    with open(path, 'rb') as f:
        raw = bz2.decompress(f.read())
    nodes, edges, generators = extract_full_graph(raw)
    print('n_nodes', len(nodes))
    print('n_edges', len(edges))
    both_connected = sum(1 for e in edges if e['connected1'] and e['connected2'] and e['bus1'] and e['bus2'])
    print('n_edges_fully_connected', both_connected)
    from collections import Counter
    nomv_dist = Counter(round(v['nominal_v'],1) for v in nodes.values())
    print('nominal_v distribution of nodes:', nomv_dist.most_common())
    kind_dist = Counter(e['kind'] for e in edges)
    print('edge kind distribution:', kind_dist)
    print('n_generators', len(generators), 'n_connected', sum(1 for g in generators if g['connected']))
    print('sample gen:', generators[0] if generators else None)
