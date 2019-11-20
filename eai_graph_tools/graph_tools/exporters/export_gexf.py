"""
    NOTES: networkx's Gephi exporter isn't working very well. To visualize a dynamic networkx into Gephi via GEXF,
    we modify the generated .gexf file to:
        - include timeformat="date" in the following line:
          <graph defaultedgetype="undirected" mode="dynamic" name="" timeformat="date">
        - Generate new edges for each "active" period of the edges in a multigraph (unsupported by networkx).
            ex:
              <edge end="2009-01-05T00:00:00" id="0" source="1" start="2009-01-01T00:00:00" target="2" weight="2.0"/>
              <edge end="2009-01-15T00:00:00" id="1" source="1" start="2009-01-10T00:00:00" target="2" weight="20.0"/>

<?xml version='1.0' encoding='utf-8'?>
<gexf version="1.2" xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:schemaLocation="http://www.w3.org/2001/XMLSchema-instance">
  <graph defaultedgetype="undirected" mode="dynamic" name="" timeformat="date">
    <meta>
      <creator>NetworkX 2.2</creator>
      <lastmodified>27/03/2019</lastmodified>
    </meta>
    <nodes>
      <node end="2009-01-20T00:00:00" id="1" label="192.168.0.1" start="2009-01-01T00:00:00" />
      <node end="2009-01-20T00:00:00" id="2" label="192.168.0.2" start="2009-01-01T00:00:00" />
      <node end="2009-01-20T00:00:00" id="3" label="192.168.0.3" start="2009-01-01T00:00:00" />
    </nodes>
    <edges>
      <edge end="2009-01-05T00:00:00" id="0" source="1" start="2009-01-01T00:00:00" target="2" weight="2.0"/>
      <edge end="2009-01-15T00:00:00" id="1" source="1" start="2009-01-10T00:00:00" target="2" weight="20.0"/>
    </edges>
  </graph>
</gexf>

"""


def dt(pd_timestamp):
    return pd_timestamp.to_pydatetime().isoformat()


def print_node(G, node, node_id):
    return f'      <node end="{dt(G.nodes[node]["end"])}" id="{node_id}" label="{G.nodes[node]["label"]}"' \
           f' start="{dt(G.nodes[node]["start"])}" />'


def print_edge(start, end, edge_id, node_src, node_dst, label):
    return f'      <edge end="{dt(end)}" id="{edge_id}" label="{label}" source="{node_src}" ' \
           f'start="{dt(start)}" target="{node_dst}" weight="1.0"/>'


def gexf_exporter(nx_g, filename):
    header = [
        "<?xml version='1.0' encoding='utf-8'?>",
        '<gexf version="1.2" xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:schemaLocation="http://www.w3.org/2001/XMLSchema-instance">',
        '  <graph defaultedgetype="undirected" mode="dynamic" name="" timeformat="date">',
        '    <meta>',
        '      <creator>NetworkX 2.2</creator>',
        '      <lastmodified>27/03/2019</lastmodified>',
        '    </meta>']

    node_start = '    <nodes>'
    node_end = '    </nodes>'
    edge_start = '    <edges>'
    edge_end = '    </edges>'

    footer = ['  </graph>',
              '</gexf>']

    G = nx_g
    node_id = 0
    edge_id = 0
    node_dict = {}
    with open(filename, 'w') as file:
        for header_line in header:
            file.write(header_line + "\n")

        file.write(node_start + "\n")
        for node in list(G.nodes()):
            node_id = node_id + 1
            node_dict[node] = node_id
            file.write(print_node(G, node, node_id) + "\n")
        file.write(node_end + "\n")

        file.write(edge_start + "\n")
        for u, v, data in G.edges(data=True):
            edge_id = edge_id + 1
            node_src = node_dict[u]
            node_dst = node_dict[v]
            start = data['start']
            end = data['end']
            label = data['label_class']
            file.write(print_edge(start, end, edge_id, node_src, node_dst, label) + "\n")
        file.write(edge_end + "\n")

        for footer_line in footer:
            file.write(footer_line + "\n")
