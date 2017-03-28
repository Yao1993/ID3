# The round Table
from graphviz import Digraph
# Create a graph object
dot = Digraph(comment='The Round Table')
# Add nodes and edges
dot.node('A', 'King Arthur')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')
dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false', label='Test')
# Save and render the source code
dot.render('../figures/graphviz_demo')
