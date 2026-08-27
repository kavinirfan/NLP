from nltk.parse import DependencyGraph

conll_data = &quot;&quot;&quot;1\tThe\t_\tDET\tDET\t_\t2\tdet\t_\t_
2\tdog\t_\tNOUN\tNOUN\t_\t3\tnsubj\t_\t_
3\tchased\t_\tVERB\tVERB\t_\t0\troot\t_\t_
4\tthe\t_\tDET\tDET\t_\t5\tdet\t_\t_
5\tcat\t_\tNOUN\tNOUN\t_\t3\tdobj\t_\t_&quot;&quot;&quot;

dg = DependencyGraph(conll_data, top_relation_label=&#39;root&#39;)

print(&quot;Root Node:&quot;, dg.root)

tree = dg.tree()
tree.pretty_print()
