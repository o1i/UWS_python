import networkx as nx
import matplotlib.pyplot as plt

dep = nx.DiGraph()
dep.add_nodes_from(["qx", "qy", "i", "km", "p", "äx", "äy", "kx", "ky", "äwx", "äwy", "uws_x", "uws_y"])
dep.add_edges_from([("qx", "äx"),
                    ("qx", "kx"),
                    ("qx", "äwx"),
                    ("qy", "äy"),
                    ("qy", "ky"),
                    ("qy", "äwy"),
                    ("i", "äx"),
                    ("i", "kx"),
                    ("i", "äwx"),
                    ("i", "äy"),
                    ("i", "ky"),
                    ("i", "äwy"),
                    ("km", "äx"),
                    ("km", "kx"),
                    ("km", "äwx"),
                    ("km", "äy"),
                    ("km", "ky"),
                    ("km", "äwy"),
                    ("p", "uws_x"),
                    ("äx", "uws_x"),
                    ("äwx", "uws_x"),
                    ("kx", "uws_x"),
                    ("p", "uws_y"),
                    ("äy", "uws_y"),
                    ("äwy", "uws_y"),
                    ("ky", "uws_y")
                    ])
nx.draw(dep, with_labels=True)