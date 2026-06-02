# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:36:46 2021

@author: abiga
"""
import networkx as nx
import gudhi as gd
try:
    from .vineyard import Simplex
    from .vineyard import Vineyard
except ImportError:
    from vineyard import Simplex
    from vineyard import Vineyard
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class PolygonalSurface:
    def __init__(self, debug_mode = False):
        self.G = nx.Graph() # represents borders of the map
        
        # R is a dictionary of regions (key = region/nbhd/county name, value = 
        # list of nodes bordering that region, in clockwise order). Contains a 
        # key called 'exterior' whose value is a list of nodes adjacent to the 
        # exterior. THE EXTERIOR NODES ARE NOT LISTED IN ANY ORDER.
        self.R = {'exterior': []}
        self.node_adj_R = {}
        self.interior_regions = {}
        self.is_triangulated = False
        self.debug_mode = debug_mode
        
    def add_region(self, name, labels):
        '''
        Parameters
        ----------
        name : Name of the region.
        labels : List of length N, where labels[i] = name of the region that 
                edge (i, i+1) borders. labels[N-1] = name of region that edge 
                (N-1, 0) borders.
        ''' 
        print("adding region:", name)
        self.is_triangulated = False # reset triangulation (probably not computed yet anyway)     
        edge_nbrs = [label for label in labels if label.split(":")[0] != 'vertex']
        N = len(edge_nbrs)
        if len(set(edge_nbrs)) == 1:
            nbr = edge_nbrs[0]
            if nbr != 'exterior':
                if nbr in self.interior_regions:
                    self.interior_regions[nbr].append(name)
                else:
                    self.interior_regions.update({nbr : [name]})
        
        if not len(self.G) == 0:
            max_node= max(self.G.nodes())
        else: max_node = -1
        
        # Make cycle representing boundary of region name, disconnected from the rest of self.G
        i = max_node + 1
        self.G.add_node(i)
        self.R.update({name : [i]}) # add name to self.R dictionary, initialize to include node i
        self.node_adj_R.update({i : {name}}) # initialize node_regions[i]. Currently, name is the only adjacent region to node i.
        for label in labels:
            split_label = label.split(":")
            if split_label[0] == "vertex":
                assert i in self.node_adj_R, f"Can't add vertex adjacency because current node isn't in node_adj_R yet.\n Labels: {labels}"
                self.node_adj_R[i] = self.node_adj_R[i].union({split_label[1]})
            else:
                if len(split_label) == 1:
                    label_name = label
                else:
                    label_name = split_label[1]
                self.node_adj_R[i] = self.node_adj_R[i].union({label_name})
                if (i+1) != max_node + 1 + N:   # max_node + 1 + N is the last node to be added
                    self.G.add_edge(i, i+1, name = label_name)
                    self.R[name].append(i+1)    # Add new node (i+1) to self.R[name]
                    self.node_adj_R.update({i+1: {name, label_name}})   # Add new node (i+1) to node_adj_R
                    i += 1
                else:
                    self.G.add_edge(i, max_node + 1, name = label_name)
                    self.node_adj_R[max_node+1] = self.node_adj_R[max_node+1].union({label_name})
        
        # Attach region N to the rest of the polygonal surface
        for i in range(N):
            x = self.R[name][i]
            y = self.R[name][(i+1)%N]
            edge_label = self.G[x][y]['name']
            if edge_label == 'exterior':
                if not x in self.R['exterior']:
                    self.R['exterior'].append(x)
                if not y in self.R['exterior']:
                    self.R['exterior'].append(y)
            elif edge_label in self.R:
                if self.debug_mode: print(f"attaching {name} to {edge_label}")
                u, v = self.get_target_edge(edge_label, name, x, y)
                
                # An interior region will have 3 edges labelled with name of region its interior to, but the outer region won't reciprocate.
                has_reciprocation = (u is not None and v is not None)
                is_interior = (self.node_adj_R[x] == {edge_label, name}) or (self.node_adj_R[y] == {edge_label, name})
                assert has_reciprocation or is_interior, f"tried to attach {name} to {edge_label} and failed"
                if has_reciprocation:
                    self.quotient(u, v, x, y)
        
        # If region 'name' has an interior region that intersects 'name's outer 
        # boundary in a vertex, quotient those vertices together.
        if name in self.interior_regions:
            inner_rgns = self.interior_regions[name]
            for label in labels:
                split_label = label.split(":")
                if split_label[0] == 'vertex' and split_label[1] in inner_rgns:
                    inner_rgn = split_label[1]
                    u, v = self.get_target_vertices(name, inner_rgn)
                    self.quotient_vertices(u, v)
                    
        # If region 'name' is interior to another region and intersects the 
        # outer region's outer boundary, quotient those vertices together.
        if len(set(edge_nbrs)) == 1 and len(edge_nbrs) < len(labels):
            outer_rgn = edge_nbrs[0]
            if outer_rgn in self.R:
                u, v = self.get_target_vertices(outer_rgn, name)
                if u is not None and v is not None:
                    self.quotient_vertices(u, v)
        
        if self.debug_mode:
            assert not self.has_degenerate_region(), f"region {name} just added was degenerate.\nNodes {self.R[name]} and labels {labels}"
        
    def compare_true_adj(self, adj):
        '''
        Parameters
        ----------
        adj : dictionary
            Key = nbhd name and value = list of adjacent regions, in clockwise 
            order (or counterclockwise-- as long as it's a consistent choice. 
            Just note that the documentation is written with the assumption that 
            adjacencies are listed in clockwise order). Exterior portions of the 
            boundary are treated as an adjacent nbhd with name 'exterior'.
            
        '''
        true_bdry_edges = [] # some edges may appear more than once here
        for rgn in self.R:
            if rgn != 'exterior':
                nodes = self.R[rgn]
                N = len(nodes)
                
                # Check that there's a cycle in G for rgn
                G_has_cycle = True
                for i in range(N):
                    true_bdry_edges.append([nodes[i], nodes[(i+1)%N]])
                    if not self.G.has_edge(nodes[i], nodes[(i+1)%N]):
                        G_has_cycle = False
                        break
                assert G_has_cycle, f"{rgn}'s boundary isn't represented by a cycle in G"
                    
                # Check that self.node_adj_R has correct information about rgn
                for n in self.node_adj_R:
                    if n in nodes:
                        n_statement = f"node {n} is in self.R[{rgn}]"
                    else:
                        n_statement = f"node {n} isn't in self.R[{rgn}]"
                    if rgn in self.node_adj_R[n]:
                        R_statement = f"region {rgn} is in self.node_adj_R[{n}]"
                    else:
                        R_statement = f"region {rgn} isn't in self.node_adj_R[{n}]"
                    is_interior_node = rgn in self.node_adj_R[n] and len(self.node_adj_R[n]) == 2 and not 'exterior' in self.node_adj_R[n]
                    if not is_interior_node:
                        assert (n in nodes) == (rgn in self.node_adj_R[n]), (n_statement + " but " + R_statement)
                    else:
                        # n must be in self.R[rgn] for one of the two regions it's adjacent to
                        nbrs = list(self.node_adj_R[n])
                        assert (nbrs[0] in self.R and n in self.R[nbrs[0]]) or (nbrs[1] in self.R and n in self.R[nbrs[1]]), f"{n} is an interior node that isn't part of the boundary of either region it says it's adjacent to ({nbrs[0]} and {nbrs[1]}"
                
                # Check that it's correctly attached to all its neighbors, as given by adj
                edge_adj = [nbr.split(":")[-1] for nbr in adj[rgn] if nbr.split(":")[0] != 'vertex']
                for i, nbr in enumerate(edge_adj):
                    u, v = nodes[i], nodes[(i+1)%N]
                    if nbr in self.R:
                        nbr_nodes = self.R[nbr]
                        found_v = False
                        found_u = False
                        if nbr == 'exterior':
                            found_u = (u in self.R['exterior'])
                            found_v = (v in self.R['exterior'])
                        else:   
                            N_nbr = len(nbr_nodes)
                            for j, node in enumerate(nbr_nodes):
                                if node == v: 
                                    found_v = True
                                    if nbr_nodes[(j+1)%N_nbr] == u:
                                        found_u = True
                                        break
                        has_reciprocation = found_u and found_v
                        is_interior = (self.node_adj_R[u] == {nbr, rgn}) or (self.node_adj_R[v] == {nbr, rgn})
                        assert has_reciprocation or is_interior, f"\nedge {u}, {v} in {rgn} was supposed to be attached to {nbr}"
                    else:
                        assert nbr in self.node_adj_R[u] and nbr in self.node_adj_R[v], f"Nodes {u} and {v} were supposed to record adjacency to {nbr}"
        
        # Check that self.G doesn't have any edges it's not supposed to
        for u, v in self.G.edges():
            assert ([u,v] in true_bdry_edges or [v, u] in true_bdry_edges), f"Edge ({u}, {v}) is in self.G but isn't part of any region's boundary"
            
        print("\nInterior regions:\n", self.interior_regions)
                
    def create_from_dict_of_adj(adj, debug_mode = False):
        '''
        Parameters
        ----------
        adj : dictionary
            Key = nbhd name and value = list of adjacent regions, in clockwise 
            order (or counterclockwise-- as long as it's a consistent choice. 
            Just note that the documentation is written with the assumption that 
            adjacencies are listed in clockwise order). Exterior portions of the 
            boundary are treated as an adjacent nbhd with name 'exterior'.

        Returns
        -------
        ps : PolygonalSurface
            Regions are the keys of adj and the borders between regions are 
            determined by adj.
            
        Notes
        ------
        Called from read_adj(filename).
        '''
        ps = PolygonalSurface(debug_mode = debug_mode)
        
        # first adjust nbhds with < 3 neighbors
        for nbhd in adj:
            numEdges = len([nbr for nbr in adj[nbhd] if nbr.split(":")[0] != 'vertex'])
            assert numEdges != 0, f"region {nbhd} has no neighbors (not even 'exterior')"
            if numEdges == 1:
                i = 0
                while adj[nbhd][i].split(":")[0] == 'vertex':
                    i += 1
                nbr = adj[nbhd][i]
                adj[nbhd].insert(i, nbr)
                adj[nbhd].insert(i, nbr)
            if numEdges == 2:
                nbr0 = adj[nbhd][0]
                nbr1 = adj[nbhd][1]
                if nbr0 == 'exterior':
                    adj.update({nbhd : ['exterior', 'exterior', nbr1]})
                elif nbr1 == 'exterior':
                    adj.update({nbhd : [nbr0, 'exterior', 'exterior']})
                else:
                    adj[nbhd].append(nbr1)  # add etra 'nbr1' to nbhd adjacency list
                    nbr1_labels = adj[nbr1]
                    for j, nbr_label in enumerate(nbr1_labels):
                        if nbr_label == nbhd:
                            nbr1_labels.insert(j, nbhd)  # add extra 'nbhd' to nbr1 labels, adjacent to the 'nbhd' that was already there
                            break
        for nbhd in adj:
            ps.add_region(nbhd, adj[nbhd])
        if debug_mode:
            ps.compare_true_adj(adj)
        return ps
    
    def expand(self, rgn, n):
        # n is the number of nodes to add to rgn's perimeter by splitting an edge.
        # Adds n nodes to the other region that's adjacent to the split edge, if it exists.
        assert n >= 0, "n must be a nonnegative number"
        if n > 0:
            edge = self.R[rgn][:2]
            u, v = edge[0], edge[1]
            adj_rgns = self.get_adj_rgns(edge)
            adj_rgns.remove(rgn)
            nbr = adj_rgns[0]
            max_node = max(self.G.nodes())
            nx.add_path(self.G, [u] + list(range(max_node + 1, max_node + n + 1)) + [v])
            self.G.remove_edge(u, v)
            for i in range(1, n+1):
                self.R[rgn].insert(i, max_node + i)
                self.node_adj_R.update({max_node + i : {nbr, rgn}})
            if nbr == 'exterior':
                self.R['exterior'] += list(range(max_node + 1, max_node + n + 1))
            elif nbr in self.R:
                nbr_nodes = self.R[nbr]
                nbr_v_idx = 0
                while nbr_nodes[nbr_v_idx] != v:
                    nbr_v_idx += 1
                for i in range(1, n+1):
                    nbr_nodes.insert(nbr_v_idx + i, max_node + n + 1 - i)
            else:
                self.G[u][max_node + 1]['name'] = nbr
                for i in range(1, n):
                    self.G[max_node + i][max_node + i + 1]['name'] = nbr
                self.G[max_node + n + 1][v]['name'] = nbr
            
    def filtration_values(self, region_values, alternative = False):
        '''
        Parameters
        ----------
        region_values is dictionary of lists where region_values[rgn][t] = value 
            of some funcion (say number of cases) at region rgn and time t.
        
        Returns
        -------
        node_values is dictionary s.t. node_values[n][t] 
            = min {nbhd_values[nbhd][t] s.t. n is adjacent to nbhd}
        edge_values is dictionary s.t. edge_values[e][t] 
            = min {nbhd_values[nbhd][t] s.t. e is adjacent to nbhd} 
        tri_values is dictionary s.t. tri_values[tri][t]
            = nbhd_values[nbhd][t], where nbhd is the unique neighborhood containing tri
        
        This is the correct format for the vineyard algorithm.
        '''
        if not self.is_triangulated: self.triangulate()
        
        nbhds = list(region_values.keys())
        numTimes = len(region_values[nbhds[0]])
        
        if alternative:
            C = nx.connected_components(self.G) # I think we only use C to create node_comp
            node_comp = {}
            components = {}
            for idx, comp in enumerate(C):
                node_comp.update({node : idx for node in comp})
                components.update({idx : []})
            for rgn, nodes in self.R.items():
                if rgn != 'exterior':
                    rgn_comp = node_comp[nodes[0]]
                    components[rgn_comp].append(rgn)
                    
            # Adjust node_comp and self.components to account for the fact that 
            # interior regions are disconnected in self.G.
            for rgn, int_rgns in self.interior_regions.items():
                rgn_comp_idx = node_comp[self.R[rgn][0]]
                for int_rgn in int_rgns:
                    nodes = self.R[int_rgn]
                    int_rgn_comp = node_comp[nodes[0]]
                    if int_rgn_comp != rgn_comp_idx:
                        node_comp.update({node : rgn_comp_idx for node in nodes})
                        components[rgn_comp_idx].append(int_rgn)
                        if int_rgn_comp in components:
                            del components[int_rgn_comp]
            
            # Adjust self.E to have separate exterior entries for each component
            self.E.update({f"exterior_{idx}" : [] for idx in components})
            for e in self.E['exterior']:
                comp_idx = node_comp[e[0]]
                self.E[f"exterior_{comp_idx}"].append(e)
            del self.E['exterior']
            
            # Adjust self.R to have separate exterior entries for each component
            self.R.update({f"exterior_{idx}" : [] for idx in components})
            for node, comp_idx in node_comp.items():
                self.R[f"exterior_{comp_idx}"].append(node)
            del self.R['exterior']
            
            # Set region_values for exterior for each component
            comp_mins = {idx : [min([region_values[rgn.split("_")[0]][t] for rgn in comp]) for t in range(numTimes)] for idx, comp in components.items()}
            region_values.update({f"exterior_{idx}" : comp_mins[idx] for idx in components})
        else:
            region_values.update({'exterior' : [min([val[t] for key, val in region_values.items()]) for t in range(numTimes)]})
            if self.debug_mode:
                print("Exterior filtration:", region_values['exterior'])
        vals = list(region_values.values())
        max_val = max([max(v) for v in vals])
        nodes = list(self.G.nodes())
        
        node_values = { n : [max_val for t in range(numTimes)] for n in nodes}  # Initialize node_values so that node_values[n][t] = max_val for every node n at every time t
        for t in range(numTimes):
            for nbhd in self.R:
                if alternative and nbhd.split("_")[0] == 'exterior':
                    nbhd_name = nbhd
                else:
                    nbhd_name = nbhd.split("_")[0]
                nbhd_nodes = self.R[nbhd]
                for n in nbhd_nodes:   
                    if region_values[nbhd_name][t] < node_values[n][t]:
                        n_vals = node_values[n]
                        n_vals[t] = region_values[nbhd_name][t]              

        edge_values = {}
        for t in range(numTimes):
            for nbhd in self.R:
                if alternative and nbhd.split("_")[0] == 'exterior':
                    nbhd_name = nbhd
                else:
                    nbhd_name = nbhd.split("_")[0]
                nbhd_edges = self.E[nbhd_name]
                for e in nbhd_edges:
                    if t == 0 and not tuple(e) in edge_values:
                        edge_values.update({tuple(e) : [max_val for t in range(numTimes)]})
                        e_vals = edge_values[tuple(e)]
                        e_vals[t] = region_values[nbhd_name][t]
                    elif region_values[nbhd_name][t] < edge_values[tuple(e)][t]:
                        e_vals = edge_values[tuple(e)]
                        e_vals[t] = region_values[nbhd_name][t]
        
        tri_values = {}
        for nbhd in self.T:
            for tri in self.T[nbhd]:
                tri_values.update({tuple(tri) : region_values[nbhd]})
        
        return node_values, edge_values, tri_values
    
    def get_adj_rgns(self, edge):
        # edge is a tuple (u, v) that is an edge of one of the regions.
        # PS must not be triangulated yet.
        # Returns list of regions adjacent to edge. At most length 2.
        assert not self.is_triangulated and edge in self.G.edges()
        u = edge[0]
        v = edge[1]
        adj_rgns = []
        possible_rgns = self.node_adj_R[u].intersection(self.node_adj_R[v])
        for rgn in possible_rgns:
            if rgn == 'exterior':
                adj_rgns.append('exterior')
            elif rgn in self.R:
                rgn_nodes = self.R[rgn]
                N = len(rgn_nodes)
                for i, node in enumerate(rgn_nodes):
                    if node == u and (rgn_nodes[(i-1)%N] == v or rgn_nodes[(i+1)%N] == v):
                        adj_rgns.append(rgn)
                        break
            else:
                if self.G.edges[u, v]['name'] == rgn:
                    adj_rgns.append(rgn)
        return adj_rgns
                
    def get_target_edge(self, rgn, other_rgn, x, y):
        # Get edge in rgn where the edge (x, y) in other_rgn should be attached,
        # where x, y are in clockwise order on the boundary of other_rgn.
        # This is the edge (u, v), where u, v are in clockwise order on the boundary
        # of rgn, whose label is other_rgn and s.t. u is adj to the same regions 
        # as y and v is adj to the same regions as x.
        N = len(self.R[rgn])
        possible_targets = []
        for i in range(N):
            u = self.R[rgn][i]
            v = self.R[rgn][(i+1)%N]
            if self.G[u][v]['name'] == other_rgn:
                possible_targets.append([u, v])
        if len(possible_targets) == 1:
            return possible_targets[0][0], possible_targets[0][1]
        elif len(possible_targets) > 0:
            for target in possible_targets:
                u = target[0]
                v = target[1]
                if self.node_adj_R[u] == self.node_adj_R[y] and self.node_adj_R[v] == self.node_adj_R[x]:
                    return u, v
        else:
            return None, None
                
    def get_target_vertices(self, outer_rgn, inner_rgn):
        # This should only be called when inner_rgn is inside outer_rgn
        # If inner_rgn intersects outer boundary of outer_rgn in a vertex,
        # returns u in outer_rgn and v in inner_rgn that should be quotiented.
        assert inner_rgn in self.interior_regions[outer_rgn]
        outer_nodes = self.R[outer_rgn]
        inner_nodes = self.R[inner_rgn]
        u = None
        for x in outer_nodes:
            if inner_rgn in self.node_adj_R[x]:
                u = x
                break
        v = None
        for x in inner_nodes:
            if self.node_adj_R[x] == self.node_adj_R[u]:
                v = x
        return u, v
            
    def has_degenerate_region(self):
        for region in self.R:
            if self.has_duplicate_node(region):
                print(f"region {region} with nodes {self.R[region]} is degenerate")
                return True
        return False
                
    def has_duplicate_node(self, region):
        nodes = self.R[region]
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if nodes[i] == nodes[j]:
                    return True
        return False
    
    def plot_1d_finite_PD(self, st, title = None, suptitle = None, rgn_to_legendname = None, legend_names = None, legend_markers = None, annotate = True, figsize = None, exclude = []):
        # Modified from gudhi plot_persistence_diagram code
        # Requires st.persistence() to be called already. Assumes that st is a
        # SimplexTree object that comes from self (e.g., returned by self.sublevel_SC).
        assert self.is_triangulated
        fig, axes = plt.subplots(1, 1)
        if figsize is not None:
            fig.set_size_inches(figsize, figsize)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fontsize = 16
        alpha = .6
        colormap = plt.cm.Set1.colors
        if rgn_to_legendname is not None:
            if legend_names is None:
                legend_names = set(rgn_to_legendname.values())
            legend_colors = {name : colormap[i] for i, name in enumerate(legend_names)}
        
        pairs = [pair for pair in st.persistence_pairs() if len(pair[0]) == 2 and len(pair[1]) > 0]
        min_birth = None
        max_death = None
        for pair in pairs:
            birth_spx = pair[0]
            death_spx = pair[1]
            name = self.tri_to_region[tuple(sorted(death_spx))]
            if not name in exclude:
                b = st.filtration(birth_spx)
                d = st.filtration(death_spx)
                if min_birth is None:
                    min_birth = b
                else:
                    min_birth = min(min_birth, b)
                if max_death is None:
                    max_death = d
                else:
                    max_death = max(max_death, d)
            
                if rgn_to_legendname is None:
                    color = colormap[0]
                    axes.scatter(b, d, alpha = alpha, color = color)
                else:
                    legendname = rgn_to_legendname[name]
                    marker = legend_markers[legendname]
                    color = legend_colors[legendname]
                    axes.scatter(b, d, alpha = alpha, color = color, label = legendname, marker = marker)
                if annotate:
                    display_name = " ".join([s.capitalize() for s in name.lower().split(" ")])
                    plt.text(b, d, display_name)
        
        inf_delta = .1
        delta = (max_death - min_birth)*inf_delta
        infinity = max_death + delta
        axis_start = min_birth - delta
        axis_end = max_death + delta / 2
        axes.add_patch(mpatches.Polygon([[axis_start, axis_start], [axis_end, axis_start], [axis_end, axis_end]], fill=True, color='lightgrey'))
        axes.set_xlabel("Birth", fontsize=fontsize)
        axes.set_ylabel("Death", fontsize=fontsize)
        axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
        if title is not None:
            axes.set_title(title, fontsize=fontsize)
        if suptitle is not None:
            plt.suptitle(suptitle)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if legend_names is not None:
            handles = [by_label[name] for name in legend_names]
            labels = legend_names
        else:
            handles = by_label.values()
            labels = by_label.keys()
        if rgn_to_legendname is not None:
            plt.legend(handles, labels, loc = "lower right", bbox_to_anchor = (1, 0))
    
    def quotient(self, u, v, x, y):
        # Update self.G
        y_nbrs = self.G.neighbors(y)
        x_nbrs = self.G.neighbors(x)
        u_nbrs = self.G.neighbors(u)
        v_nbrs = self.G.neighbors(v)
        if u != y:
            for w in y_nbrs:
                if w != x:
                    if not w in u_nbrs:
                        self.G.add_edge(u, w, name = self.G[y][w]['name'])
                    else:
                        self.G[u][w]['name'] = None
                    
        if v != x:
            for w in x_nbrs:
                if w != y:
                    if not w in v_nbrs:
                        self.G.add_edge(v, w, name = self.G[x][w]['name'])
                    else:
                        self.G[v][w]['name'] = None
        if u!= y:
            self.G.remove_node(y)
        if v!= x:
            self.G.remove_node(x)
            
        # Update self.R
        for rgn in self.R:
            nodes = self.R[rgn]
            if rgn == 'exterior':
                if y in nodes:
                    if not u in nodes:
                        nodes.append(u)
                    nodes.remove(y)
                if x in nodes:
                    if not v in nodes:
                        nodes.append(v)
                    nodes.remove(x)
            else:
                for i, node in enumerate(nodes):
                    if node == y:
                        nodes[i] = u
                    if node == x:
                        nodes[i] = v
        
        # Update node_adj_R
        if u != y:
            self.node_adj_R[u] = self.node_adj_R[u].union(self.node_adj_R[y])
            del self.node_adj_R[y]
        if v != x:
            self.node_adj_R[v] = self.node_adj_R[v].union(self.node_adj_R[x])
            del self.node_adj_R[x]
            
        self.G[u][v]['name'] = None
    
    def quotient_vertices(self, u, v):
        if u != v:
            # Update self.G
            u_nbrs = self.G.neighbors(u)
            v_nbrs = self.G.neighbors(v)
            for w in v_nbrs:
                if not w in u_nbrs:
                    self.G.add_edge(u, w, name = self.G[v][w]['name'])
                else:
                    self.G[v][w]['name'] = None
            self.G.remove_node(v)
            
            # Update self.R
            for rgn in self.R:
                nodes = self.R[rgn]
                if rgn == 'exterior':
                    if v in nodes:
                        if not u in nodes:
                            nodes.append(u)
                        nodes.remove(v)
                else:
                    for i, node in enumerate(nodes):
                        if node == v:
                            nodes[i] = u
                            
            # Update node_adj_R
            self.node_adj_R[u] = self.node_adj_R[u].union(self.node_adj_R[v])
            del self.node_adj_R[v]
                    
    def read_adj(filename, debug_mode = False):
        # filename encodes the adjacencies of the regions. First word of each
        # line is the region whose adjacencies we're reading. Then we list
        # the adjacent regions, in clockwise order. Regions are separated by tabs. 
        adj_file = open(filename, "r")
        adj = {}
        lines = adj_file.read().splitlines()
        for L in lines:
            L = L.split("\t")   # First word in L is the name of the nbhd. Then a tab. Then adjacent nhbds are separated by tabs.
            nbhd = L[0]
            nbrs = [nbr for nbr in L[1:] if len(nbr) > 0]  # line may have extra tab at end, so get rid of empty spaces
            adj.update({nbhd : nbrs})
        map_from_adj = PolygonalSurface.create_from_dict_of_adj(adj, debug_mode)
        return map_from_adj

    def sublevel_SC(self, region_values, alternative = False, t = None):
        '''
        Given regions values for the keys in self.R (where each value is a list
        of integers), create a filtered complex by filtering by sublevel sets 
        at time t. If t is None, then values in region_values must be integers.
        '''
        if not self.is_triangulated: self.triangulate()
        if t is None:
            region_values = {key : [val] for key, val in region_values.items()}
            t = 0
        node_values, edge_values, tri_values = self.filtration_values(region_values, alternative)
        st = gd.SimplexTree()
        for n in node_values:
            st.insert([n], node_values[n][t])
        for e in edge_values:
            st.insert(list(e), edge_values[e][t])
        for tri in tri_values:
            st.insert(list(tri), tri_values[tri][t])
        return st
    
    def superlevel_SC(self, region_values, alternative = False, t = None):
        if t is None:
            negative_region_values = {key : -val for key, val in region_values.items()}
        else:
            negative_region_values = {key : [-x for x in val] for key, val in region_values.items()}
        return self.sublevel_SC(negative_region_values, alternative, t)
    
    def toSimplices(self, region_vals):
        # Input: Dictionary s.t. region_vals[region] = sequence of case counts in the region (or whatever function)
        # Returns list of Simplex objects.
        if not self.is_triangulated: self.triangulate()
        n_vals, e_vals, tri_vals = self.filtration_values(region_vals)
        
        simplices = []  # initialize list of Simplex objects
        for n in n_vals:
            simplices.append(Simplex([n], n_vals[n]))
        for e in e_vals:
            simplices.append(Simplex(list(e), e_vals[e])) 
        for tri in tri_vals:
            simplices.append(Simplex(list(tri), tri_vals[tri], self.tri_to_region[tri]))
        return simplices
    
    def toVineyard(self, region_vals, dim, debug_mode = False, print_progress = False):
        # Input: Dictionary s.t. region_vals[region] = sequence of case counts in the region (or whatever function)
        # Returns Vineyard object
        simplices = self.toSimplices(region_vals)
        return Vineyard(simplices, dim, debug_mode, print_progress)
    
    def triangulate(self):
        '''
        Creates:
            Dictionary E, where E[i] = list of edges in nbhd i = union of edges 
                in the triangles of nbhd i. Each edge is a (sorted) list of two nodes.
            Dictionary T, where T[i] = list of triangles in nbhd i and each 
                triangle is a list of 3 vertices.
            Dictionary tri_to_region, where tri_to_region[sorted([i, j, k])]
                returns the region that triangle [i, j, k] belongs to.
        Assumptions: Interior regions are isolated from each other, so they only have
            3 nodes/edges. If an interior region intersects the outer boundary 
            of the region surrounding it, then it is the only interior region
            of that outer region.
        '''
        if not self.is_triangulated:
            for rgn in self.interior_regions:
                N_int_rgns = len(self.interior_regions[rgn])
                N_rgn = len(self.R[rgn])
                if N_rgn < N_int_rgns + 2:
                    self.expand(rgn, N_int_rgns + 2 - N_rgn)
                    print(f"expanded {rgn}")
                    
            self.T = {}
            self.E = {'exterior' : []}   # initialize exterior edges list
            self.tri_to_region = {} # key is triangle (tuple of 3 vertices), value is the region that it's a subset of
            
            for nbhd in self.R:
                if not nbhd == 'exterior':
                    nbhd_name = nbhd.split("_")[0]
                    nodes = self.R[nbhd]
                    n_nodes = len(nodes)
                    nbhd_edges = []
                    
                    # Add outer edges of the nbhd
                    for i in range(n_nodes):
                        e = [nodes[i], nodes[(i+1)%n_nodes]]
                        e.sort()
                        nbhd_edges.append(e)
                        
                        # Check if e is an exterior edge, and if so add it to E['exterior']
                        if self.G[e[0]][e[1]]['name'] == 'exterior':
                            self.E['exterior'].append(e)
                    
                        
                    if nbhd in self.interior_regions:
                        nbhd_tris = []
                        int_rgns = self.interior_regions[nbhd]
                        N_irgns = len(int_rgns)
                        num_inner_vertex_adj = 0
                        for irgn in int_rgns:
                            if len(set(self.R[irgn]).intersection(set(self.R[nbhd]))) > 0:
                                num_inner_vertex_adj += 1
                        assert num_inner_vertex_adj == 0 or (num_inner_vertex_adj == 1 and N_irgns == 1)
                        
                        if num_inner_vertex_adj == 0:
                            # Add standard interior edges of the nbhd
                            for i in range(2, n_nodes-1):
                                e = [nodes[0], nodes[i]]
                                e.sort()
                                nbhd_edges.append(e)
                                
                            for i, irgn in enumerate(int_rgns):
                                inodes = self.R[irgn]
                                n_inodes = len(inodes)
                                tri = [nodes[0], nodes[i+1], nodes[i+2]]
                                for j in range(3):
                                    nbhd_edges.append([inodes[j], tri[j]])
                                    nbhd_edges.append([inodes[(j+1)%n_inodes], tri[j]])
                                    nbhd_tris.append(sorted([inodes[j], inodes[(j+1)%n_inodes], tri[j]]))
                                    nbhd_tris.append(sorted([tri[j], tri[(j+1)%3], inodes[(j+1)%n_inodes]]))
                                for j in range(3, n_inodes):
                                    nbhd.edges.append([inodes[j], nodes[0]])
                                    nbhd_tris.append(sorted([inodes[j], inodes[(j+1)%n_inodes], tri[0]]))
                            for i in range(N_irgns, n_nodes - 2):
                                nbhd_tris.append(sorted([nodes[0], nodes[i+1], nodes[i+2]]))
                        else:
                            irgn = int_rgns[0]
                            inodes = self.R[irgn]
                            n_inodes= len(inodes)
                            assert n_inodes == 3    # For now, assume the interior region has 3 sides (it's isolated)
                            i = 0
                            while nodes[i] not in inodes:
                                i += 1
                            j = 0
                            while inodes[j] != nodes[i]:
                                j += 1     
                            # Interior edges
                            nbhd_edges.append([inodes[(j+1) % n_inodes], nodes[(i+1) % n_nodes]])
                            nbhd_edges.append([inodes[(j+1) % n_inodes], nodes[(i+2) % n_nodes]])
                            for k in range(2, n_nodes):
                                nbhd_edges.append([inodes[(j+2) % n_inodes], nodes[(i+k) % n_nodes]])
                            
                            # Triangles
                            nbhd_tris.append(sorted([inodes[j], inodes[(j+1) % n_inodes], nodes[(i+1) % n_nodes]]))
                            nbhd_tris.append(sorted([inodes[(j+1) % n_inodes], inodes[(j+2) % n_inodes], nodes[(i+2) % n_nodes]]))
                            nbhd_tris.append(sorted([inodes[(j+2) % n_inodes], inodes[j], nodes[(i-1) % n_nodes]]))
                            nbhd_tris.append(sorted([nodes[(i+1) % n_nodes], nodes[(i+2) % n_nodes], inodes[(j+1) % n_inodes]]))
                            for k in range(2, n_nodes - 1):
                                nbhd_tris.append(sorted([nodes[(i+k) % n_nodes], nodes[(i+k+1) % n_nodes], inodes[(j+2) % n_inodes]]))
                            
                    else:
                        # Add standard interior edges of the nbhd
                        for i in range(2, n_nodes-1):
                            e = [nodes[0], nodes[i]]
                            e.sort()
                            nbhd_edges.append(e)
                            
                        # Get triangles for nbhd
                        nbhd_tris = [sorted([nodes[0], nodes[i], nodes[i+1]]) for i in range(1, n_nodes - 1)]
                        
                    for tri in nbhd_tris:
                        self.tri_to_region.update({tuple(tri) : nbhd_name})
                    
                    if nbhd_name not in self.E:
                        self.E.update({nbhd_name: nbhd_edges})
                    else:
                        self.E[nbhd_name] = self.E[nbhd_name] + nbhd_edges
                    if nbhd_name not in self.T:
                        self.T.update({nbhd_name: nbhd_tris})
                    else:
                        self.T[nbhd_name] = self.T[nbhd_name] + nbhd_tris
                        
            self.is_triangulated = True 
