# -*- coding: utf-8 -*-
"""
Implementation of vineyard algorithm
"""
from queue import PriorityQueue
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import gudhi as gd
import pickle

def read_simplices(filename, start_time = 0, end_time = None):
    simplices = []
    file = open(filename)
    lines = file.read().splitlines()
    for line in lines:
        split_line = line.split(" ; ")
        node_chars = split_line[0].split(" ")
        nodes = [int(node) for node in node_chars]
        val_chars = split_line[1].split(" ")
        val = [float(value) for value in val_chars if len(value) > 0]
        if end_time is not None:
            val = val[start_time : end_time+1]
        else:
            val = val[start_time :]
        simplices.append(Simplex(nodes, val))
    return simplices
        
class Event:
    def __init__(self, t, y = None, event_type = None):
        self.t = t
        self.y = y
        self.event_type = event_type
        
    def __lt__(self, other_event):
        # Sort by time first.
        if self.t < other_event.t:
            return True
        elif self.t > other_event.t:
            return False
        else:
            # In this case, events occur at the same time.
            # Prioritize by event type. All activate events happen before
            # crossings, no matter the y-coordinate. And timestep events happen first.
            if self.event_type == "timestep":
                return True
            elif self.event_type == "activate" and other_event.event_type == "crossing":
                return True
            elif self.event_type == "crossing" and other_event.event_type == "activate":
                return False
            else:
                # In this case, they're the same type of event. Prioritize by 
                # y-coord. Lower y-coord events go first.
                if self.y < other_event.y:
                    return True
                elif self.y > other_event.y:
                    return False
                else:
                    # In this case, they're the same type of event, at same time, with same y-coord.
                    # For now, assume this is an activate_event... because two overlapping crossing_events are unlikely
                    if self.event_type == "activate":
                        return self.spx.curr_idx < other_event.spx.curr_idx
    
    def __str__(self):
        return f"Time: {self.t} \ny-coordinate: {self.y} \nEvent type: {self.event_type}"

class CrossingEvent(Event):
    def __init__(self, spx1, spx2, t, y):
        super().__init__(t, y, "crossing")
        self.spx1 = spx1
        self.spx2 = spx2
        
class ActivateEvent(Event):
    # At each time step, we reactivate each simplex.
    def __init__(self, spx, t):
        y = spx[t]
        super().__init__(t, y, "activate")
        self.spx = spx
        self.t = t

class Node:
    # In the vinyard context- each node stores its row index in the original reduced matrix
    def __init__(self, data = None):
        self.data = data
        self.next = None
    
    def copy(node):
        copy_node = Node(node.data)
        copy_node.next = node.next
        return copy_node
    
    def print(self):
        print("Data value: ",self.data)
        if self.next is not None:
            print("Next node value: ", self.next.data)
        else:
            print("Next node: None")
        
class linkedList:
    # In the vineyard context- need to be able to add nodes to the end of the 
    # list, remove node with particular value, merge two lists (delete 
    # duplicate nodes), insert nodes at correct place, and check if any of the 
    # nodes have a particular value. The linked list must stay sorted by value, 
    # which represents a node's original row index.
    def __init__(self):
        self.head = None
        
    def __iadd__(self, other):
        other_curr = other.head
        #if other_curr is not None:
        while (other_curr is not None and (self.head is None or other_curr.data <= self.head.data)):
            if self.head is None or other_curr.data < self.head.data:
                old_head = self.head
                self.head = Node(other_curr.data)
                self.head.next = old_head
                other_curr = other_curr.next
            else:
                self.head = self.head.next
                other_curr = other_curr.next
        self_curr = self.head
            
        # Now self_curr is not None and other_curr.data > self_curr.data.
        # We're going to insert a copy of other_curr somewhere after self_curr, 
        # although maybe not immediately afterwards.
        while other_curr is not None:
            if self_curr.next is None or other_curr.data < self_curr.next.data:
                # insert a copy of other_curr immediately after self.curr
                new_node = Node(other_curr.data)
                new_node.next = self_curr.next
                self_curr.next = new_node
                other_curr = other_curr.next # increment other_curr
                if self_curr.next is not None:
                    self_curr= self_curr.next
            elif other_curr.data == self_curr.next.data:
                self_curr.next = self_curr.next.next # delete self_curr from self
                other_curr = other_curr.next # increment other_curr
            elif self_curr.next is not None:
                self_curr = self_curr.next # increment self_curr
        return self
                    
    def append(self, node):
        # Append node to the end of the linked list.
        # Seems to mostly (only?) get used in debugging/test functions.
        if self.head is None:
            self.head = node
        else:
            curr = self.head
            while(curr.next is not None):
                curr = curr.next
            curr.next = node
    
    def toList(self):
        # This is only for debugging.
        node_data = []
        curr = self.head
        while curr is not None:
            node_data.append(curr.data)
            curr = curr.next
        return node_data
            
    def has(self, data_val):
        # Returns true if any of the nodes have data = data_val.
        curr = self.head
        while(curr is not None):
            if curr.data == data_val:
                return True
            curr = curr.next
        return False
    
    def insert(self, node):
        # Insert node at correct position to keep list sorted.
        # This gets used when constructing the initial boundary matrix.
        curr = self.head
        if curr is None or curr.data > node.data:
            node.next = self.head
            self.head = node
        else: 
            while(curr.next is not None and curr.next.data < node.data):
                curr = curr.next
            # Now either curr.next is None (we're at the end of the list), or 
            # curr.next is the first node whose value is >= node.data. In either
            # case, we want to insert node right after curr
            node.next = curr.next # could be None
            curr.next = node
        
    def print(self):
        nodes = self.toList()
        print(nodes)
    
    def remove(self, data_val):
        # remove node with node.data = data_val, if it's there.
        # Assume there's only one (for vineyard context, this is true)
        # Used for setting entries of a sparseMatrix equal to zero.
        if self.head is not None:
            curr = self.head
            if curr.data == data_val:
                self.head = curr.next
            else:
                while(curr.next is not None and not curr.next.data == data_val):
                    curr = curr.next
                # curr is now either the last node or the node right before the 
                # node whose data is data_val
                if not curr.next is None:
                    curr.next = curr.next.next

class sparseMatrix:
    '''
    Represents sparse binary matrices.
    '''
    def __init__(self, n, col = True, print_progress = False):
        # Initialize empty nxn sparse matrix. Use col = True if you want to add
        # columns quickly, or col = False if you want to add rows quickly.
        self.reduced = False
        self.print_progress = print_progress
        self.n = n
        
        # If col = False, will store the transpose of A.
        self.col = col
        
        # col_array[i] is the index in col_list of the linked list that represents column i in the current matrix
        self.col_array = [j for j in range(n)]
        
        # row_array[i] represents the original ith row. It stores its current 
        # index in the matrix and also stores the reverse link, i.e. row_array[i] is
        # a tuple s.t. row_array[i][0] = current index of the original ith row,
        # and row_array[i][1] = original index of what is now the ith row
        self.row_array = [[i, i] for i in range(n)]

        # Initialize empty linked list for each column. 
        # col_list[i] stores what was originally column i. The nodes in the list
        # have values equal to the ORIGINAL row indices with non-zero entries.
        # If col = False, should just store the transpose and remember later when adding rows/cols, etc
        self.col_list = [linkedList() for i in range(n)]

    def __eq__(self, other):
        # Returns True if self and other represent the same underlying matrix
        assert self.col and other.col, "haven't implemented for col = False yet"
        if self.n != other.n:
            return False
        for j in range(self.n):
            self_entries = self.get_col_nonzero_entries(j)
            other_entries = other.get_col_nonzero_entries(j)
            if not set(self_entries) == set(other_entries):
                print("col ", j, " is incorrect")
                print(self_entries)
                print(other_entries)
                return False
        return True        
    
    def __getitem__(self, idx):
        # Returns A_{ij} where i = idx[0] and j = idx[1]
        i = idx[0]
        j = idx[1]
        if self.col:
            col_j_list = self.get_col(j)
            orig_i_idx = self.row_array[i][1]
            if col_j_list.has(orig_i_idx):
                return 1
            else:
                return 0
        else:
            row_i_list = self.get_row(i)
            orig_j_idx = self.row_array[j][1]
            if row_i_list.has(orig_j_idx):
                return 1
            else:
                return 0
                
    def __mul__(self, other):
        assert other.n == self.n, "matrix dimensions have to agree"
        assert self.col and not other.col, "haven't implemented when self.col = False or when other.col = True"
        prod = sparseMatrix(self.n)
        for k in range(self.n):
            col = self.get_col(k)
            row = other.get_row(k)
            col_curr = col.head
            while(col_curr is not None):
                i = self.get_row_idx(col_curr)
                row_curr = row.head
                while(row_curr is not None):
                    j = other.get_col_idx(row_curr)
                    prod.plus_1(i, j)
                    row_curr = row_curr.next
                col_curr = col_curr.next
        return prod
                    
    def add_col(self, i, j):
        # Add what is currently column i to what is currently column j.
        assert self.col, "Can't add columns if col = False"
        
        # Update inverse_low dictionary
        if self.reduced:
            low_j = self.low(j)
            if low_j != -1:
                if low_j != -1 and len(self.inverse_low[low_j]) == 1:
                    del self.inverse_low[low_j]
                else:
                    self.inverse_low[low_j].remove(j)
                    
            col_j = self.get_col(j)
            col_j += self.get_col(i)
            
            # Finish updating inverse_low
            new_low_j = self.low(j)
            if new_low_j != -1:
                if new_low_j in self.inverse_low:
                    self.inverse_low[new_low_j].append(j)
                else:
                    self.inverse_low.update({new_low_j : [j]})
        else:
            col_j = self.get_col(j)
            col_j += self.get_col(i)
            
    def add_row(self, i, j):
        # Add row i to row j. Can only be called if col = False. In that case
        # we're actually storing the transpose, so add column i to column j
        # as in add_col
        assert not self.col, "Haven't implemented add_row if col = True"
        row_j = self.get_row(j)
        row_j += self.get_row(i)
    
    def copy(self):
        # Returns a sparseMatrix that represents the same underlying matrix. Deep copy
        copy = sparseMatrix(self.n, col = self.col)
        copy.col_array = self.col_array.copy()
        copy.row_array = [pair.copy() for pair in self.row_array]
        for i in range(self.n):
            # Copy what's currently ith column (or row) in A to copy.col_list[i]
            self_curr = self.col_list[i].head
            if self_curr is not None: 
                copy.col_list[i].head = Node(self_curr.data)
                copycurr = copy.col_list[i].head
                while self_curr.next is not None:
                    copycurr.next = Node(self_curr.next.data)
                    self_curr = self_curr.next
                    copycurr = copycurr.next
        return copy
        
    def get_col(self, i):
        # Gets the current ith col, stored as linked list
        assert self.col, "Haven't implemented get_col for col = False"
        return self.col_list[self.col_array[i]]
    
    def get_col_idx(self, node):
        # Get current col idx of a node. Only call if self.col = False
        assert not self.col, "Haven't implemented get_col_idx for col = True"
        if node is None:
            return None
        else:
            return self.row_array[node.data][0]
    
    def get_col_nonzero_entries(self, i):
        # Get the (current) row indices of the nonzero entries in col i. Return list
        if self.col:
            row_indices = []
            curr = self.get_col(i).head
            while curr is not None:
                row_indices.append(self.get_row_idx(curr))
                curr = curr.next
            return row_indices
        else:
            # This is slow if col = False
            row_indices = []
            # Check each row to see if there's a node in col i
            for j in range(self.n):
                curr = self.get_row(j).head
                while curr is not None:
                    if self.get_col_idx(curr) == i:
                        row_indices.append(j)
                        break
                    curr = curr.next
            return row_indices
                          
    def get_row_nonzero_entries(self, i):
        if self.col:
            # Slow if col = True.
            col_indices = []
            for j in range(self.n):
                curr = self.get_col(j).head
                while curr is not None:
                    if self.get_row_idx(curr) == i:
                        col_indices.append(j)
                        break
                    curr = curr.next
            return col_indices
        else:
            col_indices = []
            curr = self.get_row(i).head
            while curr is not None:
                col_indices.append(self.get_col_idx(curr))
                curr = curr.next
            return col_indices
        
    def get_row(self, i):
        # Gets the current ith row, stored as linked list
        assert not self.col, "Have't implemented for col = True"
        return self.col_list[self.col_array[i]]
    
    def get_row_idx(self, node):
        # Get current row index of a node.
        assert self.col, "Haven't implemented for self.col = False"
        if node is None:
            return None
        else:
            return self.row_array[node.data][0]

    def identity(n, col= True):
        # Returns sparseMatrix representation of the nxn identity matrix
        I = sparseMatrix(n, col = col) # zero matrix
        for j in I.col_array:
            I.col_list[j].append(Node(j))
        return I

    def is_positive(self, j):
        # Returns True if column j is a zero column (correspondingly, simplex j
        # is positive), false otherwise. Only works when col = True because in 
        # the vineyard context, we only ever apply this for R.
        return self.col_list[self.col_array[j]].head is None
    
    def is_upper(self):
        if self.col:
            for i in range(self.n):
                col = self.get_col(i)
                curr = col.head
                while(curr is not None):
                    if self.get_row_idx(curr) > i:
                        return False
                    curr = curr.next
            return True
        else:
            for i in range(self.n):
                row = self.get_row(i)
                curr = row.head
                while(curr is not None):
                    if self.get_col_idx(curr) < i:
                        return False
                    curr = curr.next
            return True
    
    def plus_1(self, i, j):
        # Only ever gets called when multiplying matrices, not when the matrix is reduced.
        # Does NOT preserve the self.inverse_low mapping
        assert self.col, "Haven't implemented for self.col = False"
        old_entry = self[i, j]
        curr = self.get_col(j).head
        orig_i = self.row_array[i][1]
        if curr is None:
            self.get_col(j).head = Node(orig_i)
        elif curr.data == orig_i:
            self.get_col(j).head = curr.next
        elif curr.data > orig_i:
            new_head = Node(orig_i)
            new_head.next = curr
            self.get_col(j).head = new_head
        else:
            while(curr.next is not None and curr.next.data < orig_i):
                curr = curr.next
            # Now curr is the last node s.t. curr.data < orig_i
            if curr.next is not None and curr.next.data == orig_i:
                # Then remove curr.next from list
                curr.next = curr.next.next
            else:
                new_node = Node(orig_i)
                new_node.next = curr.next
                curr.next = new_node

    def swap_cols(self, i, j):
        # Exchange cols i and j. Implementation depends on if col = True.
        if self.col:
            if self.reduced:
                low_i = self.low(i)
                low_j = self.low(j)
                if low_i != -1:
                    self.inverse_low[low_i].remove(i)
                    self.inverse_low[low_i].append(j)
                if low_j != -1:
                    self.inverse_low[low_j].remove(j)
                    self.inverse_low[low_j].append(i)
                    
                i_idx = self.col_array[i]
                self.col_array[i] = self.col_array[j]
                self.col_array[j] = i_idx
            else:  
                i_idx = self.col_array[i]
                self.col_array[i] = self.col_array[j]
                self.col_array[j] = i_idx
        else:
            # exhange the "rows" i and j as in swap_rows(i, j), because 
            # we're actually storing the transpose
            i_orig_idx = self.row_array[i][1]
            j_orig_idx = self.row_array[j][1]
            self.row_array[i_orig_idx][0] = j # current index of j_orig_idx
            self.row_array[j_orig_idx][0] = i # current index of i_orig_idx
            self.row_array[i][1] = j_orig_idx
            self.row_array[j][1] = i_orig_idx
    
    def swap_rows(self, i, j):
        # Exchange what are currently rows i and j. Implementation depends on if col = True. 
        if self.col:
            if self.reduced:
                k_list = []
                if i in self.inverse_low:
                    k_list = self.inverse_low[i]
                if (i+1) in self.inverse_low:
                    k_list = k_list + self.inverse_low[i+1]
                k_list = set(k_list)    # set of cols k s.t. low(k) = i or i+1
                old_lows = {k : self.low(k) for k in k_list}    # every self.low(k) is either i or i+1
                
                i_orig_idx = self.row_array[i][1]
                j_orig_idx = self.row_array[j][1]
                self.row_array[i_orig_idx][0] = j # current index of j_orig_idx
                self.row_array[j_orig_idx][0] = i # current index of i_orig_idx
                self.row_array[i][1] = j_orig_idx
                self.row_array[j][1] = i_orig_idx
                
                # Update inverse_low after swap
                for k in k_list:
                    new_low_k = self.low(k) # either i or i+1
                    if new_low_k != old_lows[k]:
                        if len(self.inverse_low[old_lows[k]]) == 1:
                            del self.inverse_low[old_lows[k]]
                        else:
                            self.inverse_low[old_lows[k]].remove(k)
                        if new_low_k in self.inverse_low:
                            self.inverse_low[new_low_k].append(k)
                        else:
                            self.inverse_low.update({new_low_k : [k]})
            else:
                i_orig_idx = self.row_array[i][1]
                j_orig_idx = self.row_array[j][1]
                self.row_array[i_orig_idx][0] = j # current index of j_orig_idx
                self.row_array[j_orig_idx][0] = i # current index of i_orig_idx
                self.row_array[i][1] = j_orig_idx
                self.row_array[j][1] = i_orig_idx
        else:
            # exchange the "columns" i and j as in swap_cols, because
            # we're actually storing the transpose
            i_idx = self.col_array[i]
            self.col_array[i] = self.col_array[j]
            self.col_array[j] = i_idx
    
    def low(self, j):
        # Returns current row index of lowest 1 in what is currently column j.
        # In vineyard context, only gets called for R, so this code only works
        # if col = True. Returns -1 if there are no 1s in col j. (MAY WANT TO CHANGE THIS TO NONE)
        if self.col:
            col_j_list = self.col_list[self.col_array[j]]
            curr_node = col_j_list.head
            curr_max = -1
            while curr_node is not None:
                curr_row = self.row_array[curr_node.data][0]
                if curr_row > curr_max:
                    curr_max = curr_row
                curr_node = curr_node.next
            return curr_max

    def print(self):
        for i in range(self.n):
            for j in range(self.n):
                print(self[i, j], end = "\t")
            print("\n")
            
    def reduce(self):
        '''
        Only ever gets called for D, so only needs to work when col = True.
        Implements simplex pairing reduction algorithm from paper. D gets reduced to R.
        Returns the matrix U s.t. if D is the original matrix and R is the reduced matrix,
        then D = RU where U is upper-triangular.
        When computing U, I'm making the assumption that D and R are square (which is true in the vineyard context)
        '''
        assert self.col, "Only implemented for self.col = True"
        if self.print_progress: print("started reducing")
        U = sparseMatrix.identity(self.n, col = False)
        self.inverse_low = {}
        for j in range(self.n):
            if self.print_progress: print(f"col {j}")
            i = self.low(j)
            while i in self.inverse_low:
                j_ = self.inverse_low[i][0]
                #if self.print_progress: print(f"adding col {j_} to {j}")
                self.add_col(j_, j)
                U.add_row(j, j_)
                i = self.low(j)
            if i != -1:
                self.inverse_low.update({i : [j]})
        self.reduced = True
        return U
            
    def set_zero(self, i, j):
        '''
        Set A_{ij} equal to 0, if it isn't already. To do this: If col = True, 
        remove the node with value row_array[i][1] (if it's there) from the linked list 
        representing what is currently column j. If col = False, remove the node
        with value row_array[j][1] (if it's there) from the linked list
        representing what is currently row i.
        '''
        if self.col:
            col_j_list = self.col_list[self.col_array[j]]
            col_j_list.remove(self.row_array[i][1])
        else:
            row_i_list = self.col_list[self.col_array[i]]
            row_i_list.remove(self.row_array[j][1])
            
class RU:
    def __init__(self, D, debug_mode = False, print_progress = False):
        # D is a square nxn sparseMatrix that represents the boundary matrix
        self.print_progress = print_progress
        self.debug_mode = debug_mode
        if debug_mode: self.D = D.copy()   # Need to copy D before it gets reduced
            
        self.U = D.reduce() # The reduce function reduces D in place to R and returns the upper triangular matrix U s.t. D = R*U
        self.R = D
        self.n = len(D.col_array)
        
        if debug_mode: 
            self.check_decomposition()
            print("initial reduce is correct")
        
    def check_decomposition(self, i = None, case = None):
        if i is not None and case is not None:
            additional_info = f" after swapping {i} and {i+1} (case {case})"
        elif i is not None:
            additional_info = f" after swapping {i} and {i+1}"
        elif case is not None:
            additional_info = f" after a case {case} swap"
        else:
            additional_info = ""
        assert self.D.is_upper(), "D isn't upper triangular anymore (not a filtration)"+additional_info
        assert self.U.is_upper(), "U isn't upper triangular anymore"+additional_info
        assert self.is_R_reduced(), "R isn't reduced anymore"+additional_info
        assert self.D == self.R*self.U, "RU != D"+additional_info
            
    def is_R_reduced(self):
        pairs = {}
        for j in range(self.n):
            i = self.R.low(j)
            if i != -1:
                if i in pairs:
                    return False
                else:
                    pairs.update({i : j})
        return True
            
    def swap(self, i, dim_i, dim_iplus):
        '''
        If P is the permutation matrix that swaps i, i+1, then PDP is the matrix
        obtained by swapping rows i, i+1 and cols i, i+1 of D. Update R, U
        to be a decomposition for PDP.
        dim_i = dimension of current ith simplex
        dim_iplus = dimension of current (i+1)st simplex
        Returns True if we need to update the simplex pairings in the vineyard, False otherwise.
        '''
        need_update = False
        
        if self.debug_mode:
            self.D.swap_cols(i, i+1)
            self.D.swap_rows(i, i+1)
            
        if dim_i != dim_iplus:
            case = "0"
            self.R.swap_cols(i, i+1)
            self.R.swap_rows(i, i+1)
            self.U.swap_cols(i, i+1)
            self.U.swap_rows(i, i+1)
        else:    
            i_pos = self.R.is_positive(i)
            iplus_pos = self.R.is_positive(i+1)
            
            if i_pos and iplus_pos:
                # Case 1: i and i+1 are positive simplices
                # In this case, can set U_{i, i+1} = 0 and still have an RU decomp.
                
                self.U.set_zero(i, i+1)
                
                # Now PUP will be upper triangular.
                self.U.swap_rows(i, i+1)
                self.U.swap_cols(i, i+1)
                
                # Figure out if we're in Case 1.1: There are cols k, l s.t. 
                # low_R(k) = i and low_R(l) = i+1 and R[i, l] = 1.
                case11 = False
                if (i+1) in self.R.inverse_low:
                    l = self.R.inverse_low[i+1][0]
                    if self.R[i, l] == 1 and i in self.R.inverse_low:
                        k = self.R.inverse_low[i][0]
                        case11 = True
                        
                if case11:
                    # Case 1.1. Now compute PRP
                    self.R.swap_rows(i, i+1)
                    self.R.swap_cols(i, i+1)
                    
                    # Reduce PRP.
                    if k < l:
                        # Case 1.1.1
                        case = "1.1.1"
                        self.R.add_col(k, l)    # add col k to col l in PRP
                        self.U.add_row(l, k)    # add row l to row k in PUP
                    if l < k:
                        # Case 1.1.2
                        case = "1.1.2"
                        need_update = True
                        self.R.add_col(l, k)
                        self.U.add_row(k, l)
                else:
                    # Case 1.2. Just compute PRP and we're done.
                    case = "1.2"
                    self.R.swap_rows(i, i+1)
                    self.R.swap_cols(i, i+1)
                    
            elif not i_pos and not iplus_pos:
                # Case 2: i and i+1 are both negative simplices. In this case, rows
                # i and i+1 can't contain the lowest 1s of an columns, so PRP is 
                # reduced. Just need to fix PUP.
                if self.U[i, i+1] == 1:
                    # Case 2.1
                    if self.R.low(i) < self.R.low(i+1):
                        # Case 2.1.1
                        case = "2.1.1"
                        self.R.add_col(i, i+1)
                        self.R.swap_rows(i, i+1)
                        self.R.swap_cols(i, i+1)
                        
                        self.U.add_row(i+1, i)
                        self.U.swap_rows(i, i+1)
                        self.U.swap_cols(i, i+1)
                    else:
                        # Case 2.1.2: self.R.low(i) > self.R.low(i+1)
                        case = "2.1.2"
                        need_update = True
                        self.R.add_col(i, i+1)
                        self.R.swap_rows(i, i+1)
                        self.R.swap_cols(i, i+1)
                        self.R.add_col(i, i+1)
                        
                        self.U.add_row(i+1, i)
                        self.U.swap_rows(i, i+1)
                        self.U.swap_cols(i, i+1)
                        self.U.add_row(i+1, i)
                else:
                    # Case 2.2
                    case = "2.2"
                    self.R.swap_cols(i, i+1)
                    self.R.swap_rows(i, i+1)
                    
                    self.U.swap_cols(i, i+1)
                    self.U.swap_rows(i, i+1)
                    
            elif not i_pos and iplus_pos:
                # Case 3: i is a negative simplex and i+1 is a postive simplex
                if self.U[i, i+1] == 1:
                    # Case 3.1
                    case = "3.1"
                    need_update = True
                    self.R.add_col(i, i+1)
                    self.R.swap_rows(i, i+1)
                    self.R.swap_cols(i, i+1)
                    self.R.add_col(i, i+1)
                    
                    self.U.add_row(i+1, i)
                    self.U.swap_rows(i, i+1)
                    self.U.swap_cols(i, i+1)
                    self.U.add_row(i+1, i)
                else:
                    # Case 3.2
                    case = "3.2"
                    self.R.swap_cols(i, i+1)
                    self.R.swap_rows(i, i+1)
                    
                    self.U.swap_cols(i, i+1)
                    self.U.swap_rows(i, i+1)
            else:
                # Case 4: i is a positive simplex and i+1 is a negative simplex
                case = "4"
                self.U.set_zero(i, i+1)
                self.R.swap_cols(i, i+1)
                self.R.swap_rows(i, i+1)
                
                self.U.swap_cols(i, i+1)
                self.U.swap_rows(i, i+1)
        
        if self.debug_mode: self.check_decomposition(i, case)
        return need_update

class Simplex:
    # Represents a simplex with a sequence of filtration values
    def __init__(self, nodes, val, name = None):
        # nodes is the list of nodes (ints) that make up the simplex
        # val is a list of filtration values (floats)
        # name is optional string. In PolygonalSurface context, could be the name of the region that the simplex came from.
        nodes.sort()
        self.nodes = tuple(nodes)
        self.val = val
        self.curr_idx = None
        self.dim = len(nodes)-1
        self.name = name
    
    def __lt__(self, other):
        '''
        Compares by initial filtration value. If they're equal, keep going 
        until you find the first different value. If all are equal, return True
        if dim(self) <= dim(other), return False otherwise.
        '''
        i = 0
        while i < len(self.val) and self.val[i] == other.val[i]:
            i += 1
        if i < len(self.val):
            return self.val[i] < other.val[i]
        else:
            return self.dim <= other.dim
    
    def __getitem__(self, t):
        # Get the filtration value at time t, where we interpolate linearly between integer times.
        frac_t = t%1
        int_t = int(t - frac_t)
        if frac_t == 0:
            return self.val[int_t]
        else:
            left_y = self.val[int_t]
            right_y = self.val[int_t +1]
            return (1-frac_t)*left_y + frac_t*right_y
    
    def intersection(self, other_spx, t, curr_timestep):
        '''
        Returns (t_intersect, y_intersect) coordinates of the intersection 
        of self and other_spx, if one exists in the time interval [t, curr_timestep+1).
        If the line segments are parallel along the interval (no intersection
        or infinite intersections), return None. If the line segments intersect
        but it's not in the interval, return None.
        '''
        denom = other_spx[curr_timestep + 1] - other_spx[curr_timestep] + self[curr_timestep] - self[curr_timestep + 1]
        if denom == 0:
            return None
        else:
            t_intersect = (self[curr_timestep] - other_spx[curr_timestep])/denom + curr_timestep
            if t_intersect >= t and t_intersect < curr_timestep + 1:
                y_intersect = (1- t_intersect%1)*self[curr_timestep]  + (t_intersect%1)*self[curr_timestep + 1]
                return [t_intersect, y_intersect]
            else:
                return None
    
    def is_degenerate(self):
        # Returns True if there are any duplicate nodes, False otherwise.
        for i in range(self.dim+1):
            for j in range(i+1, self.dim+1):
                if self.nodes[i] == self.nodes[j]: return True
        return False
    
    def __str__(self):
        return f"Nodes: {' '.join([str(node) for node in self.nodes])}\nName: {self.name}\nFiltration values: {' '.join([str(value) for value in self.val])}\nCurrent index: {self.curr_idx}\n"
        
class Vine():
    def __init__(self, T, spx1, spx2= None):
        # spx1 is the initial birth simplex for this homological generator, spx2 is initial death simplex
        # T is the end time of the vine.
        self.pairs = [(spx1, spx2)]
        self.times = [0, T] # pairing change times
        self.T = T
        
        self.computed_vertices = False
        self.births = None
        self.deaths = None
        self.vertex_times = None
        self.dist_from_diag = None
    
    def __getitem__(self, t):
        if not self.computed_vertices: self.compute_vertices()
        start = 0
        while self.vertex_times[start + 1] < t:
            start += 1
        end = start + 1
        t0 = self.vertex_times[start]
        t1 = self.vertex_times[end]
        b0 = self.births[start]
        b1 = self.births[end]
        d0 = self.deaths[start]
        d1 = self.deaths[end]
        w = (t - t0)/(t1 - t0)
        b = (1-w)*b0 + w*b1
        if d0 == np.inf or d1 == np.inf:
            d = np.inf
        else:
            d = (1-w)*d0 + w*d1
        return b, d
    
    def append_pairing(self, spx1, spx2, t):
        self.pairs.append((spx1, spx2))
        self.times[-1] = t
        self.times.append(self.T)
        self.computed_vertices = False
        self.dist_from_diag = None
    
    def compute_vertices(self):
        # Get the vertices of the linear vines.
        # We linearly interpolate between these vertices to plot the full vine.
        if not self.computed_vertices:
            births = []
            deaths = []
            vertex_times = []
            for i, ti in enumerate(self.times[:-1]):
                spx_b = self.pairs[i][0]
                spx_d = self.pairs[i][1]
                next_ti = self.times[i+1]
                if not ti%1 == 0:
                    vertex_times.append(ti)
                    births.append(spx_b[ti])
                    if spx_d is None:
                        deaths.append(np.inf)
                    else:
                        deaths.append(spx_d[ti])
                for s in range(math.ceil(ti), math.ceil(next_ti)):
                    # If next_ti is an int, gets all ints s s.t. ceil(ti) <= s <= next_ti - 1. If not an int, gets all ints s s.t. ceil(ti) <= s <= floor(next_ti)
                    vertex_times.append(s)
                    births.append(spx_b[s])
                    if spx_d is None:
                        deaths.append(np.inf)
                    else:
                        deaths.append(spx_d[s])
            T = self.times[-1]
            vertex_times.append(T)
            births.append(spx_b[T])
            if spx_d is None:
                deaths.append(np.inf)
            else:
                deaths.append(spx_d[T])
            self.births = births
            self.deaths = deaths
            self.vertex_times = vertex_times
            self.computed_vertices = True
    
    def dist_from_diag_at_time(self, t):
        # Vine's distance from diagonal plane at time t
        b, d = self[t]
        if d == np.inf:
            return np.inf
        else:
            return (d - b)/np.sqrt(2)
        
    def dist_from_diag_at_vertex(self, i):
        # Vine's distance from diagonal plane at vertex i
        if not self.computed_vertices: self.compute_vertices()
        if self.deaths[i] == np.inf:
            return np.inf
        else:
            return (self.deaths[i] - self.births[i])/np.sqrt(2)
    
    def get_dist_from_diag(self, start_timestep = 0, end_timestep = None):
        if self.dist_from_diag is None or start_timestep != 0 or end_timestep is not None:
            times, births, deaths = self.get_vertices(start_timestep, end_timestep)
            num_edges = len(times)-1
            dist = 0
            for i in range(num_edges):
                t0 = times[i]
                t1 = times[i+1]
                dist += (t1 - t0)*(self.dist_from_diag_at_time(t0) + self.dist_from_diag_at_time(t1))/2
            if start_timestep == 0 and end_timestep is None:
                self.dist_from_diag = dist
        else:
            dist = self.dist_from_diag
        return dist
    
    def get_edge_labels_by_death(self, vertex_times = None):
        if not self.computed_vertices: self.compute_vertices()
        edge_labels = []
        j = 0
        if vertex_times is None: vertex_times = self.vertex_times
        for i, pair in enumerate(self.pairs):
            if pair[1] is None:
                spx_d_region = None
            else:
                spx_d_region = pair[1].name
            pair_end_time = min(self.times[i+1], vertex_times[-1])
            while(vertex_times[j] < pair_end_time):
                edge_labels.append(spx_d_region)
                j += 1
        return edge_labels
    
    def get_vertices(self, start_timestep = 0, end_timestep = None):
        if not self.computed_vertices: self.compute_vertices()
        if start_timestep == 0 and (end_timestep is None or end_timestep == self.T):
            return self.vertex_times, self.births, self.deaths
        if end_timestep is None:
            end_timestep = self.T
        indices = [i for i, t in enumerate(self.vertex_times) if start_timestep < t < end_timestep]
        start_idx = indices[0]
        end_idx = indices[-1] + 1
        times = [start_timestep] + self.vertex_times[start_idx:end_idx] + [end_timestep]
        startbirth, startdeath = self[start_timestep]
        endbirth, enddeath = self[end_timestep]
        births = [startbirth] + self.births[start_idx:end_idx] + [endbirth]
        deaths = [startdeath] + self.deaths[start_idx:end_idx] + [enddeath]
        return times, births, deaths
        
    def get_primary_death_name(self):
        # Get name of region that spends the longest amount of time (weighted by distance from diagonal plane) as peak.
        assert self.get_dist_from_diag != np.inf, "This vine doesn't die"
        region_caseduration = {}
        if not self.computed_vertices: self.compute_vertices()
        j = 0
        for i, pair in enumerate(self.pairs):
            spx_d_region = pair[1].name
            pair_end_time = self.times[i+1]
            while(self.vertex_times[j] < pair_end_time):
                # Area between vine and diagonal plane between jth and (j+1)st vertex
                caseduration = (self.vertex_times[j+1] - self.vertex_times[j])*(self.dist_from_diag_at_vertex(j) + self.dist_from_diag_at_vertex(j+1))/2
                if spx_d_region in region_caseduration:
                    region_caseduration[spx_d_region] += caseduration
                else:
                    region_caseduration[spx_d_region] = caseduration
                j += 1
        return max(region_caseduration, key = region_caseduration.get)

    def get_death_names(self):
        # Return all region names that represent this vine across time.
        return {
            pair[1].name
            for pair in self.pairs
            if pair[1] is not None and pair[1].name is not None
        }
        
    def print(self):
        for pair in self.pairs:
            print(f"{pair[0].nodes} \t {pair[1].nodes}")
        print("times : ", self.times, "\n")

class Vineyard():
    def __init__(self, simplices, dim, debug_mode = False, print_progress = False):
        self.debug_mode = debug_mode
        self.print_progress = print_progress
        if print_progress: print("Started vineyard")
        
        self.curr_timestep = 0
        self.dim = dim # homology dimension
        simplices.sort()
        self.simplices = simplices
        self.n = len(simplices)
        self.T = len(simplices[0].val)-1    # max time in the filtration
        if print_progress: print(f"{self.n} simplices")
        
        # Initialize RU decomposition to rep initial boundary matrix
        C = {}
        for i in range(self.n):
            if self.debug_mode:
                assert not self.simplices[i].is_degenerate(), f"{i}: {self.simplices[i].nodes} is a degenerate simplex"
            C.update({self.simplices[i].nodes : i})
            self.simplices[i].curr_idx = i # Initialize current index of each simplex
        bdry_matrix = sparseMatrix(self.n, print_progress = print_progress)
        for spx in C:
            dim = len(spx) - 1
            if not dim == 0:
                bdry = bdry_matrix.col_list[C[spx]]
                for j in range(dim + 1):
                    face = tuple([spx[k] for k in range(dim+1) if not k == j])
                    bdry.insert(Node(C[face]))
        if print_progress: print("Calculated boundary matrix")
        self.curr_RU = RU(bdry_matrix, debug_mode)
        if print_progress: print("Reduced boundary matrix")
        
        self.spx_to_vine = {}   # Simplex object --> Vine object s.t. if v = spx_to_vine{spx}, then spx is one of the simplices in the pairing that currently represents the vine v
        self.vines = [] # list of Vine objects
        
        if debug_mode and self.T < 2:
            self.are_curr_pairs_correct()
            
        # Initialize the vines
        pairs = self.curr_pairs()
        for i in pairs:
            j = pairs[i]
            if j is not None:
                self.vines.append(Vine(self.T, self.simplices[i], self.simplices[j]))
                self.spx_to_vine.update({self.simplices[j] : self.vines[-1]})
            else:
                self.vines.append(Vine(self.T, self.simplices[i]))
            self.spx_to_vine.update({self.simplices[i] : self.vines[-1]})
        if self.print_progress: print("Initialized vines")    
        
        # Initialize event queue
        self.eq = PriorityQueue()
        for t in range(0, self.T):
            self.eq.put(Event(t, event_type = "timestep"))
                
        while not self.eq.empty():
            e = self.eq.get()
            self.handle(e)
        # For now, don't do anything with intersections that happen at time T.
        # This is fine because if two simplices intersect at time T, the ordering 
        # is still correct with or without the swap.

    def are_curr_pairs_correct(self):
        # Test whether the current simplex pairings are correct by comparing to what gudhi does
        # Current pairs, as calculated by SimplicialComplex
        sc_pair_indices = self.curr_pairs()
        sc_pairs = {} # Stores the paired simplices, where each simplex is stored as a tuple of nodes
        for i in sc_pair_indices:
            j = sc_pair_indices[i]
            if j is None:
                sc_pairs.update({tuple(sorted(self.simplices[i].nodes)) : ()})
            else:
                sc_pairs.update({tuple(sorted(self.simplices[i].nodes)) : tuple(sorted(self.simplices[j].nodes))})

        # Current pairs, as calculated by gudhi
        st = self.toSimplexTree()
        st.compute_persistence(homology_coeff_field = 2)
        all_pairs = st.persistence_pairs()
        dim_pairs = {}
        for pair in all_pairs:
            if len(pair[0]) == self.dim + 1:
                dim_pairs.update({tuple(sorted(pair[0])) : tuple(sorted(pair[1]))})
        
        # Test if the pairs are the same
        pairs_correct = True
        for b_spx in sc_pairs:
            d_spx = dim_pairs.pop(b_spx, None)
            if d_spx != sc_pairs.get(b_spx):
                print("\nIncorrect pair:", b_spx, sc_pairs[b_spx])
                print("Correct pair: ", b_spx, d_spx)
                pairs_correct = False
        
        # Return True if dim_pairs is empty, False otherwise.
        for b_spx in dim_pairs:
            print("Missing (correct) pair: ", b_spx, dim_pairs[b_spx])
        return (not dim_pairs and pairs_correct)
    
    def curr_pairs(self):
        pairs = {}
        for j in range(self.n):
            dim_spx_j = self.simplices[j].dim
            if self.curr_RU.R.is_positive(j) and dim_spx_j == self.dim:
                pairs.update({j: None})
            else:
                i = self.curr_RU.R.low(j)
                if dim_spx_j == self.dim+1:
                    pairs.update({i : j})
        return pairs
        
    def handle(self, event):
        if event.event_type == "activate":
            lower_nbr = self.nbr_lower(event.spx)
            if lower_nbr is not None:
                intersection = event.spx.intersection(lower_nbr, event.t, self.curr_timestep)
                '''
                The activate events are ordered s.t. event.curr_idx > lower_nbr.curr_idx.
                This implies event.curr_idx[event.t] >= lower_nbr[event.t].
                If event.spx[event.t + 1] > lower_nbr[event.t + 1], then 
                event.spx >= lower_nbr on the whole interval [event.t = self.curr_timestep, event.t + 1],
                so they're already in the correct order. So do nothing. Otherwise...
                '''
                if (intersection is not None and event.spx[event.t + 1] < lower_nbr[event.t + 1]):
                    self.eq.put(CrossingEvent(lower_nbr, event.spx, intersection[0], intersection[1]))
        if event.event_type == "crossing" and event.spx1.curr_idx == event.spx2.curr_idx-1:
            '''
            Checking spx1.curr_idx == spx2.curr_idx-1 is necessary when there 
            are multiple crossings at the same coordinates. This happens often 
            at the beginning of time intervals. Checking spx1.curr_idx < spx2.curr_idx 
            also ensures we don't perform the same crossing more than once. (The 
            same crossing can be added to the event queue more than once in some 
            situations). Assumes that spx1.curr_idx was less than spx2.curr_idx 
            when the crossing event was added.
            '''
            if self.debug_mode:
                print("crossing: ", event.spx1.nodes, f"(curr_idx = {event.spx1.curr_idx})", event.spx2.nodes, f"(curr_idx = {event.spx2.curr_idx})", "time = ", event.t, "y = ", event.y)
            if self.print_progress: print("time = ", event.t)
            # Get any vines that are involved with these simplices
            v1 = self.spx_to_vine.get(event.spx1)
            v2 = self.spx_to_vine.get(event.spx2)
            
            if v1 is not None:
                pair = v1.pairs[-1]
                if pair[1] is None:
                    v1_idx = [pair[0].curr_idx, None]
                else:
                    v1_idx = [pair[0].curr_idx, pair[1].curr_idx]
            if v2 is not None:
                pair = v2.pairs[-1]
                if pair[1] is None:
                    v2_idx = [pair[0].curr_idx, None]
                else:
                    v2_idx = [pair[0].curr_idx, pair[1].curr_idx]
            
            # Swap the simplices. Assume that spx1.curr_idx < spx2.curr_idx before the crossing.
            # If debug_mode, this will also check if the RU decomposition is still correct after the swap.
            need_update = self.swap(event.spx1.curr_idx) and (v1 is not None or v2 is not None)
            
            if self.debug_mode:
                spx1_info = f"{event.spx1.nodes} values on interval: {event.spx1[self.curr_timestep]}, {event.spx1[self.curr_timestep + 1]}"
                spx2_info = f"{event.spx2.nodes} values on interval: {event.spx2[self.curr_timestep]}, {event.spx2[self.curr_timestep + 1]}"
                assert self.is_filt_nondecreasing(event.t), "filtration isn't nondecreasing (as a function of curr_idx) anymore\n" + spx1_info + "\n"+spx2_info
                
            if self.debug_mode: assert self.are_curr_pairs_correct()
    
            # Update the vines if needed
            if need_update:
                if self.debug_mode or self.print_progress: print("updating vines after crossing", event.spx1.nodes, event.spx2.nodes, "time = ", event.t)
                if v1 is not None:
                    if v1_idx[1] is None:
                        v1.append_pairing(self.simplices[v1_idx[0]], None, event.t)
                    else:
                        v1.append_pairing(self.simplices[v1_idx[0]], self.simplices[v1_idx[1]], event.t)
                    self.spx_to_vine.update({event.spx2 : v1})
                if v2 is not None:
                    if v2_idx[1] is None:
                        v2.append_pairing(self.simplices[v2_idx[0]], None, event.t)
                    else:
                        v2.append_pairing(self.simplices[v2_idx[0]], self.simplices[v2_idx[1]], event.t)
                    self.spx_to_vine.update({event.spx1 : v2})
            
            # Check for new crossing events with new neighbors. spx1 checks its
            # new upper neighbor, and spx2 checks its new lower neighbor.
            spx1_up_nbr = self.nbr_upper(event.spx1)
            if spx1_up_nbr is not None:
                spx1_up_intersect = event.spx1.intersection(spx1_up_nbr, event.t, self.curr_timestep)
                if spx1_up_intersect is not None and event.spx1[self.curr_timestep + 1] > spx1_up_nbr[self.curr_timestep+ 1]:
                    self.eq.put(CrossingEvent(event.spx1, spx1_up_nbr, spx1_up_intersect[0], spx1_up_intersect[1]))
            spx2_low_nbr = self.nbr_lower(event.spx2)
            if spx2_low_nbr is not None:
                spx2_low_intersect = event.spx2.intersection(spx2_low_nbr, event.t, self.curr_timestep)
                if spx2_low_intersect is not None and spx2_low_nbr[self.curr_timestep + 1] > event.spx2[self.curr_timestep + 1]:
                    self.eq.put(CrossingEvent(spx2_low_nbr, event.spx2, spx2_low_intersect[0], spx2_low_intersect[1]))
        if event.event_type == "timestep":
            self.curr_timestep = event.t
            if self.debug_mode or self.print_progress: print(f"TIMESTEP = {self.curr_timestep}")
            for spx in self.simplices:
                self.eq.put(ActivateEvent(spx, event.t))
                
    def is_filt_nondecreasing(self, t):
        # Check that if spx1.curr_idx < spx2.curr_idx, then spx1[t] <= spx2[t]
        eps = .00001
        for i in range(self.n-1):
            if self.simplices[i][t] > self.simplices[i+1][t] + eps:
                print(f"idx i: {self.simplices[i].nodes}, val = {self.simplices[i][t]}")
                print(f"idx i+1: {self.simplices[i+1].nodes}, val = {self.simplices[i+1][t]}")
                return False
        return True
    
    def nbr_lower(self, spx):
        # Return the lower neighbor of spx
        if spx.curr_idx > 0:
            return self.simplices[spx.curr_idx - 1]
        else: return None
    
    def nbr_upper(self, spx):
        if spx.curr_idx <= self.n-2:
            return self.simplices[spx.curr_idx + 1]
        else: return None
    
    def nontrivial_finite_vines(self, start_k = 0, end_k = None):
        # Gets top k nontrivial finite vines, sorted by distance from diagonal. If k = None, returns all nontrivial vines.
        nontrivial_vines = [v for v in self.vines if v.get_dist_from_diag() != np.inf and v.get_dist_from_diag() > 0]
        start_k = max(0, int(start_k))
        N = len(nontrivial_vines)
        if end_k is None:
            end_k = N - 1
        else:
            end_k = min(int(end_k), N - 1)
        #if (isinstance(k, int) or (isinstance(k, float) and k.is_integer())) and k <= len(nontrivial_vines):
        if (end_k - start_k) + 1 <= len(nontrivial_vines):
            nontrivial_vines.sort(key = lambda v : v.get_dist_from_diag(), reverse = True)
            return nontrivial_vines[start_k : end_k + 1]
        else:
            return nontrivial_vines
        
    def nontrivial_vines(self, start_k = 0, end_k = None):
        # Gets top k nontrivial vines, sorted by distance from diagonal. If k = None, returns all nontrivial vines.
        nontrivial_vines = [v for v in self.vines if v.get_dist_from_diag() > 0]
        N = len(nontrivial_vines)
        if end_k is None:
            end_k = N-1
        else:
            end_k = min(int(end_k), N-1)
        if (end_k - start_k) + 1 <= N:
            nontrivial_vines.sort(key = lambda v : v.get_dist_from_diag(), reverse = True)
            return nontrivial_vines[start_k : end_k + 1]
        else:
            return nontrivial_vines

    def plot_vines(self, start_k = 0, end_k = None, include_represented_vines = False):
        # Get the seed top-k finite vines, optionally expanded by any vines that
        # share a representative region with one of those top-k vines.
        seed_vines = self.nontrivial_finite_vines(start_k, end_k)
        if not include_represented_vines:
            return seed_vines

        representative_names = set()
        for vine in seed_vines:
            representative_names.update(vine.get_death_names())

        if not representative_names:
            return seed_vines

        selected_vines = list(seed_vines)
        selected_ids = {id(vine) for vine in selected_vines}
        for vine in self.nontrivial_finite_vines():
            if id(vine) in selected_ids:
                continue
            if vine.get_death_names() & representative_names:
                selected_vines.append(vine)
                selected_ids.add(id(vine))
        return selected_vines
    
    def plot(self, start_k = 0, end_k = None, label_by_region = False, show_legend = True, start_timestep = 0, end_timestep = None, exclude = [], region_colors = None, include_represented_vines = False):
        # Plot top k nontrivial finite vines. If include_represented_vines is
        # True, also plot any vines represented by one of those top-k vines.
        # If label_by_region, color the edges of vines by which region the death simplex comes from.
        plt.figure(figsize = (10, 10))
        ax = plt.axes(projection = '3d')
        cmap = plt.cm.tab20
        indices = np.linspace(0, 1, 20)
        indices = np.append(indices[::2], indices[1::2])
        color = cmap(indices)
        ax.set_prop_cycle(color = color)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_zlabel('Time')
        
        if label_by_region: 
            prev_region_colors = {}
            if region_colors is not None:
                with open(region_colors, 'rb') as f:
                    prev_region_colors = pickle.load(f)
            region_colors = {}
        max_birth = 0
        plot_vines = self.plot_vines(start_k, end_k, include_represented_vines = include_represented_vines)
        if not plot_vines:
            plt.show()
            return region_colors if label_by_region else None

        plot_start_time = None
        plot_end_time = None
        for v in plot_vines:
            # v.births, v.death, v.vertex_times are computed when nontrivial_vines(k) is called.
            vertex_times, births, deaths = v.get_vertices(start_timestep, end_timestep)
            plot_start_time = vertex_times[0] if plot_start_time is None else min(plot_start_time, vertex_times[0])
            plot_end_time = vertex_times[-1] if plot_end_time is None else max(plot_end_time, vertex_times[-1])
            n = len(births)
            if label_by_region:
                edge_labels = v.get_edge_labels_by_death(vertex_times)
                assert len(edge_labels) == len(births)-1
            else:
                vinecolor = None
            for i in range(n-1):
                if births[i] != deaths[i] or births[i+1] != deaths[i+1]:
                    if label_by_region:
                        region = edge_labels[i]
                        if not region in exclude:
                            if region in region_colors:
                                ax.plot([births[i], births[i+1]], [deaths[i], deaths[i+1]], [vertex_times[i], vertex_times[i+1]], color = region_colors[region])
                            elif region in prev_region_colors:
                                p = ax.plot([births[i], births[i+1]], [deaths[i], deaths[i+1]], [vertex_times[i], vertex_times[i+1]], color = prev_region_colors[region])
                                region_colors.update({region : p[-1].get_color()})
                            else:
                                p = ax.plot([births[i], births[i+1]], [deaths[i], deaths[i+1]], [vertex_times[i], vertex_times[i+1]])
                                region_colors.update({region : p[-1].get_color()})
                    else:
                        if vinecolor is None:
                            p = ax.plot([births[i], births[i+1]], [deaths[i], deaths[i+1]], [vertex_times[i], vertex_times[i+1]])
                            vinecolor = p[-1].get_color()
                        else:
                            p = ax.plot([births[i], births[i+1]], [deaths[i], deaths[i+1]], [vertex_times[i], vertex_times[i+1]], color = vinecolor)
            max_birth = max(max_birth, max(births))
        if label_by_region and show_legend:
            display_region_colors = {}
            for region, color in region_colors.items():
                disp_region = region.lower()
                symbs = [" ", "-", "/"]
                for symb in symbs:
                    disp_region = symb.join([s[0:1].upper()+s[1:] for s in disp_region.split(symb)])
                display_region_colors.update({disp_region : color})
            patches = [mpatches.Patch(color= color, label= region) for region, color in display_region_colors.items()]
            ncol = 2
            plt.legend(handles=patches, ncol = ncol, loc = "center right", bbox_to_anchor = (0, .5))
        
        # Plot diagonal plane for reference
        stepsize = max(max_birth/100, 1e-15)
        xs = np.arange(0, max_birth + stepsize, stepsize)
        z_start = math.floor(plot_start_time)
        z_end = math.ceil(plot_end_time)
        xx, zz = np.meshgrid(xs, range(z_start, z_end + 1))
        yy = xx
        ax.plot_surface(xx, yy, zz, alpha = .2, color = 'slategrey')
        plt.show()
        return region_colors
    
    def print_nontrivial_vines(self, start_k, end_k):
        nontrivial_vines = self.nontrivial_vines(start_k, end_k)  # v.births, v.deaths, v.vertex_times are computed when nontrivial_vines() is called
        for v in nontrivial_vines:
            if self.T < 20:
                print("births: ", v.births)
                print("deaths: ", v.deaths)
                print("times: ", v.vertex_times)
            print("regions: ", [pair[1].name if pair[1] is not None else None for pair in v.pairs])
            print("pairing time intervals: ", v.times)
            print("birth simplices: ", [pair[0].nodes for pair in v.pairs])
            print("death simplices: ", [pair[1].nodes if pair[1] is not None else None for pair in v.pairs])
            print("Distance from diagonal: ", v.get_dist_from_diag(), "\n")
    
    def print_vines(self, vines):
        for v in vines:
            if self.T < 20:
                print("births: ", v.births)
                print("deaths: ", v.deaths)
                print("times: ", v.vertex_times)
            print("regions: ", [pair[1].name if pair[1] is not None else None for pair in v.pairs])
            print("pairing time intervals: ", v.times)
            print("birth simplices: ", [pair[0].nodes for pair in v.pairs])
            print("death simplices: ", [pair[1].nodes if pair[1] is not None else None for pair in v.pairs])
            print("Distance from diagonal: ", v.get_dist_from_diag(), "\n")
            
    def swap(self, i):
        '''
        Swaps the order of the simplex that has curr_idx = i with the simplex
        that has curr_idx = i+1.
        Returns True if simplex pairing function needs to be updated, false otherwise
        '''
        
        need_update = self.curr_RU.swap(i, self.simplices[i].dim, self.simplices[i+1].dim)
        self.simplices[i].curr_idx = i+1
        self.simplices[i+1].curr_idx = i
        self.simplices[i+1], self.simplices[i] = self.simplices[i], self.simplices[i+1]
        return need_update
    
    def toSimplexTree(self):
        # Make gd.SimplexTree with the simplices in their current order (actual filtration value doesn't matter as long as order is preserved)
        simplices_in_order = self.simplices
        st = gd.SimplexTree()
        for i, spx in enumerate(simplices_in_order):
            st.insert(list(spx.nodes), i)
        return st
