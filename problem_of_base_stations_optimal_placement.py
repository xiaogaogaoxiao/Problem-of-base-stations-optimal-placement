"""
2019

The problem of Base Stations Optimal Placement in Wireless Networks with linear
topology is solved in the form of the extremal combinatorial model. The
algorithm of the branch and bound method is proposed.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class BranchAndBound:
    def __init__(self, placement, coverage, link_range, table):
        """
        Algorithm of the branch and bound method for solving the extremal
        problem "Problem of Base Stations Optimal Placement in Wireless
        Networks with Linear Topology"
        :param placement: <class 'tuple'>
        :param coverage: <class 'tuple'>
        :param link_range: <class 'tuple'>
        :param table: <class 'type'>
        """
        self.placement = placement
        self.coverage = coverage
        self.link_range = link_range
        self.record = self.placement[-1]
        self.table = table
        self.G = nx.Graph()
        self._pi = np.ones((len(placement) - 2, len(coverage) - 2)) * np.inf
        self._p = np.arange(1, len(self.placement) - 1, 1)
        self._s = np.arange(1, len(self.coverage) - 1, 1)
        self._busy_p = []
        self._busy_s = []
        self._est_mat = np.ones((len(placement)-2, len(coverage)-2)) * np.inf
        self._two_loop_exit = None  # exit flag
        self._not_able_p = None
        self._not_able_s = None

    def _in_range(self, first, second, range_):
        """check whether 'first' and 'second' position is in 'range_' """
        return abs(self.placement[first] - self.placement[second]) <= range_

    def is_able_exist(self):
        """ checking the existence of feasible solution """
        max_range = max(self.link_range)
        without_max_range = tuple(i for i in self.link_range if i != max_range)
        max_range_2 = max(without_max_range)

        last = len(self.placement) - 1

        if not (self._in_range(0, 1, max_range) and
                self._in_range(last, last - 1, max_range)):
            return False

        for i in range(1, last - 1):
            if not self._in_range(i, i + 1, max_range_2):
                return False

        return True

    def _left_placed(self):
        """
        Determine the indices of the last placed station (pi = 1)
        :return: indices of left placed station
        """
        if 1 in self._pi:
            i, j = np.where(self._pi == 1)
            i = i[-1]
            j = j[-1]
            return [i + 1, j + 1]
        else:
            return [0, 0]

    def is_able_link_left(self, p_ind, s_ind):
        """
        check left of placement point
        :param p_ind: index of placement
        :param s_ind: index of station
        :return: (bool) -True if link range is greater than distance or
            False otherwise
        """
        left_p, left_s = self._left_placed()

        if not self._in_range(left_p, p_ind, self.link_range[s_ind]):
            return False

        if left_p != 0:
            if not self._in_range(left_p, p_ind, self.link_range[left_s]):
                return False
        return True

    def is_able_link_right(self, p_ind, s_ind):
        """
        check right of placement point
        :param p_ind: index of placement
        :param s_ind: index of station
        :return: (bool) -True if link range is greater than distance or
            False otherwise
        """
        if len(self._busy_s) == len(self._s):
            right_index = -1
        else:
            right_index = p_ind + 1

        if not self._in_range(right_index, p_ind, self.link_range[s_ind]):
            return False

        if p_ind != (len(self.placement) - 1):
            ind_placed_stations = [i-1 for i in self._busy_s]
            unplaced_station = np.delete(self.link_range, ind_placed_stations)
            max_unplaced_range = np.max(unplaced_station)
            if not self._in_range(right_index, p_ind, max_unplaced_range):
                return False
        return True

    def is_able_within_unplaced(self, k, unplaced):
        """
        check unplaced station
        :param k: index of placement
        :param unplaced: unplaced station
        :return: (bool) -True if link range of unplaced station is greater than
            distance or False otherwise
        """
        unplaced_link_range = [self.link_range[i] for i in unplaced]
        if len(unplaced) == 1:
            for i in range(len(self.placement[k:-1])):
                if not self._in_range(i+k+1, i+k, unplaced_link_range[0]) and \
                        not self._in_range(-1, i+k+1, unplaced_link_range[0]):
                    return False

        if len(unplaced) > 1:
            max_range = max(unplaced_link_range)
            unplaced_link_range.remove(max_range)
            max_range_2 = max(unplaced_link_range)

            for i in range(len(self.placement[k:-1])):
                if i == len(self.placement) - 2:
                    if not self._in_range(i + 1, i, max_range):
                        return False
                else:
                    if not self._in_range(i + 1, i, max_range_2):
                        return False
        return True

    def under_coverage_est(self):
        """
        calculate the estimates of under coverage
        :return: estimates of under coverage
        """
        busy_p = [0]
        busy_s = [0]
        left_under_coverage = 0
        if not self._busy_p == []:
            busy_p = busy_p + self._busy_p
            busy_s = busy_s + self._busy_s

            for i in range(len(busy_p) - 1):
                left_distance = (self.placement[busy_p[i + 1]] -
                                 self.placement[busy_p[i]])
                left_coverage = (self.coverage[busy_s[i + 1]] +
                                 self.coverage[busy_s[i]])
                under_coverage = np.max([left_distance - left_coverage, 0])
                left_under_coverage = left_under_coverage + under_coverage

        unplaced_sta_coverage = np.delete(self.coverage, busy_s)
        sum_right = np.sum(2 * [unplaced_sta_coverage])

        right_distance = self.placement[-1] - self.placement[busy_p[-1]]
        right_coverage = self.coverage[busy_s[-1]] + sum_right
        right_under_coverage = np.max([right_distance - right_coverage, 0])

        return left_under_coverage + right_under_coverage

    def _num_of_empty_plc(self):
        """
        Determination of the number of free seats on the right for the number
        of unplaced stations
        Args
            :param pi: matrix of split parameter
            :param stations: still unused stations
            :param places:
        :return: (bool) True if there are empty seats or False otherwise
        """
        sum_by_row = self._pi.sum(axis=1)
        forbidden_place = sum_by_row[np.where(sum_by_row == 0)]
        if len(self._p) - len(forbidden_place) >= len(self._s):
            return True
        return False

    def get_new_node(self, i, j):
        """

        :param i: placement index
        :param j: station index
        :return: new node of solution tree
        """
        param = self._pi[i - 1, j - 1]
        # determine the parent node at this iteration
        parents_node = self.table.detect_parent(i, j, self._pi)
        if param == 1: # Pi == 1
            self._est_mat[i - 1, j - 1] = self.under_coverage_est()
            est = self._est_mat[i - 1, j - 1]
        else:  # Pi == 0
            est = self.table.solution.W[parents_node]

        self.table.write(est, i, j, param)
        child_node = self.table.last_node()

        self.G.add_edge(parents_node, child_node)

    def check_loop_exit(self, i, j):
        """
        checking loop exit conditions
        :param i: placement index
        :param j: station index
        :return: (bool) True if need to exit and False otherwise
        """
        # condition 1
        # comparing score with current record
        if self.record <= self._est_mat[i - 1, j - 1]:
            return True
        # condition 2
        # condition of placement of all stations
        if len(self._pi[np.where(self._pi == 1)]) == len(self._s):
            if self._est_mat[i - 1, j - 1] < self.record:
                self.record = self._est_mat[i - 1, j - 1]
            return True

    def node_forbid(self):
        """
        forbid to place j station to i placement
        :return: add forbidden node
        """
        i = self._not_able_p
        j = self._not_able_s
        if i is None and j is None:
            i, j = self._left_placed()

            self._busy_p.remove(i)
            self._busy_s.remove(j)

            # the condition of the prohibition of placing the j station
            # on the i placement
            self._pi[i - 1, j - 1] = 0
            self._pi[i:, :] = np.inf

        else:
            # the condition of the prohibition of placing the j station
            # on the i placement
            self._pi[i - 1, j - 1] = 0

            self._busy_p.remove(self._p[i - 1])
            self._busy_s.remove(self._s[j - 1])

        self.get_new_node(i, j)

    def to_select_place(self, i, j):
        """
        select placement for station
        :param i: placement index
        :param j: station index
        :return: get right child node of solution
        """
        if self._pi[i-1, j-1] != 0 and \
                self._pi[i-1, j-1] != 1 and \
                i not in self._busy_p and \
                j not in self._busy_s:

            self._busy_p.append(self._p[i - 1])
            self._busy_s.append(self._s[j - 1])

            # unplaced stations
            unplaced_s = self._s[np.where(self._s != self._busy_s)]

            if self.is_able_link_left(i, j) and \
                    self.is_able_link_right(i, j) and \
                    self.is_able_within_unplaced(i, unplaced_s):

                # place the j-th station to the i-th place
                self._pi[i-1, j-1] = 1
                self.get_new_node(i, j)

                if self.check_loop_exit(i, j):
                    self._two_loop_exit = True
            else:
                self._two_loop_exit = True

                self._not_able_p = i
                self._not_able_s = j

    def find_record(self):
        """ get record of under coverage """
        while self._num_of_empty_plc():
            self._two_loop_exit = False

            self._not_able_p = None
            self._not_able_s = None

            for i in self._p:
                for j in self._s:
                    self.to_select_place(i, j)

                    if self._two_loop_exit:
                        break

                if self._two_loop_exit:
                    break

            self.node_forbid()
            self.draw_graph()

    def initiate_solution_tree(self):
        """ get init estimate and parent init node """
        all_coverage = sum([2 * i for i in self.coverage])
        init_estimate = max(self.placement[-1] - all_coverage, 0)
        self.table.write(init_estimate)
        parents_node = self.table.last_node()
        self.G.add_node(parents_node)

    def solution(self):
        """ main solution """
        if not self.is_able_exist():
            print('there is no solution')
        else:
            self.initiate_solution_tree()
            self.find_record()
            print('Received solution is ' + str(self.record))

    def _hierarchy_pos(self, root, width=1., vert_gap=1, vert_loc=0, xcenter=.5,
                      pos=None, parent=None):
        """
        If there is a cycle that is reachable from root, then this will see
        infinite recursion.
        :param root: the root node of current branch
        :param width: horizontal space allocated for this branch - avoids
            overlap with other branches
        :param vert_gap: gap between levels of hierarchy
        :param vert_loc: vertical location of root
        :param xcenter: horizontal location of root
        :param pos: a dict saying where all nodes go if they have been
            assigned
        :param parent: parent of this branch.
        :return: Node positions in plot
        """
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        neighbors = list(self.G.neighbors(root))
        if parent is not None:  # this should be removed for directed graphs.
            neighbors.remove(parent)  # if directed, then parent not in
            # neighbors.
        if len(neighbors) != 0:
            dx = width / len(neighbors)
            nextx = xcenter - width / 2 - dx / 2
            for neighbor in neighbors:
                nextx += dx
                pos = self._hierarchy_pos(neighbor,
                                          width=dx,
                                          vert_gap=vert_gap,
                                          vert_loc=(vert_loc - vert_gap),
                                          xcenter=nextx, pos=pos,
                                          parent=root)
        return pos

    def draw_graph(self):
        plt.clf()
        nx.draw(self.G,
                pos=self._hierarchy_pos(0),
                labels=dict(
                    r'$\pi_{' +
                    self.table.solution.i +
                    self.table.solution.j +
                    '}$' +
                    '=' +
                    self.table.solution.Pi
                ),
                node_size=200,
                alpha=0.9,
                node_color='royalblue')
        plt.show()

    def write_csv(self, table):
        fd = open(
            'Branch and bound  ' +
            'l = ' +
            str(self.placement) +
            ' r = ' +
            str(self.coverage) +
            ').csv',
            'w'
        )
        fd.write('Branch and bound' + '\n')
        fd.write('coverage = ' + str(self.coverage) + '\n')
        fd.write('link_range = ' + str(self.link_range) + '\n')
        fd.write('placement coordinates = ' + str(self.placement) + '\n')
        fd.write('under coverage = ' + str(self.record) + '\n')
        fd.write('Number of nodes = ' + str(table.last_node()) + '\n')
        fd.close()
        table.solution.to_csv(
            'Branch and bound  ' +
            'l = ' +
            str(self.placement) +
            ' r = ' +
            str(self.coverage) +
            ').csv',
            mode='a'
        )


class Table:
    """solution stack"""
    def __init__(self):
        self.solution = None

    def write(self, w, i=0, j=0, parameter=1):
        """
        :param w: estimate of under coverage
        :param i: placement index
        :param j: station index
        :param parameter:split parameter
        :return: to write to solution table
        """
        if self.solution is None:
            self.solution = pd.DataFrame(
                            [{
                                'W': float(w),
                                'i': str(i),
                                'j': str(j),
                                'Pi': str(parameter)
                            }, ])
        else:
            self.solution = self.solution.append(
                {
                    'W': float(w),
                    'i': str(i),
                    'j': str(j),
                    'Pi': str(parameter)
                },
                ignore_index=True)

    def last_node(self):
        """ last node in tree"""
        return self.solution.last_valid_index()

    def detect_parent(self, i_last, j_last, pi):
        """
        Determine parent node
        :param i_last: index of place
        :param j_last: index of station
        :param pi: matrix of split parameter
        :return: indices of parent node
        """
        mas = np.copy(pi)
        mas[i_last - 1, j_last - 1] = np.inf
        if np.all(mas == np.inf):
            return 0
        i, j = np.where(mas != np.inf)
        i = i[-1]
        j = j[-1]
        parents_node = self.solution[
            (self.solution.Pi == str(pi[i, j])) &
            (self.solution.i == str(i + 1)) &
            (self.solution.j == str(j + 1))
            ].index[-1]
        return parents_node


if __name__ == "__main__":
    l_vector = (0, 20, 30, 40, 50)
    r = (0, 20, 5, 0)
    R = (0, 40, 20, 0)

    bab = BranchAndBound(l_vector, r, R, Table())
    bab.solution()

