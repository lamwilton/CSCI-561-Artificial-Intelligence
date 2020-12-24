# CSCI-561-Artificial-Intelligence

## HW1

This is a programming assignment in which you will apply AI search techniques to solve some
sophisticated 3D Mazes. As shown in Figure 1, each 3D maze is a grid of points (not cells) with
(x, y, z) locations in which your agent may use one of the 18 elementary actions (see their
definitions below), named X+, X-, Y+, Y-, Z+, Z-; X+Y+, X-Y+, X+Y-, X-Y-, X+Z+, X+Z-, X-Z+, X-Z-,
Y+Z+, Y+Z-, Y-Z+, Y-Z-; to move to one of the 18 neighboring grid point locations. At each grid
point, your agent is given a list of actions that are available for the current point your agent is
at. Your agent can select and execute one of these available actions to move inside the 3D
maze. For example, in Figure 1, there is a “path” from (0,0,0) to (10,0,0) and to travel this path
starting from (0,0,0), your agent would make nine actions: X+, X+, X+, X+, X+, X+, X+, X+, X+, and
visit the following list of grid points: (0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0), (5,0,0), (6,0,0),
(7,0,0), (8,0,0), (9,0,0), (10,0,0). At each grid point, your agent is given a list of available actions
to select and execute. For example, in Figure 1, at the grid point (60,45,30), there are two
actions for your agent: Z+ for going up, and y- for going backwards. At the grid point
(60,103,97), the available actions are X+ and Y-. At (60,45,97), the three available actions are
Y+, Z-, and X-Y+. If a grid point has no actions available, then that means such a point has
nowhere to go. For example, the point (24,86,31) (not shown in Figure 1) has nowhere to go
and is not accessible.

## HW3

In this programming homework, you will implement a multi-layer perceptron (MLP) neural
network and use it to classify hand-written digits shown in Figure 1. You can use numerical
libraries such as Numpy/Scipy, but machine learning libraries are NOT allowed. You need to
implement feedforward/backpropagation as well as training process by yourselves