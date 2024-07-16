# optimization.py

import argparse
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import time

### 
# IMPLEMENT ME! REPLACE WITH YOUR ANSWER TO PART 1B
OPTIMAL_STEP_SIZE = 0.1
###


def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='optimization.py')
    parser.add_argument('--func', type=str, default='QUAD', help='function to optimize (QUAD or NN)')
    parser.add_argument('--lr', type=float, default=1., help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    return args


def quadratic(x1, x2):
    """
    Quadratic function of two variables
    :param x1: first coordinate
    :param x2: second coordinate
    :return:
    """
    return (x1 - 1) ** 2 + 8 * (x2 - 1) ** 2


def btls(alpha, beta, eta, x):

    q = np.array([-22,-14.5,13])
    Q = np.array([[13,12,-2],[12,17,6],[-2,6,12]])
    fprime = np.dot(Q, x) + q

    while True:

        xnd = x - (eta * fprime)
        f1 = 0.5 * np.dot((np.dot(xnd, Q)), xnd) + np.dot(q, xnd) - 1
        fx = (0.5 * np.dot((np.dot(x, Q)), x) + np.dot(q, x) - 1)
        f2 = fx - alpha * eta * np.dot(fprime, fprime)

        if f1 <= f2:
            break
        eta = eta *beta

    return eta


def quadratic_grad(x1, x2):
    """
    Should return a numpy array containing the gradient of the quadratic function defined above evaluated at the point
    :param x1: first coordinate
    :param x2: second coordinate
    :return: a two-dimensional numpy array containing the gradient
    """
    q = np.array([-22,-14.5,13])
    Q = np.array([[13,12,-2],[12,17,6],[-2,6,12]])
    x_star = np.array([1.25814396, 0.31423343, -1.02068144])
    # Q = np.asmatrix(Q1)
    xprev = np.array([5, 5, 5])
    xnext = np.array([0, 0, 0])
    t = 1.0
    res = []
    f_star = 0.5 * np.dot((np.dot(x_star, Q)), x_star) + np.dot(q, x_star) - 1
    fx = 0
    eta = 0.5

    delta = 0.1
    while True:

        # eta = (1 / 27.89)
        eta = btls(0.25, 0.5, eta, xprev)
        xnext = xprev - eta*(np.dot(Q, xprev) + q)
        fx = 0.5 * np.dot((np.dot(xnext, Q)), xnext) + np.dot(q, xnext) - 1

        # res.append(np.linalg.norm(xnext - xprev))
        res.append(abs(fx - f_star))
        t += 1
        if abs(fx - f_star) <= delta:
            break
        xprev = xnext


    print("Fx: " + str(fx))
    print("F_star: " + str(f_star))
    plt.plot(np.arange(1, t, 1), res)
    plt.title('eta = BTLS')
    plt.xlabel('iterations')
    plt.ylabel("Error")
    plt.show()


def sgd_test_quadratic(args):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = quadratic(X, Y)
    plt.figure()

    args.lr = OPTIMAL_STEP_SIZE
    # Track the points visited here
    points_history = []
    curr_point = np.array([0, 0])
    for iter in range(0, args.epochs):
        grad = quadratic_grad(curr_point[0], curr_point[1])
        if len(grad) != 2:
            raise Exception("Gradient must be a two-dimensional array (vector containing [df/dx1, df/dx2])")
        next_point = curr_point - args.lr * grad
        points_history.append(curr_point)
        print("Point after epoch %i: %s" % (iter, repr(next_point)))
        curr_point = next_point
    points_history.append(curr_point)
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.plot([p[0] for p in points_history], [p[1] for p in points_history], color='k', linestyle='-', linewidth=1, marker=".")
    plt.title('SGD on quadratic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    exit()


if __name__ == '__main__':
    args = _parse_args()
    quadratic_grad(1, 2)
    # sgd_test_quadratic(args)
