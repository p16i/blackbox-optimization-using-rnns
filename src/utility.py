from __future__ import division
import numpy as np
import scipy.interpolate as si
import bezier
import xfoil as xf
import os
import errno

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def prepare_controls(ys): #ys: np.array of ys of the variable points
    control_points_list = []
    #control_points_list.append([1, 0])
    #ys= np.array([0.05, 0.1, 0.2, -0.2, -0.1, -0.05])
    xs =      [1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1]
    counter = 0
    #print('ys: ', ys)
    for x in xs:
        if x == 1:
            control_points_list.append([1,0])
        elif x == 0:
            control_points_list.append([0,0])
        else:
            control_points_list.append([x, ys[counter]])
            counter = counter + 1
    #control_points_list.append([1, 0])
    return  control_points_list

def xfoil(control_points_list, airfoil_file_name, path, method, timeout, alpha):
    """This function will take the control points and pass it to generate_airfoil to generate the cordinates of the
    airfoil, saves it in a file and call xfoill to calculate the L/D coeff. and return it"""
    # TODO: Add type check for arguments: control_pts->list, cp->list
    generate_airfoil(control_points_list, airfoil_file_name, path, method=method)
    command = xf.Command(airfoil_file_name, path, alfa=alpha)
    command.run(timeout)
    ldc = command.getLDfromLog()
    #xf.simulate_foil(airfoil_file_name, path)
    #ldc = xf.getLDfromLog(airfoil_file_name, path)
    return ldc

def generate_airfoil(control_points_list,airfoil_file_name, path, method):
    if method=="bezier":
        #xpoints = [p[0] for p in control_points_list]
        #ypoints = [p[1] for p in control_points_list]
        xvals, yvals = bezier.bezier_curve(control_points_list, nTimes=100)
        foilfile = open('{}/{}.dat'.format(path,airfoil_file_name), 'w') #'w')
        # foilfile.write("test_airfoil\n")
        for i in range(len(xvals)):
            foilfile.write(" {:.6f}    {:.6f}\n".format(xvals[i], yvals[i]))
        foilfile.close()
    elif method == "spline":
        """takes the nodes as control points and generates the shape and returns the file name"""
        # nodes = np.array(control_points_list)
        # x = nodes[:, 0]
        # y = nodes[:, 1]
        # tck, u = interpolate.splprep([x, y], s=0)
        # xnew, ynew = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)

        degree = 3

        control_points_list = control_points_list + control_points_list[0:degree + 1]
        points = np.array(control_points_list)
        n_points = len(points)
        x = points[:, 0]
        y = points[:, 1]

        t = range(len(x))
        ipl_t = np.linspace(1.0, len(points) - degree, 1000)

        x_tup = si.splrep(t, x, k=degree, per=1)
        y_tup = si.splrep(t, y, k=degree, per=1)
        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = [0.0] + xl + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = [0.0] + yl + [0.0, 0.0, 0.0, 0.0]

        xnew = si.splev(ipl_t, x_list)
        ynew = si.splev(ipl_t, y_list)

        foilfile = open('{}/{}.dat'.format(path, airfoil_file_name), 'w')  # 'w')
        # foilfile.write("test_airfoil\n")
        for i in range(len(xnew)):
            foilfile.write(" {:.6f}    {:.6f}\n".format(xnew[i], ynew[i]))
        foilfile.close()
    else:
        print('Specified point genereation method Error')

# def generate_airfoil_spline(control_points_list,airfoil_file_name, path):
#     """takes the nodes as control points and generates the shape and returns the file name"""
#     nodes = np.array(control_points_list)
#     x = nodes[:, 0]
#     y = nodes[:, 1]
#     tck, u = interpolate.splprep([x, y], s=0)
#     xnew, ynew = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
#     foilfile = open('{}/{}.dat'.format(path, airfoil_file_name), 'w')  # 'w')
#     #foilfile.write("test_airfoil\n")
#     for i in range(len(xnew)):
#         foilfile.write(" {:.6f}    {:.6f}\n".format(xnew[i], ynew[i]))
#     foilfile.close()

###### Test ###########
if __name__ == "__main__":
    #TODO: write test scripts for each funtion here
    results_dir = r'./airfoil-log/'
    command = xf.Command('10', results_dir, alfa=5)
    command.run(timeout=10)
    ldc = command.getLDfromLog()

    #test xfoil()##############
    ys = [0.05, 0.1, 0.2, -8.2, -0.1, -0.05]
    control_points_list = prepare_controls(ys)
    ldc = xfoil(control_points_list, '10', results_dir, method= 'bezier', timeout=5, alpha=5)
    print("Iteration:   " + '10' + '\n' + 'control points: ' + str(control_points_list)
          + '\n' + "L/D = " + str(ldc) + "\n")
