import subprocess, threading
import os
import utility as u
import numpy as np

foilnum = 0

def simulate(airfoil_file_name, path):
    """
    airfoil_file_name: str parameter for name of the airfoil.
    path: str value for the directory of the airfoil_file path
    This function runs simulate the airfoil and stores the results in the same directory as a .log file
    """
    #xfoilpath = '/Applications/Xfoil.app/Contents/Resources/xfoil'
    xfoilpath = 'C:/Users/User/Downloads/XFOIL6.99/xfoil'
    process = subprocess.Popen(xfoilpath,stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
    out, err = process.communicate(
            # "plop\n"
            # "g\n"
            # " \n"
            "load {}\n"
            "foil{}\n"
            "pane\n"
            "oper\n"
            "visc 5e005\n"
            "M 0.2\n"
            "ITER\n"
            "300\n"
            "pacc\n"
            "{}\n"
            " \n"
            # "alfa 5"
            # "aseq {} {} {}\n"
            "aseq 5 5 1\n"
            # "hard"
            " \n"
            "quit\n".format(path + airfoil_file_name + '.dat',
                            airfoil_file_name,
                            path + airfoil_file_name + '.log'))

def getLDfromLog(airfoil_file_name, path):
    """
    airfoil_file_name: str parameter for name of the airfoil.
    path: str value for the directory of the airfoil_file path
    reads the airfoil_file_name.log file and returns the L/D value for angle_of_attach(alpha)=5
    if the log file does not have an entry, it returns -1 which means the simulation was unsuccesful because xfoil could not
    converge or the specified airfoil shape was really bad.
    """
    filename = path + airfoil_file_name + ".log"
    f = open(filename, 'r')
    flines = f.readlines()
    LD = dict()
    for i in range(12, len(flines)):
        # print flines[i]
        words = str.split(flines[i])
        alfa = words[0]
        LD[alfa] = float(words[1]) / float(words[2])
    #print(LD)
    if "5.000" in LD.keys():
            return LD["5.000"]
    else:
            return -1



def objective(ys, debug=False):
    """
    ys: y-cordinates of the control points
    takes the ys and combine them with xs -> control points~(x,y)
    pass the (x,y) to bizer to create airfoil shape -> writes it in .dat file
    calls simulate function to simulate the airfoil shape and returns the L/D coefficient
    """
    ##### append remaining ys as we are taking only 2 ys out of 6 for now.
    tmp = [0.05, 0.1, 0.2]
    tmp.append(ys[0])
    tmp.append(ys[1])
    tmp.append(-0.05)
    ys = np.array(tmp)
    ########################
    #print(ys)
    ##############################
    global foilnum
    control_points_list = u.prepare_controls(ys)
    path = r'.test_opt/'
    u.make_sure_path_exists(path)
    u.generate_airfoil(control_points_list, str(foilnum), path, method='bezier')
    simulate(str(foilnum), path)
    ldc = getLDfromLog(str(foilnum), path)
    if debug:
        print ("Iteration:   " + str(foilnum) + '\n' + 'control points: ' + str(control_points_list)
                + '\n' + "L/D = " + str(ldc) + "\n")
    foilnum = foilnum + 1
    return ldc


if __name__ == '__main__':
    y0 =  np.array([-0.1,  -0.1]) # only 2 ys for now.
    obj_value  = objective(y0)
    print("L/G : %f" % (obj_value))
