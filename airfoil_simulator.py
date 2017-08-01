import subprocess, threading
import os
import utility as u
import numpy as np
import os
import shutil

foilnum = 0

def simulate(airfoil_file_name, path, alpha):
    """
    airfoil_file_name: str parameter for name of the airfoil.
    path: str value for the directory of the airfoil_file path
    This function runs simulate the airfoil and stores the results in the same directory as a .log file
    """
    xfoilpath = os.environ['XFOIL_PATH']
    process = subprocess.Popen(xfoilpath,stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
    def target():
        out, err = process.communicate(
            "plop\n"
            "g\n"
            " \n"
            "load {foil}\n"
            "foil{foil_name}\n"
            "pane\n"
            "oper\n"
            "visc 5e005\n"
            "M 0.2\n"
            "ITER\n"
            "300\n"
            "pacc\n"
            "{log}\n"
            " \n"
            "alfa {alpha}\n"
            # "aseq {} {} {}\n"
            "aseq 5 5 1\n"
            # "hard"
            " \n"
            "quit\n".format(
                foil = path + airfoil_file_name + '.dat',
                foil_name = airfoil_file_name,
                log = path + airfoil_file_name + '.log',
                alpha = alpha
            )
        )

    thread = threading.Thread(target=target)
    thread.start()

    thread.join(5)
    if thread.is_alive():
        #print ('Terminating process')
        process.terminate()
        thread.join()
    try:
        retutncode = process.returncode
        if retutncode==-15:
            print('XFOIL : Not converge')
        else:
            print('XFOIL : Converged')
    except:
        pass
            # raise TerminationException(self.path, airfoil_file_name)
    # except TerminationException as te:
    #     te.handle()
    # except AttributeError as e:
    #     error_log_file = open("./airfoil-failed.log", 'a')
    #     error_log_file.write(airfoil_file_name +": xfoil not executed" '\n')
    #     error_log_file.close()

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
    # LD = dict()
    # for i in range(12, len(flines)):
        # print flines[i]
    if len(flines) >= 13:
        words = str.split(flines[12])
        return float(words[1]) / float(words[2])
    else:
        return -1
    # #print(LD)
    # if "5.000" in LD.keys():
    #         return LD["5.000"]
    # else:
    #         return -1



def objective(ys, pos1=3, pos2=4, alpha=5, debug=False):
    """
    ys: y-cordinates of the control points
    takes the ys and combine them with xs -> control points~(x,y)
    pass the (x,y) to bizer to create airfoil shape -> writes it in .dat file
    calls simulate function to simulate the airfoil shape and returns the L/D coefficient
    """
    ##### append remaining ys as we are taking only 2 ys out of 6 for now.
    tmp = [0.05, 0.1, 0.2, -0.2, -0.1, -0.05]
    tmp[pos1] = ys[0]
    tmp[pos2] = ys[1]

    ys = np.array(tmp)
    ########################
    #print(ys)
    ##############################
    global foilnum
    control_points_list = u.prepare_controls(ys)
    path = r'.test_opt/'
    u.make_sure_path_exists(path)
    u.generate_airfoil(control_points_list, str(foilnum), path, method='bezier')
    simulate(str(foilnum), path, alpha=alpha)
    ldc = getLDfromLog(str(foilnum), path)
    if debug:
        print ("Iteration:   " + str(foilnum) + '\n' + 'control points: ' + str(control_points_list)
                + '\n' + "L/D = " + str(ldc) + "\n")
    else:
        os.remove('%s%s.log' % (path, foilnum))
        os.remove('%s%s.dat' % (path, foilnum))

    foilnum = foilnum + 1

    return ldc

def objective6d(ys, alpha=5, debug=False):
    """
    ys: y-cordinates of the control points
    takes the ys and combine them with xs -> control points~(x,y)
    pass the (x,y) to bizer to create airfoil shape -> writes it in .dat file
    calls simulate function to simulate the airfoil shape and returns the L/D coefficient
    """
    ##### append remaining ys as we are taking only 2 ys out of 6 for now.
    tmp = [0.05, 0.1, 0.2, -0.2, -0.1, -0.05]
    tmp = list(np.array(tmp)+np.array(ys))

    ys = np.array(tmp)
    ########################
    #print(ys)
    ##############################
    global foilnum
    control_points_list = u.prepare_controls(ys)
    path = r'.test_opt/'
    u.make_sure_path_exists(path)
    u.generate_airfoil(control_points_list, str(foilnum), path, method='bezier')
    simulate(str(foilnum), path, alpha=alpha)
    ldc = getLDfromLog(str(foilnum), path)
    if debug:
        print ("Iteration:   " + str(foilnum) + '\n' + 'control points: ' + str(control_points_list)
                + '\n' + "L/D = " + str(ldc) + "\n")
    else:
        os.remove('%s%s.log' % (path, foilnum))
        os.remove('%s%s.dat' % (path, foilnum))

    foilnum = foilnum + 1

    return ldc


if __name__ == '__main__':
    y0 =  np.array([-0.1,  -0.1]) # only 2 ys for now.
    obj_value  = objective(y0)
    print("L/G : %f" % (obj_value))
