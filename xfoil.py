"""
http://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout
"""
import subprocess, threading
import utility as u
import os

class Error(Exception):
    pass
class TerminationException(Error):
    def __init__(self, path, airfoil_file_name):
        self.path = path
        self.airfoil_file_name = airfoil_file_name
    def handle(self):
        error_log_file = open(self.path + "failed.log", 'a')
        error_log_file.write(self.airfoil_file_name + " : Convergence Failed"'\n')
        error_log_file.close()


class Command(object):
    def __init__(self,airfoil_file_name, path, alfa = 5):
        self.cmd = '/Applications/Xfoil.app/Contents/Resources/xfoil'
        self.process = None
        self.airfoil_file_name = airfoil_file_name
        self.path = path
        self.alfa = alfa
        self.xfoilpath = self.cmd

    def run(self, timeout):
        def target():
            #print ('Thread started for foil : {}'.format(self.airfoil_file_name))
            try:
             f =open(self.path+self.airfoil_file_name+'.dat', 'r')
             f.close()
            except FileNotFoundError as e:
                error_log_file = open(self.path+"failed.log",'a')
                error_log_file.write(self.airfoil_file_name+" : file not found"'\n')
                error_log_file.close()
            else:
                self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)

                out, err = self.process.communicate(
                                     # "plop\n"
                                      #"g\n"
                                      #" \n"
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
                                      #"alfa 5"
                                      #"aseq {} {} {}\n"
                                      "aseq 5 5 1\n"
                                      #"hard"
                                      " \n"
                                      "quit\n".format(self.path + self.airfoil_file_name + '.dat',
                                                      self.airfoil_file_name,
                                                      self.path + self.airfoil_file_name + '.log'))#,
                                                      #self.alfa_range[0], self.alfa_range[1], self.alfa_range[2]))
                #print(out)
                #print ('Thread finished for foil: {}'.format(self.airfoil_file_name))

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            #print ('Terminating process')
            self.process.terminate()
            thread.join()
        try:
            retutncode = self.process.returncode
            if retutncode==-15:
                raise TerminationException(self.path, self.airfoil_file_name)
        except TerminationException as te:
            te.handle()
        except AttributeError as e:
            error_log_file = open(self.path + "failed.log", 'a')
            error_log_file.write(self.airfoil_file_name +": xfoil not executed" '\n')
            error_log_file.close()

    def getLDfromLog(self):
        filename = self.path + self.airfoil_file_name + ".log"
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



if __name__ == "__main__":
    results_dir = './airfoil-log/'
    command = Command('10', results_dir, alfa = 5)
    command.run(timeout=5)
    print(command.getLDfromLog())
