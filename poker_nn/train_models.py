import subprocess
import platform
import sys
import os
import shutil

sys.path.append("/home/amiraj/Dropbox/Courses/CS_6700_Advanced AI/caffe-master/python/caffe/")
sys.path.append("/home/amiraj/Dropbox/Courses/CS_6700_Advanced AI/caffe-master/python/")
import caffe
caffe.set_mode_gpu()
import lmdb

from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

#Custom Modules
import dataload_lmdb

def main():
    #Spit out system information
    print "OS:     ", platform.platform()
    print "Python: ", sys.version.split("\n")[0]
    print "CUDA:   ", subprocess.Popen(["nvcc","--version"], stdout=subprocess.PIPE).communicate()[0].split("\n")[3]
    print "LMDB:   ", ".".join([str(i) for i in lmdb.version()])
    print ""

    no_cards = 0

    while True:
        try:
            no_cards = int(raw_input("No of cards for system? [3/4/5/0] (0 for all):\t"))
            if no_cards == 3 or no_cards == 4 or no_cards == 5:
                break
            else:
                raise ValueError
        except ValueError:
            print "Incorrect input, try again!"

    lmdb_train_path = "../dataset/lmdb/cards_" + str(no_cards) +"_train_data_lmdb"
    lmdb_test_path = "../dataset/lmdb/cards_" + str(no_cards) +"_test_data_lmdb"
    train_data_set = "../dataset/poker-hand-training-true.data"
    test_data_set = "../dataset/poker-hand-testing.data"
    caffe_path = "../../caffe-master/build/tools/caffe"
    config_path = "../nn_config/cards_" + str(no_cards) + "_config/"
    config_filename = "config.prototxt"
    model_test_filaname = "model_test.prototxt"
    caffe_nets = "../caffe_nets/cards_" + str(no_cards) + "_net/"

    max_iters_3_cards = 100000
    max_iters_4_cards = 100000
    max_iters_5_cards = 100000

    caffemodel_filename = "_iter_"
    solverstate_filaname = "_iter_"

    if no_cards == 3:
        caffemodel_filename += str(max_iters_3_cards)
        solverstate_filaname += str(max_iters_3_cards)
    elif no_cards == 4:
        caffemodel_filename += str(max_iters_4_cards)
        solverstate_filaname += str(max_iters_4_cards)
    elif no_cards == 5:
        caffemodel_filename += str(max_iters_5_cards)
        solverstate_filaname += str(max_iters_5_cards)

    caffemodel_filename += ".caffemodel"
    solverstate_filaname += ".solverstate"

    while True:
        load_data = raw_input("Load data into LMDB? Deletes old data if found. [y/n]:\t")
        if load_data == "y" or load_data == "Y":
            print "Loading data into LMDB..."

            if os.path.isdir(lmdb_train_path):
                shutil.rmtree(lmdb_train_path)
            if os.path.isdir(lmdb_test_path):
                shutil.rmtree(lmdb_test_path)

            df = pd.read_csv(train_data_set, sep=",")
            testing_data = pd.read_csv(test_data_set, sep=",")

            training_features = df.ix[:,:(no_cards * 2)].as_matrix()
            training_labels = df.ix[:,-1].as_matrix()

            testing_features = testing_data.ix[:,:(no_cards * 2)].as_matrix()
            testing_labels = testing_data.ix[:,-1].as_matrix()

            dataload_lmdb.load_data_into_lmdb(lmdb_train_path, training_features, training_labels)
            dataload_lmdb.load_data_into_lmdb(lmdb_test_path, testing_features, testing_labels)

            break
        elif load_data == "N" or load_data == "n":
            break
        else:
            print "Incorrect input, try again!"


    while True:
        train_net = raw_input("Train the network? [y/n]:\t")
        if train_net == "y" or train_net == "Y":
            #dataload_lmdb.get_data_for_case_from_lmdb(lmdb_train_path, "00012345")
            print "Training..."

            proc = subprocess.Popen(
                [caffe_path, "train", "--solver=" + config_path + config_filename], 
                stderr=subprocess.PIPE)
            res = proc.communicate()[1]
            
            if proc.returncode != 0:
                print "Error in Caffe training!"
                print res
                sys.exit()

            shutil.move(caffemodel_filename, caffe_nets + caffemodel_filename)
            shutil.move(solverstate_filaname, caffe_nets + solverstate_filaname)

            break
        elif train_net == "n" or train_net == "N":
            break
        else:
            print "Incorrect input, try again!"

    while True:
        test_net = raw_input("Test the caffe net? [y/n]:\t")
        if test_net == "y" or test_net == "Y":

            if not os.path.exists(config_path + model_test_filaname):
                print "Model_test.prototxt for cards_" + str(no_cards) + "_config not found!"
                break
            if not os.path.exists(caffe_nets + caffemodel_filename):
                print "Caffemodel for cards_" + str(no_cards) + "_net not found, first train the network for cards_"\
                     + str(no_cards) + " first!"
                break

            print "Testing..."
            net = caffe.Net(config_path + model_test_filaname, caffe_nets + caffemodel_filename, caffe.TEST)
            labels, features = dataload_lmdb.get_data_for_case_from_lmdb(lmdb_test_path, "00001230")
            
            out = net.forward(**{net.inputs[0]: np.asarray([features])})
            print np.argmax(out["prob"][0]) == labels, "\n", out
            plt.bar(range(10),out["prob"][0])
            pylab.show()
            break
        elif test_net == "n" or test_net == "N":
            break
        else:
            print "Incorrect input, try again!"

    print "\n..........End of script.........."

    # pylab.show()
    # proc = subprocess.Popen(
    #     ["/home/amiraj/Dropbox/Courses/CS_6700_Advanced AI/caffe-master/build/tools/caffe","train",
    #     "--solver=" + config_path + config_filename], 
    #     stderr=subprocess.PIPE)
    # res = proc.communicate()[1]

    # #print res

    # net = caffe.Net("model_prod.prototxt","./_iter_100000.caffemodel", caffe.TEST)

    # l, f = get_data_for_case_from_lmdb("/home/amiraj/Dropbox/Courses/CS_6700_Advanced AI/pokerbot/test_data_lmdb/", "00001230")
    # out = net.forward(**{net.inputs[0]: np.asarray([f])})

    # # if the index of the largest element matches the integer
    # # label we stored for that case - then the prediction is right
    # print np.argmax(out["prob"][0]) == l, "\n", out
    # plt.bar(range(10),out["prob"][0])
    # pylab.show()

if __name__ == "__main__":
    main()