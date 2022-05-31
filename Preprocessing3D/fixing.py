import os
import numpy as np
import shutil
import sys

NUM_FRAME = 448

def getNumber(st):
    start = st.find('_')
    end = st.find('.')
    return int(st[start + 1:end])

def prediction(folder,start,end,out_path):
    start_path = out_path + folder + '/' + start
    end_path = out_path + folder + '/' + end

    print("Start path ",start_path)
    print("End path ",end_path)

    st_vector = np.load(start_path)
    end_vector = np.load(end_path)

    pred = []
    for i in range(0,len(st_vector)):
        l = []
        for j in range(0,len(st_vector[i])):
            l.append((st_vector[i][j] + end_vector[i][j])/2)
        pred.append(l)

    return np.array(pred)

def remove(folders,out_path):
    for folder in folders:
        shutil.rmtree(out_path + folder, ignore_errors=False, onerror=None)

def fix(out_path):
    """
    Check whether the videos have a great amount of npy files
    """
    l = os.listdir(out_path)
    toDelete = []
    good = []
    for folder in l:
        lf = os.listdir(out_path + folder)
        # caso base
        if getNumber(lf[0]) <= 5 and getNumber(lf[len(lf) - 1]) >= NUM_FRAME -10:
            for i in range(0,len(lf) - 1):
                if getNumber(lf[i]) >= NUM_FRAME - 10:
                    #print("va bene ",lf[i])
                    good.append(folder)
                    break
                elif (getNumber(lf[i+1]) - (getNumber(lf[i]))) > 10:
                    toDelete.append(folder)
                    break
        else:
            toDelete.append(folder)

    #print("dir to fill = ", good)
    #print("dir to delete = ", toDelete)
    print("length of good = ",len(good))
    print("length of toDelete = ",len(toDelete))
    fill(good,out_path)
    remove(toDelete,out_path)



def removing_files(folder,files,out_path):
    print("Removing files...")
    for i in range(NUM_FRAME,len(files)):
        os.remove(out_path + folder + '/' + files[i])


def fill(folders,out_path):
    for folder in folders:
        i = 0
        files = os.listdir(out_path + folder)
        # Fase di sola copia
        # se parte dopo video_001
        if getNumber(files[0]) > 1:
            print("primi 10 file...")
            for i in range(1,getNumber(files[0])):
                to_copy = np.load(out_path + folder + '/' + files[0])
                if i == 10:
                    np.save(out_path + folder + '/' + folder + '_0000' + str(i) + '.npy',to_copy)
                else:
                    np.save(out_path + folder + '/' + folder + '_00000' + str(i) + '.npy',to_copy)
            i = getNumber(files[0]) - 1
            files = os.listdir(out_path + folder)
        #video non arriva a 448
        if getNumber((files[len(files) - 1])) < NUM_FRAME:
            to_copy = np.load(out_path + folder + '/' + files[len(files) - 1])
            for i in range(getNumber((files[len(files) - 1])) + 1,NUM_FRAME + 1):
                np.save(out_path + folder + '/' + folder + '_000' + str(i) + '.npy',to_copy)
                files = os.listdir(out_path + folder)
                i = 0

        # caso base buchi nel mezzo
        while(i < NUM_FRAME - 10):

            if getNumber(files[i]) > i+1:
                # predire precedente
                vector = prediction(folder, files[i - 1], files[i],out_path)
                to_fill = getNumber(files[i]) #npy to fill
                j = i + 1
                while(j < to_fill):
                    if getNumber(files[i]) <= 10:
                        path = folder + '/' + folder + '_00000' + str(j) + '.npy'
                    elif getNumber(files[i]) > 100 :
                        path = folder + '/' + folder + '_000' + str(j) + '.npy'
                    else: #due cifre
                        path = folder + '/' + folder + '_0000' + str(j) + '.npy'

                    print("I'm saving path ",path)
                    np.save(out_path + path,vector)
                    j = j + 1
                    files = os.listdir(out_path + folder)

            i = i + 1
        removing_files(folder,files,out_path)

def main():
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
    else:
        out_path = 'output/'

    print("Output path = ",out_path)
    fix(out_path)

if __name__ == '__main__':
    main()
