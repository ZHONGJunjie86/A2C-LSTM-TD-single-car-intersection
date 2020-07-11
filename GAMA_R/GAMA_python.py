import time
import os
import csv
import numpy as np

def connect():
    f=open('D:/Software/GamaWorkspace/Python/from_GAMA.csv', "r+")
    f.truncate()
    #等
    while (os.stat("D:/Software/GamaWorkspace/Python/from_GAMA.csv").st_size == 0):
        time.sleep(1)
        print("wait")
    #加载
    state = np.loadtxt("D:/Software/GamaWorkspace/Python/from_GAMA.csv", delimiter=",")
    #清空
    f=open('D:/Software/GamaWorkspace/Python/from_GAMA.csv', "r+")
    f.truncate()

    print("Recived:",state)
    #f = open('D:/Software/GamaWorkspace/Python/from_python.csv','w',encoding='utf-8')
    #csv_writer = csv.writer(f)
    #csv_writer.writerow([1,state + 1])
    #f.close()
    return_ = [[1,state + 1]]
    np.savetxt('D:/Software/GamaWorkspace/Python/from_python.csv',return_,delimiter=',')

def main():
    while(True):
        connect()

if __name__ == '__main__':
    main()


    #b = np.loadtxt("D:/Software/GamaWorkspace/Python/save_data.csv", delimiter=",")
    #print("b[0]",b[0])
    #time.sleep(3)
    #data = np.array([1,b.sum(axis=0)]).reshape(1,2)
    #data = []
    #np.savetxt("D:/Software/GamaWorkspace/Python/GAMA_intersection_data.csv",data,delimiter=',')
    #print(os.stat("D:/Software/GamaWorkspace/Python/GAMA_intersection_data.csv").st_size == 0)
    

    #df_test = pd.read_csv("D:/Software/GamaWorkspace/Python/from_python.csv") # or pd.read_excel(filename) for xls file
    #print(df_test.empty)