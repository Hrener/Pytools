import time
from multiprocessing import Pool
def run(fn) :
   time.sleep(2)
   print(fn)
if __name__ == "__main__" :
     startTime = time.time()
     testFL = [1,2,3,4,5]
     pool = Pool(10)#可以同时跑10个进程
     pool.map(run,testFL)
     pool.close()
     pool.join()
     endTime = time.time()
     print("time :", endTime - startTime)