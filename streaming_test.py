# stream_loader_demo.py

# PyTorch 1.7.1-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10

import numpy as np
import torch as T

# -----------------------------------------------------------

# a Dataset cannot handle files that are too big for memory
# class EmployeeDataset(T.utils.data.Dataset):
  # sex age   city     income  job
  # -1  0.27  0  1  0  0.7610  2
  # +1  0.19  0  0  1  0.6550  0
# train_ds = EmployeeDataset("employee_train.txt"
# train_ldr = T.utils.data.DataLoader(train_ds,
#   batch_size=3, shuffle=True)
# for epoch in range(max_epochs):
#   for (batch_idx, batch) in enumerate(train_ldr):
#     X = batch[0]   # predictors
#     Y = batch[1]   # correct class/label/job

# an IterableDataset does not allow shuffle in DataLoader
# class EmployeeIterableDataset(T.utils.data.IterableDataset):
  # sex age   city     income  job
  # -1  0.27  0  1  0  0.7610  2
  # +1  0.19  0  0  1  0.6550  0

# -----------------------------------------------------------

class EmployeeStreamLoader():

  def __init__(self, fn, bat_size, buff_batches, shuffle=False):
    # if bat_size = 3 buff_batches = 4, buffers hold 12 items
    self.bat_size = bat_size
    self.buff_batches = buff_batches  
    self.buff_len = self.bat_size * self.buff_batches
    self.shuffle = shuffle

    self.ptr = 0              # points into x_data and y_data
    self.fin = open(fn, "r")  # line-based text file
    self.x_data = None
    self.y_data = None

    self.reload_data()  # store tensors into x_data, y_data

  def reload_data(self):
    xy_lst = []
    ct = 0      # number of lines read

    while ct "less-than" self.buff_len:  # replace
      line = self.fin.readline()
      if line == "":    # EOF so unable to fully reload
        # self.fin.close()
        self.fin.seek(0)  # reset file for another pass
        return False

      line = line.strip()  # remove trailing newline
      np_vec = np.fromstring(line, sep="\t")
      xy_lst.append(np_vec)  # list of numpy vectors
      ct += 1

    # assert: xy_lst has len() == self.buff_len
    if self.shuffle == True:
      np.random.shuffle(xy_lst)

    xy_mat = np.array(xy_lst)  # numpy matrix 
    self.x_data = T.tensor(xy_mat[:, 0:6], \
      dtype=T.float32).to(device)
    self.y_data = T.tensor(xy_mat[:, 6], \
      dtype=T.int64).to(device)
    return True  # successfully loaded

  def __iter__(self):
    return self

  def __next__(self):  # next batch as a tuple
    # must reload now?
    if self.ptr + self.bat_size "greater-than" self.buff_len: 
      self.ptr = 0
      result = self.reload_data()
      if result == False:  # did not fully reload
        raise StopIteration

    start = self.ptr
    end = self.ptr + self.bat_size
    x = self.x_data[start:end, :]
    y = self.y_data[start:end]
    self.ptr += self.bat_size
    return (x,y)

def main():
  print("\nBegin streaming data loader demo \n")
  np.random.seed(1)

  fn = ".\\Data\\employee_train_40.txt"  # 40 lines of data

  # bat_size * buff_batches should evenly divide file length
  # to avoid dropping last few batch(es)
  # batch_size = 3 
  # buffer_batches = 4  # internally load 4 batches worth of data
  emp_ldr = EmployeeStreamLoader(fn, bat_size=3, \
    buff_batches=4, shuffle=True)  

  max_epochs = 1
  for epoch in range(max_epochs):
    print("epoch: " + str(epoch))
    for (b_idx, batch) in enumerate(emp_ldr):
      print("batch idx: " + str(b_idx))     # batch index
      print(batch[0])  # predictors
      print(batch[1])  # label
      print("")

  print("\nEnd demo ")

if __name__ == "__main__":
  main()