import numpy as np
import sys
import os
import re
import h5py
import mmap


def orb(line):
  num_orb = line.split()[0]
  print(num_orb + ' orbitals found')
  
  return int(num_orb)

def find_start2el(inf):
  pattern = r'END'
  lines = next(inf.index(i) for i in inf if re.search(pattern, i) is not None)
  print(int(lines) + 1)

  return int(lines) + 1

def find_end2el(inf):
  pattern = r'(\s+)1(\s+)1(\s+)0(\s+)0(\s+)'
  rinf = inf[::-1]
  rlines = next(rinf.index(i) for i in rinf if re.search(pattern, i) is not None)
  lines = len(inf)-1-rlines
  print(int(lines) - 1)

  return int(lines) - 1


ffock = sys.argv[1]
feri = sys.argv[2]

#################
# for Fock matrix
#################
ffock = open(ffock, 'rb')

# load the whole file
fmapped = mmap.mmap(ffock.fileno(), 0, access=mmap.ACCESS_READ)

inf = []
for line in iter(fmapped.readline, b""):
  inf.append(line.decode('utf-8'))

# get number of primitives
num_orb = orb(inf[-1])

fock = np.zeros((num_orb,num_orb), dtype=np.float32)
for f in inf:
  i,a = [x-1 for x in map(int, f.split()[:-1])]
  val = float(f.split()[-1])
  fock[i,a] = val
  fock[a,i] = val

################
# for ERI matrix
################
feri = open(feri, 'rb')

# load the whole file
fmapped = mmap.mmap(feri.fileno(), 0, access=mmap.ACCESS_READ)

inf = []
for line in iter(fmapped.readline, b""):
  inf.append(line.decode('utf-8'))

# feed the value to matrix
eri = np.zeros((num_orb,num_orb,num_orb,num_orb), dtype=np.float32)
for t in inf:
  i,j,a,b = [x-1 for x in map(int, t.split()[:-1])]
  val = float(t.split()[-1])
  eri[i,j,a,b] = val
  eri[i,b,a,j] = val
  eri[a,j,i,b] = val
  eri[a,b,i,j] = val
  eri[j,i,b,a] = val
  eri[b,i,j,a] = val
  eri[j,a,b,i] = val
  eri[b,a,j,i] = val

# save it to npy
np.save('F.npy',np.asarray(fock, dtype=np.float32))
np.save('ERI.npy',np.asarray(eri, dtype=np.float32))       





