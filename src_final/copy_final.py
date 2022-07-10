import shutil
import os

workdir = os.path.dirname(os.getcwd())
source_dir = '/scratch/qhaomin/pf_ms/'
destination_dir = workdir + '/output/'

N = 100_000
T = 282
batch_num = 135

for i in range(batch_num):
    case = 'actual data, seed = ' + str(i) + ', T = ' + str(T) + ', N = ' + str(N)
    casedir = destination_dir + case  + '/'
    try: 
        os.mkdir(casedir)
        shutil.copy(source_dir + case  + '/θ_282.pkl', casedir)
    except:
        pass
