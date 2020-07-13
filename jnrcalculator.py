
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm.auto import tqdm
from multiprocessing import Pool, Lock
import numpy as np
import contextlib
import sys
import pickle
@contextlib.contextmanager
def suppress_stdout(suppress=True):
    std_ref = sys.stdout
    if suppress:
        sys.stdout = open('/dev/null', 'w')
        yield
    sys.stdout = std_ref

def _calculate_support_point(opers,normal,sparse=True): #it has to be like this - multiprocesssing pickles functions to call and they *have* to be on the top level
    F=sum(n*X for (n,X) in zip(normal,opers)) #np functions cause MemoryError
    with suppress_stdout():
        (energy,vector)=F.groundstate(sparse=sparse)    
    expvals=np.array([X.matrix_element(vector,vector).real for X in opers])
    return (normal,expvals)


class GeneralJNRCalculator:
    def __init__(self, calculator,jnr_dim, filename):
        self.calculator=calculator
        self.filename=filename
        self.jnr_dim=jnr_dim
        self.data=[]

    def process_random_points(self, npoints):
        normals = np.random.randn(npoints, self.jnr_dim)
        normals /= np.linalg.norm(normals, axis=0)
        bar = tqdm(total=npoints)
        processes=[]
        def update(result):
            self.data.append(np.array(result))
            with open(self.filename,'wb') as handle:
                pickle.dump(self, handle)
            bar.update()
        with Pool() as p:
            for normal in normals:
                    p.apply_async(self.calculator,args=(normal,),callback=update)
            p.close()
            p.join()
            
    
    def process_random_points_sync(self, npoints):
        normals = np.random.randn(npoints, self.jnr_dim)
        normals /= np.linalg.norm(normals, axis=0)
        bar = tqdm(total=npoints)
        processes=[]
        def update(result):
            self.data.append(np.array(result))
            with open(self.filename,'wb') as handle:
                pickle.dump(self, handle)
            bar.update()
        for normal in normals:
            update(self.calculator(normal))

    def pts_coordinates(self, i):
        return [p[i].real for (n,p) in self.data]
    def draw_2d(self, i, j):
        xs=self.pts_coordinates(i)
        ys=self.pts_coordinates(j)
        plt.scatter(xs,ys)
    def draw_3d(self, i, j, k):
        xs=self.pts_coordinates(i)
        ys=self.pts_coordinates(j)
        zs=self.pts_coordinates(k)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs,ys,zs)
    def dump_csv(self,filename): # each line contains normal and points concatenated to 2dim vector
        np.savetxt(filename,np.array(self.data).reshape(-1,2*self.jnr_dim))


class JNRCalculator(GeneralJNRCalculator):
    def __init__(self, opers, filename, sparse=True):
        def calculator(normal):
            return _calculate_support_point(opers,normal,sparse=sparse)
        super().__init__(calculator, len(self.opers),filename)
        self.opers=opers
        self.sparse=sparse

