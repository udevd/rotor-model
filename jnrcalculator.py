
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
class JNRCalculator:
    def __init__(self, opers, sparse=True):
        self.opers=opers
        self.jnr_dim=len(self.opers)
        self.data=[]
        self.sparse=sparse
    def support_point(self, normal):
        F=sum(n*X for (n,X) in zip(normal,self.opers)) #np functions cause MemoryError
        (energy,vector)=F.groundstate(sparse=self.sparse)
        expvals=np.array([X.matrix_element(vector,vector) for X in self.opers])
        self.data.append((normal,expvals))
    def random_points(self, npoints):
        normals = np.random.randn(npoints, self.jnr_dim)
        normals /= np.linalg.norm(normals, axis=0)


        for normal in tqdm(normals):
            self.support_point(normal)
            bar.next()

    def pts_coordinates(self, i):
        return [p[i].real for (n,p) in self.data]
    def draw_2d(self, i, j):
        xs=self.pts_coordinates(i)
        ys=self.pts_coordinates(j)
        plt.scatter(xs,ys)
    def draw_3d(self, i, j, k):
        xs=self.pts_coordinates(i)
        ys=self.pts_coordinates(j)
        zs=self.pts_coordinates(j)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs,ys,zs)
