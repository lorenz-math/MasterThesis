#this file was made long after handing in my thesis and is used to generate pretty pictures for the paper submission
from ImportFile import *
from mpl_toolkits import mplot3d
from scipy.interpolate import CubicSpline
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


pi = math.pi

a=2*pi
T=0.5
w=pi*math.sqrt(2 - (a ** 2/(4 * pi ** 2)))


#Omega = [-0.5,0.5]^2, t\in[0,0.5]
extrema_values = torch.tensor([[0, T],
                               [-0.5, 0.5],
                               [-0.5, 0.5]])

"""
#code für plotten der Lösungen, der PINNs und des Errors in colourmaps
def exact(inputs):
    t = inputs[:, 0]
    x = inputs[:, 1]
    y = inputs[:, 2]

    u=(torch.exp(- (a/2)*t)*(torch.cos(w * t)+torch.sin(w * t)) * torch.cos(pi * x) * torch.cos(pi * y)).reshape(-1,1)

    return u


trained_model = torch.load(os.path.join("Run05_Fin", "3125_15625_0", "Sample_0", "Retrain_0", "TrainedModel", "model.pkl"), map_location=torch.device('cpu'))

spacing = np.linspace(-0.5, 0.5, 100, endpoint=True) 
zeros = np.full(spacing.size**2, 0.5)
x = np.repeat(spacing, spacing.size)
y = np.tile(np.flip(spacing), spacing.size)
XY=np.vstack((zeros, x,y)).T
XY = torch.from_numpy(XY)
ExSol = exact(XY).numpy()
ExSol = ExSol.reshape(spacing.size,spacing.size).T
z = trained_model(XY.float()).detach().numpy()
z = z.reshape(spacing.size,spacing.size).T
c = plt.imshow(z, cmap =plt.cm.RdBu, vmin = 0, vmax = 1.0, extent = [spacing.min(), spacing.max(), spacing.min(), spacing.max()]) 
plt.colorbar(c) 
plt.title("PINN solution at T=0.5, 18750 training points")
plt.show()

Error = np.subtract(ExSol, z)
ErrorPlot = plt.imshow(Error, cmap =plt.cm.Spectral, vmax = 0.007, vmin = -0.007, extent = [spacing.min(), spacing.max(), spacing.min(), spacing.max()]) 
#ErrorPlot = plt.contourf(Error, levels = 3, cmap =plt.cm.Spectral, vmax = 0.007, vmin = -0.007, extent = [spacing.min(), spacing.max(), spacing.min(), spacing.max()]) 
plt.colorbar(ErrorPlot) 
plt.title("Error at T=0.5, 18750 training points")
plt.show()

"""
#code für plotten der Traininspunkte in 3D und über der PINN Lösung im ersten time slice
space_dimensions = 2
time_dimensions = 1
input_dimensions = time_dimensions + space_dimensions

extrema_0 = extrema_values[:, 0]
extrema_f = extrema_values[:, 1]


def generator_points(samples, dim):
    h=round(samples ** (1/dim))
    midpoint_1D=np.arange(1/(2*h),1,(1/h))
    midpoint_nD=list(it.product(midpoint_1D, repeat=dim))
    return torch.FloatTensor(midpoint_nD)

n_coll = 1000
n_u = 500
n_b = int(n_u / 5)
n_i = int(n_u / 5)

x_coll = generator_points(n_coll, input_dimensions)
x_coll = x_coll * (extrema_f - extrema_0) + extrema_0

x_list_b = list()
for i in range(time_dimensions, time_dimensions + space_dimensions):
    x_boundary_0 = generator_points(n_b, space_dimensions)
    zero = torch.zeros(n_b,1)
    x_boundary_0 = torch.cat([x_boundary_0[:,:i], zero, x_boundary_0[:,i:]],1)
    x_list_b.append(x_boundary_0)
    x_boundary_1 = generator_points(n_b, space_dimensions)
    one = torch.full((n_b,1), 1.)
    x_boundary_1 = torch.cat([x_boundary_1[:,:i], one, x_boundary_1[:,i:]],1)
    x_list_b.append(x_boundary_1)
x_b = torch.cat(x_list_b, 0)
x_b = x_b * (extrema_f - extrema_0) + extrema_0


x_time_0 = generator_points(n_i, space_dimensions)
padding = nn.ConstantPad2d((1,0,0,0),0)
x_time_0 = padding(x_time_0)
x_time_0 = x_time_0 * (extrema_f - extrema_0) + extrema_0

x_time_0=x_time_0.numpy()
tt,xt,yt = x_time_0.T
x_b=x_b.numpy()
tb, xb, yb = x_b.T
x_coll = x_coll.numpy()
tcoll,xcoll,ycoll = x_coll.T

fig = plt.figure()
ax = plt.axes(projection ="3d")

ax.scatter3D(xt,yt,tt, color= 'green', label = 'Initial points')
ax.scatter3D(xb,yb,tb, color = 'orange', label = 'Boundary points')
ax.scatter3D(xcoll,ycoll,tcoll, color = 'blue', label = 'Interior points')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('t')
plt.legend()
plt.show()


trained_model = torch.load(os.path.join("Run05_Fin", "500_1000_0", "Sample_0", "Retrain_0", "TrainedModel", "model.pkl"), map_location=torch.device('cpu'))
spacing = np.linspace(-0.5, 0.5, 100, endpoint=True) 
zeros = np.full(spacing.size**2, 0.025)
x = np.repeat(spacing, spacing.size)
y = np.tile(np.flip(spacing), spacing.size)
XY=np.vstack((zeros, x,y)).T
XY = torch.from_numpy(XY)
#ExSol = exact(XY).numpy()
#ExSol = ExSol.reshape(spacing.size,spacing.size).T
z = trained_model(XY.float()).detach().numpy()
z = z.reshape(spacing.size,spacing.size).T
c = plt.imshow(z, cmap =plt.cm.RdBu, vmin = 0, vmax = 1.0, extent = [spacing.min(), spacing.max(), spacing.min(), spacing.max()]) 
plt.colorbar(c) 

#x_coll = x_coll.numpy()
x_coll = x_coll[x_coll[:,0]==0.0250]
tcoll,xcoll,ycoll = x_coll.T
#x_b = x_b.numpy()
x_b = x_b[x_b[:,0]==0.0250]
tb, xb, yb = x_b.T
plt.scatter(xb,yb,color = 'orange', marker='x', label = 'Boundary points')
plt.scatter(xcoll,ycoll,color = 'blue', marker='x', label = 'Interior points')
plt.legend(loc = 'upper right')
plt.title("T=0.025, 1500 training points")

plt.show()


"""
#code für plot der individuellen Training errors
mean_df = pd.read_csv("Run05_Fin\meanConvergenceAnalysis.csv")
mean_df["E_T"]=np.sqrt(mean_df["res_loss"] ** 2 + mean_df["loss_su"] ** 2 
                              + mean_df["loss_sut"] ** 2 + mean_df["loss_u0"] ** 2 
                              + mean_df["loss_u1"] ** 2 + mean_df["loss_nablau"] ** 2)
mean_df['Nu']=mean_df['Nu_train']+mean_df['Nf_train']
plt.grid(True, which="both", ls=":")

cs1 = CubicSpline(mean_df.Nu.values, mean_df.E_T.values)
cs2 = CubicSpline(mean_df.Nu.values, mean_df.res_loss.values)
cs3 = CubicSpline(mean_df.Nu.values, mean_df.loss_su.values)
cs4 = CubicSpline(mean_df.Nu.values, mean_df.loss_sut.values)
cs5 = CubicSpline(mean_df.Nu.values, mean_df.loss_u0.values)
cs6 = CubicSpline(mean_df.Nu.values, mean_df.loss_u1.values)
cs7 = CubicSpline(mean_df.Nu.values, mean_df.loss_nablau.values)
plt.plot(mean_df.Nu.values, cs1(mean_df.Nu.values),label=r'$\varepsilon_{Train}$', color='gray')
plt.plot(mean_df.Nu.values, cs2(mean_df.Nu.values), label=r'$\varepsilon^{PDE}_T$', color='forestgreen')
plt.plot(mean_df.Nu.values, cs3(mean_df.Nu.values), label=r'$\varepsilon^{s,u}_T$', color='red')
plt.plot(mean_df.Nu.values, cs4(mean_df.Nu.values), label=r'$\varepsilon^{s,u_t}_T$', color='blue')
plt.plot(mean_df.Nu.values, cs5(mean_df.Nu.values), label=r'$\varepsilon^{u_0}_T$', color='yellow')
plt.plot(mean_df.Nu.values, cs6(mean_df.Nu.values), label=r'$\varepsilon^{u_1}_T$', color='orange')
plt.plot(mean_df.Nu.values, cs7(mean_df.Nu.values), label=r'$\varepsilon^{\nabla u}_T$', color='purple')

plt.legend()
plt.xscale("log")
plt.xlabel(r'$M$')
plt.yscale("log")
plt.show()

mean_df = pd.read_csv("Run05_Fin\meanConvergenceAnalysis.csv")
mean_df['Nu']=mean_df['Nu_train']+mean_df['Nf_train']
plt.grid(True, which="both", ls=":")
cs = CubicSpline(mean_df.Nu.values, mean_df.train_time.values)
plt.plot(mean_df.Nu.values, cs(mean_df.Nu.values))
plt.xlabel(r'$M$')
plt.ylabel(r'$s$')
plt.title("Average training time")
plt.show()
"""