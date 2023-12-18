#this file loads the trained models and computes the constants for the bound

from ImportFile import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pi = math.pi
a=2*pi
w=pi*math.sqrt(2 - (a ** 2/(4 * pi ** 2)))
T=0.5

extrema_values = torch.tensor([[0, T],
                               [-0.5, 0.5],
                               [-0.5, 0.5]])

space_dimensions = 2
time_dimensions = 1
input_dimensions = time_dimensions + space_dimensions


#exact solution
def exact(inputs):
    t = inputs[:, 0]
    x = inputs[:, 1]
    y = inputs[:, 2]

    u=(torch.exp(- (a/2)*t)*(torch.cos(w * t)+torch.sin(w * t)) * torch.cos(pi * x) * torch.cos(pi * y)).reshape(-1,1)

    return u

#u_t
def u_t(inputs):
    t = inputs[:, 0]
    x = inputs[:, 1]
    y = inputs[:, 2]

    u=(- (a/2) * torch.exp(- (a/2)*t)*(torch.cos(w * t)+torch.sin(w * t)) * torch.cos(pi * x) * torch.cos(pi * y) + w * torch.exp(- (a/2)*t)*(torch.cos(w * t)-torch.sin(w * t)) * torch.cos(pi * x) * torch.cos(pi * y)).reshape(-1,1)

    return u

#u(t=0) and its derivatives
def u0(inputs):
    x = inputs[:, 1]
    y = inputs[:, 2]
    return(torch.cos(pi * x)*torch.cos(pi * y)).reshape(-1,1)  

def dx_u0(inputs):
    x = inputs[:, 1]
    y = inputs[:, 2]
    return -pi*torch.sin(pi*x)*torch.cos(pi*y)

def dy_u0(inputs):
    x = inputs[:, 1]
    y = inputs[:, 2]
    return -pi*torch.cos(pi*x)*torch.sin(pi*y)

#u_t(t=0)
def u1(inputs):
    x = inputs[:, 1]
    y = inputs[:, 2]
    return (- (a/2)*torch.cos(pi * x)*torch.cos(pi * y) + w * torch.cos(pi * x)*torch.cos(pi * y)).reshape(-1,1)

#This stretches a tensor filled with random uniformly distributed points on [0,1]^3 to Omega
def convert(vector, extrema_values):
    vector = np.array(vector)
    max_val = np.max(np.array(extrema_values), axis=1)
    min_val = np.min(np.array(extrema_values), axis=1)
    vector = vector * (max_val - min_val) + min_val
    return torch.from_numpy(vector).type(torch.FloatTensor)


# computing training error and constants for bounds
def select_over_retrainings2(folder_path, selection="error_train", mode="mean", compute_std=False, compute_val=False, rs_val=0):
    retrain_models = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    models_list = list()
    for retraining in retrain_models:
        rs = int(folder_path.split("_")[-1])
        retrain_path = os.path.join(folder_path, retraining)
        number_of_ret = retraining.split("_")[-1]

        if os.path.isfile(os.path.join(retrain_path, "InfoModel.txt")):
            models = pd.read_csv(os.path.join(retrain_path, "InfoModel.txt"), header=0, sep=",")
            models["retraining"] = number_of_ret

            if os.path.isfile(os.path.join(retrain_path, "TrainedModel", "model.pkl")):
                #load model
                trained_model = torch.load(os.path.join(retrain_path, "TrainedModel", "model.pkl"))
                if torch.cuda.is_available():
                    trained_model.cuda()
                trained_model.eval()

                extrema_0 = extrema_values[:, 0]
                extrema_f = extrema_values[:, 1]

                test_inp = convert(torch.rand([10000, extrema_values.shape[0]]), extrema_values)
                if torch.cuda.is_available():
                    test_inp = test_inp.cuda()
                test_inp.requires_grad=True
                
                #compute the total error of the time derivative
                Exact = (u_t(test_inp)).cpu().detach().numpy()
                u_u=trained_model(test_inp).reshape(-1, )
                inputs = torch.ones(test_inp.shape[0], )
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                grad_u= torch.autograd.grad(u_u, test_inp, grad_outputs=inputs, create_graph=True)[0]
                grad_u_t = grad_u[:, 0].reshape(-1, )
                grad_u_t = torch.unsqueeze(grad_u_t, dim = -1)
                grad_u_t = grad_u_t.cpu().detach().numpy()
                L2_t_test = np.sqrt(np.mean(((Exact - grad_u_t) ** 2)))
                models["L2_t_test"]=L2_t_test

                
                
                # C^2 norm of for the squared residuals
                #first: R_PDE^2, evaluated on 100000 random test points in the domain
                res_train = compute_residual(trained_model, test_inp) ** 2
                u_C0 = torch.max(res_train).cpu().detach().numpy()                

                #first derivatives
                grad_res_t= torch.autograd.grad(res_train, test_inp, grad_outputs=inputs, create_graph=True)[0][:,0].reshape(-1, )
                grad_res_x= torch.autograd.grad(res_train, test_inp, grad_outputs=inputs, create_graph=True)[0][:,1].reshape(-1, )
                grad_res_y= torch.autograd.grad(res_train, test_inp, grad_outputs=inputs, create_graph=True)[0][:,2].reshape(-1, )
                max_grad_res_t= torch.max(torch.abs(grad_res_t)).cpu().detach().numpy()
                max_grad_res_x= torch.max(torch.abs(grad_res_x)).cpu().detach().numpy()
                max_grad_res_y= torch.max(torch.abs(grad_res_y)).cpu().detach().numpy()
                u_C1 = max(u_C0, max_grad_res_t, max_grad_res_x, max_grad_res_y)

                #second derivatives
                grad_res_tt= torch.autograd.grad(grad_res_t, test_inp, grad_outputs=inputs, create_graph=True)[0][:,0].reshape(-1, )
                grad_res_tx= torch.autograd.grad(grad_res_t, test_inp, grad_outputs=inputs, create_graph=True)[0][:,1].reshape(-1, )
                grad_res_ty= torch.autograd.grad(grad_res_t, test_inp, grad_outputs=inputs, create_graph=True)[0][:,2].reshape(-1, )
                grad_res_xt= torch.autograd.grad(grad_res_x, test_inp, grad_outputs=inputs, create_graph=True)[0][:,0].reshape(-1, )
                grad_res_xx= torch.autograd.grad(grad_res_x, test_inp, grad_outputs=inputs, create_graph=True)[0][:,1].reshape(-1, )
                grad_res_xy= torch.autograd.grad(grad_res_x, test_inp, grad_outputs=inputs, create_graph=True)[0][:,2].reshape(-1, )
                grad_res_yt= torch.autograd.grad(grad_res_y, test_inp, grad_outputs=inputs, create_graph=True)[0][:,0].reshape(-1, )
                grad_res_yx= torch.autograd.grad(grad_res_y, test_inp, grad_outputs=inputs, create_graph=True)[0][:,1].reshape(-1, )
                grad_res_yy= torch.autograd.grad(grad_res_y, test_inp, grad_outputs=inputs, create_graph=True)[0][:,2].reshape(-1, )

                max_grad_res_tt= torch.max(torch.abs(grad_res_tt)).cpu().detach().numpy()
                max_grad_res_tx= torch.max(torch.abs(grad_res_tx)).cpu().detach().numpy()
                max_grad_res_ty= torch.max(torch.abs(grad_res_ty)).cpu().detach().numpy()
                max_grad_res_xt= torch.max(torch.abs(grad_res_xt)).cpu().detach().numpy()
                max_grad_res_xx= torch.max(torch.abs(grad_res_xx)).cpu().detach().numpy()
                max_grad_res_xy= torch.max(torch.abs(grad_res_xy)).cpu().detach().numpy()
                max_grad_res_yt= torch.max(torch.abs(grad_res_yt)).cpu().detach().numpy()
                max_grad_res_yx= torch.max(torch.abs(grad_res_yx)).cpu().detach().numpy()
                max_grad_res_yy= torch.max(torch.abs(grad_res_yy)).cpu().detach().numpy()
                R_PDE = max(u_C1, max_grad_res_tt, max_grad_res_tx, max_grad_res_ty, max_grad_res_xt, max_grad_res_xx, max_grad_res_xy, max_grad_res_yt, max_grad_res_yx, max_grad_res_yy)
                models["R_PDE"]=R_PDE
                

                #next: R_s,u_t^2 
                #create 4000 random test points on the boundary
                test_inp_b_list = list()
                for i in range(time_dimensions, time_dimensions + space_dimensions):
                    x_boundary_0 = torch.rand([1000, space_dimensions])
                    zero = torch.zeros(1000,1)
                    x_boundary_0 = torch.cat([x_boundary_0[:,:i], zero, x_boundary_0[:,i:]],1)
                    test_inp_b_list.append(x_boundary_0)
                    x_boundary_1 = torch.rand([1000, space_dimensions])
                    one = torch.full((1000,1), 1.)
                    x_boundary_1 = torch.cat([x_boundary_1[:,:i], one, x_boundary_1[:,i:]],1)
                    test_inp_b_list.append(x_boundary_1)
                test_inp_b = torch.cat(test_inp_b_list, 0)
                test_inp_b = test_inp_b * (extrema_f - extrema_0) + extrema_0

                inputs_b = torch.ones(test_inp_b.shape[0], )

                if torch.cuda.is_available():
                    test_inp_b = test_inp_b.cuda()
                    inputs_b = inputs_b.cuda()
                
                test_inp_b.requires_grad = True
                
                u_b = trained_model(test_inp_b).reshape(-1,)
                grad_u_b = torch.autograd.grad(u_b, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0]
                grad_u_b_t = grad_u_b[:, 0].reshape(-1, )
                u_pred_var_list=list()
                u_pred_var_list.append(grad_u_b_t)
                u_pred_var_list = torch.cat(u_pred_var_list, 0)
                if torch.cuda.is_available():
                    u_pred_var_list=u_pred_var_list.cuda()
                R_sut_train=u_pred_var_list ** 2

                u_C0 = torch.max(R_sut_train).cpu().detach().numpy()               

                #first derivatives
                grad_sut_t= torch.autograd.grad(R_sut_train, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,0].reshape(-1, )
                grad_sut_x= torch.autograd.grad(R_sut_train, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,1].reshape(-1, )
                grad_sut_y= torch.autograd.grad(R_sut_train, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,2].reshape(-1, )
                max_grad_sut_t= torch.max(torch.abs(grad_sut_t)).cpu().detach().numpy()
                max_grad_sut_x= torch.max(torch.abs(grad_sut_x)).cpu().detach().numpy()
                max_grad_sut_y= torch.max(torch.abs(grad_sut_y)).cpu().detach().numpy()
                u_C1 = max(u_C0, max_grad_sut_t, max_grad_sut_x, max_grad_sut_y)

                #second derivatives
                grad_sut_tt= torch.autograd.grad(grad_sut_t, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,0].reshape(-1, )
                grad_sut_tx= torch.autograd.grad(grad_sut_t, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,1].reshape(-1, )
                grad_sut_ty= torch.autograd.grad(grad_sut_t, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,2].reshape(-1, )
                grad_sut_xt= torch.autograd.grad(grad_sut_x, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,0].reshape(-1, )
                grad_sut_xx= torch.autograd.grad(grad_sut_x, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,1].reshape(-1, )
                grad_sut_xy= torch.autograd.grad(grad_sut_x, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,2].reshape(-1, )
                grad_sut_yt= torch.autograd.grad(grad_sut_y, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,0].reshape(-1, )
                grad_sut_yx= torch.autograd.grad(grad_sut_y, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,1].reshape(-1, )
                grad_sut_yy= torch.autograd.grad(grad_sut_y, test_inp_b, grad_outputs=inputs_b, create_graph=True)[0][:,2].reshape(-1, )

                max_grad_sut_tt= torch.max(torch.abs(grad_sut_tt)).cpu().detach().numpy()
                max_grad_sut_tx= torch.max(torch.abs(grad_sut_tx)).cpu().detach().numpy()
                max_grad_sut_ty= torch.max(torch.abs(grad_sut_ty)).cpu().detach().numpy()
                max_grad_sut_xt= torch.max(torch.abs(grad_sut_xt)).cpu().detach().numpy()
                max_grad_sut_xx= torch.max(torch.abs(grad_sut_xx)).cpu().detach().numpy()
                max_grad_sut_xy= torch.max(torch.abs(grad_sut_xy)).cpu().detach().numpy()
                max_grad_sut_yt= torch.max(torch.abs(grad_sut_yt)).cpu().detach().numpy()
                max_grad_sut_yx= torch.max(torch.abs(grad_sut_yx)).cpu().detach().numpy()
                max_grad_sut_yy= torch.max(torch.abs(grad_sut_yy)).cpu().detach().numpy()
                R_sut = max(u_C1, max_grad_sut_tt, max_grad_sut_tx, max_grad_sut_ty, max_grad_sut_xt, max_grad_sut_xx, max_grad_sut_xy, max_grad_sut_yt, max_grad_sut_yx, max_grad_sut_yy)
                models["R_sut"]=R_sut
                
                #R_u0^2
                #create 10000 random test points for t=0
                test_inp_0 = torch.rand([10000, space_dimensions])
                padding = nn.ConstantPad2d((1,0,0,0),0)
                test_inp_0 = padding(test_inp_0)
                test_inp_0 = test_inp_0 * (extrema_f - extrema_0) + extrema_0

                inputs_0 = torch.ones(test_inp_0.shape[0], )

                if torch.cuda.is_available():
                    test_inp_0 = test_inp_0.cuda()
                    inputs_0 = inputs_0.cuda()

                test_inp_0.requires_grad=True

                u_pred_var_list = list()
                u_train_var_list = list()
                u_pred_var_list.append(trained_model(test_inp_0))
                u_train_var_list.append(u0(test_inp_0))
                u_pred_var_list = torch.squeeze(torch.cat(u_pred_var_list, 0))
                u_train_var_list = torch.squeeze(torch.cat(u_train_var_list, 0))
                if torch.cuda.is_available():
                    u_pred_var_list=u_pred_var_list.cuda()
                    u_train_var_list=u_train_var_list.cuda()
                R_u0_train=(u_pred_var_list - u_train_var_list) ** 2


                u_C0 = torch.max(R_u0_train).cpu().detach().numpy()                

                #first derivatives
                grad_u0_t= torch.autograd.grad(R_u0_train, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_u0_x= torch.autograd.grad(R_u0_train, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_u0_y= torch.autograd.grad(R_u0_train, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )
                max_grad_u0_t= torch.max(torch.abs(grad_u0_t)).cpu().detach().numpy()
                max_grad_u0_x= torch.max(torch.abs(grad_u0_x)).cpu().detach().numpy()
                max_grad_u0_y= torch.max(torch.abs(grad_u0_y)).cpu().detach().numpy()
                u_C1 = max(u_C0, max_grad_u0_t, max_grad_u0_x, max_grad_u0_y)

                #second derivatives
                grad_u0_tt= torch.autograd.grad(grad_u0_t, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_u0_tx= torch.autograd.grad(grad_u0_t, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_u0_ty= torch.autograd.grad(grad_u0_t, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )
                grad_u0_xt= torch.autograd.grad(grad_u0_x, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_u0_xx= torch.autograd.grad(grad_u0_x, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_u0_xy= torch.autograd.grad(grad_u0_x, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )
                grad_u0_yt= torch.autograd.grad(grad_u0_y, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_u0_yx= torch.autograd.grad(grad_u0_y, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_u0_yy= torch.autograd.grad(grad_u0_y, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )

                max_grad_u0_tt= torch.max(torch.abs(grad_u0_tt)).cpu().detach().numpy()
                max_grad_u0_tx= torch.max(torch.abs(grad_u0_tx)).cpu().detach().numpy()
                max_grad_u0_ty= torch.max(torch.abs(grad_u0_ty)).cpu().detach().numpy()
                max_grad_u0_xt= torch.max(torch.abs(grad_u0_xt)).cpu().detach().numpy()
                max_grad_u0_xx= torch.max(torch.abs(grad_u0_xx)).cpu().detach().numpy()
                max_grad_u0_xy= torch.max(torch.abs(grad_u0_xy)).cpu().detach().numpy()
                max_grad_u0_yt= torch.max(torch.abs(grad_u0_yt)).cpu().detach().numpy()
                max_grad_u0_yx= torch.max(torch.abs(grad_u0_yx)).cpu().detach().numpy()
                max_grad_u0_yy= torch.max(torch.abs(grad_u0_yy)).cpu().detach().numpy()
                R_u0 = max(u_C1, max_grad_u0_tt, max_grad_u0_tx, max_grad_u0_ty, max_grad_u0_xt, max_grad_u0_xx, max_grad_u0_xy, max_grad_u0_yt, max_grad_u0_yx, max_grad_u0_yy)
                models["R_u0"]=R_u0

                #R_u1^2
                u_pred_var_list = list()
                u_train_var_list = list()
                test_inp_0.requires_grad = True
                u_u=trained_model(test_inp_0).reshape(-1, )
                grad_u_u = torch.autograd.grad(u_u, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0]
                grad_u_u_t = grad_u_u[:, 0].reshape(-1, )
                u_pred_var_list.append(grad_u_u_t)
                u_train_var_list.append(u1(test_inp_0))
                u_pred_var_list = torch.squeeze(torch.cat(u_pred_var_list, 0))
                u_train_var_list = torch.squeeze(torch.cat(u_train_var_list, 0))
                if torch.cuda.is_available():
                    u_pred_var_list=u_pred_var_list.cuda()
                    u_train_var_list=u_train_var_list.cuda()
                R_u1_train=(u_pred_var_list - u_train_var_list) ** 2

                u_C0 = torch.max(R_u1_train).cpu().detach().numpy()                

                #first derivatives
                grad_u1_t= torch.autograd.grad(R_u1_train, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_u1_x= torch.autograd.grad(R_u1_train, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_u1_y= torch.autograd.grad(R_u1_train, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )
                max_grad_u1_t= torch.max(torch.abs(grad_u1_t)).cpu().detach().numpy()
                max_grad_u1_x= torch.max(torch.abs(grad_u1_x)).cpu().detach().numpy()
                max_grad_u1_y= torch.max(torch.abs(grad_u1_y)).cpu().detach().numpy()
                u_C1 = max(u_C0, max_grad_u1_t, max_grad_u1_x, max_grad_u1_y)

                #second derivatives
                grad_u1_tt= torch.autograd.grad(grad_u1_t, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_u1_tx= torch.autograd.grad(grad_u1_t, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_u1_ty= torch.autograd.grad(grad_u1_t, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )
                grad_u1_xt= torch.autograd.grad(grad_u1_x, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_u1_xx= torch.autograd.grad(grad_u1_x, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_u1_xy= torch.autograd.grad(grad_u1_x, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )
                grad_u1_yt= torch.autograd.grad(grad_u1_y, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_u1_yx= torch.autograd.grad(grad_u1_y, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_u1_yy= torch.autograd.grad(grad_u1_y, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )

                max_grad_u1_tt= torch.max(torch.abs(grad_u1_tt)).cpu().detach().numpy()
                max_grad_u1_tx= torch.max(torch.abs(grad_u1_tx)).cpu().detach().numpy()
                max_grad_u1_ty= torch.max(torch.abs(grad_u1_ty)).cpu().detach().numpy()
                max_grad_u1_xt= torch.max(torch.abs(grad_u1_xt)).cpu().detach().numpy()
                max_grad_u1_xx= torch.max(torch.abs(grad_u1_xx)).cpu().detach().numpy()
                max_grad_u1_xy= torch.max(torch.abs(grad_u1_xy)).cpu().detach().numpy()
                max_grad_u1_yt= torch.max(torch.abs(grad_u1_yt)).cpu().detach().numpy()
                max_grad_u1_yx= torch.max(torch.abs(grad_u1_yx)).cpu().detach().numpy()
                max_grad_u1_yy= torch.max(torch.abs(grad_u1_yy)).cpu().detach().numpy()
                R_u1 = max(u_C1, max_grad_u1_tx, max_grad_u1_tt, max_grad_u1_ty, max_grad_u1_xt, max_grad_u1_xx, max_grad_u1_xy, max_grad_u1_yt, max_grad_u1_yx, max_grad_u1_yy)
                models["R_u1"]=R_u1

                #R_nablau^2
                u_pred_var_list1 = list()
                u_train_var_list1 = list()
                u_pred_var_list2 = list()
                u_train_var_list2 = list()
                grad_u_u_x = grad_u_u[:, 1].reshape(-1, )
                grad_u_u_y = grad_u_u[:, 2].reshape(-1, )
                u_pred_var_list1.append(grad_u_u_x)
                u_pred_var_list2.append(grad_u_u_y)
                u_train_var_list1.append(dx_u0(test_inp_0))
                u_train_var_list2.append(dy_u0(test_inp_0))
                u_pred_var_list1 = torch.squeeze(torch.cat(u_pred_var_list1, 0))
                u_train_var_list1 = torch.squeeze(torch.cat(u_train_var_list1, 0))
                u_pred_var_list2 = torch.squeeze(torch.cat(u_pred_var_list2, 0))
                u_train_var_list2 = torch.squeeze(torch.cat(u_train_var_list2, 0))
                if torch.cuda.is_available():
                    u_pred_var_list1=u_pred_var_list.cuda()
                    u_train_var_list1=u_train_var_list.cuda()
                    u_pred_var_list2=u_pred_var_list.cuda()
                    u_train_var_list2=u_train_var_list.cuda()
                R_nablau_train=(u_pred_var_list1 - u_train_var_list1) ** 2 + (u_pred_var_list2 - u_train_var_list2)**2

                u_C0 = torch.max(R_nablau_train).cpu().detach().numpy()                

                #first derivatives
                grad_nablau_t= torch.autograd.grad(R_nablau_train, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_nablau_x= torch.autograd.grad(R_nablau_train, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_nablau_y= torch.autograd.grad(R_nablau_train, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )
                max_grad_nablau_t= torch.max(torch.abs(grad_nablau_t)).cpu().detach().numpy()
                max_grad_nablau_x= torch.max(torch.abs(grad_nablau_x)).cpu().detach().numpy()
                max_grad_nablau_y= torch.max(torch.abs(grad_nablau_y)).cpu().detach().numpy()
                u_C1 = max(u_C0, max_grad_nablau_t, max_grad_nablau_x, max_grad_nablau_y)

                #second derivatives
                grad_nablau_tt= torch.autograd.grad(grad_nablau_t, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_nablau_tx= torch.autograd.grad(grad_nablau_t, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_nablau_ty= torch.autograd.grad(grad_nablau_t, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )
                grad_nablau_xt= torch.autograd.grad(grad_nablau_x, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_nablau_xx= torch.autograd.grad(grad_nablau_x, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_nablau_xy= torch.autograd.grad(grad_nablau_x, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )
                grad_nablau_yt= torch.autograd.grad(grad_nablau_y, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,0].reshape(-1, )
                grad_nablau_yx= torch.autograd.grad(grad_nablau_y, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,1].reshape(-1, )
                grad_nablau_yy= torch.autograd.grad(grad_nablau_y, test_inp_0, grad_outputs=inputs_0, create_graph=True)[0][:,2].reshape(-1, )

                max_grad_nablau_tt= torch.max(torch.abs(grad_nablau_tt)).cpu().detach().numpy()
                max_grad_nablau_tx= torch.max(torch.abs(grad_nablau_tx)).cpu().detach().numpy()
                max_grad_nablau_ty= torch.max(torch.abs(grad_nablau_ty)).cpu().detach().numpy()
                max_grad_nablau_xt= torch.max(torch.abs(grad_nablau_xt)).cpu().detach().numpy()
                max_grad_nablau_xx= torch.max(torch.abs(grad_nablau_xx)).cpu().detach().numpy()
                max_grad_nablau_xy= torch.max(torch.abs(grad_nablau_xy)).cpu().detach().numpy()
                max_grad_nablau_yt= torch.max(torch.abs(grad_nablau_yt)).cpu().detach().numpy()
                max_grad_nablau_yx= torch.max(torch.abs(grad_nablau_yx)).cpu().detach().numpy()
                max_grad_nablau_yy= torch.max(torch.abs(grad_nablau_yy)).cpu().detach().numpy()
                R_nablau = max(u_C1, max_grad_nablau_tt, max_grad_nablau_tx, max_grad_nablau_ty, max_grad_nablau_xt, max_grad_nablau_xx, max_grad_nablau_xy, max_grad_nablau_yt, max_grad_nablau_yx, max_grad_nablau_yy)
                models["R_nablau"]=R_nablau

                
                #|nabla u_theta - u|_C^0
                u_C1=torch.sqrt((u_pred_var_list1 - u_train_var_list1) ** 2 + (u_pred_var_list2 - u_train_var_list2)**2)
                u_C1=torch.max(u_C1).cpu().detach().numpy() 
                models["u_C1"] = u_C1


                
                #recalculating the training and collocation points to calculate the inidividual terms
                #that make up the training error (3.17)
                n_coll = int(models["Nf_train"])
                n_u = int(models["Nu_train"])
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
                
                #training error on the PDE
                res_train = compute_residual(trained_model, x_coll)
                if torch.cuda.is_available():
                    res_train = res_train.cuda()
                res_loss = round(float(torch.sqrt(torch.mean(res_train ** 2))), 6)
                models["res_loss"] = res_loss

                #training error on the boundary for u
                u_pred_var_list = list()
                if torch.cuda.is_available():
                    x_b=x_b.cuda()
                u_pred_var_list.append(trained_model(x_b))
                u_pred_var_list = torch.cat(u_pred_var_list, 0)
                if torch.cuda.is_available():
                    u_pred_var_list=u_pred_var_list.cuda()
                loss_su=round(float(torch.sqrt(torch.mean(u_pred_var_list ** 2))), 6)
                models["loss_su"] = loss_su

                #training error on the boundary for u_t
                u_pred_var_list = list()
                x_b.requires_grad = True
                u_b=trained_model(x_b).reshape(-1, )
                inputs = torch.ones(x_b.shape[0], )
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                grad_u_b = torch.autograd.grad(u_b, x_b, grad_outputs=inputs, create_graph=True)[0]
                grad_u_b_t = grad_u_b[:, 0].reshape(-1, )
                u_pred_var_list.append(grad_u_b_t)
                u_pred_var_list = torch.cat(u_pred_var_list, 0)
                if torch.cuda.is_available():
                    u_pred_var_list=u_pred_var_list.cuda()
                loss_sut=round(float(torch.sqrt(torch.mean(u_pred_var_list ** 2))), 6)
                models["loss_sut"] = loss_sut

                #training error for initial data u_0 and u
                u_pred_var_list = list()
                u_train_var_list = list()
                if torch.cuda.is_available():
                    x_time_0=x_time_0.cuda()
                u_pred_var_list.append(trained_model(x_time_0))
                u_train_var_list.append(u0(x_time_0))
                u_pred_var_list = torch.cat(u_pred_var_list, 0)
                u_train_var_list = torch.cat(u_train_var_list, 0)
                if torch.cuda.is_available():
                    u_pred_var_list=u_pred_var_list.cuda()
                    u_train_var_list=u_train_var_list.cuda()
                loss_u0=round(float(torch.sqrt(torch.mean((u_pred_var_list - u_train_var_list) ** 2))), 6)
                models["loss_u0"] = loss_u0

                #training error for initial data u_1 and u
                u_pred_var_list = list()
                u_train_var_list = list()
                x_time_0.requires_grad = True
                u_u=trained_model(x_time_0).reshape(-1, )
                inputs = torch.ones(x_time_0.shape[0], )
                if torch.cuda.is_available():
                    inputs=inputs.cuda()
                grad_u_u = torch.autograd.grad(u_u, x_time_0, grad_outputs=inputs, create_graph=True)[0]
                grad_u_u_t = grad_u_u[:, 0].reshape(-1, )
                u_pred_var_list.append(grad_u_u_t)
                u_train_var_list.append(u1(x_time_0))
                u_pred_var_list = torch.cat(u_pred_var_list, 0)
                u_train_var_list = torch.cat(u_train_var_list, 0)
                if torch.cuda.is_available():
                    u_pred_var_list=u_pred_var_list.cuda()
                    u_train_var_list=u_train_var_list.cuda()
                loss_u1=round(float(torch.sqrt(torch.mean((u_pred_var_list - u_train_var_list) ** 2))), 6)
                models["loss_u1"] = loss_u1

                #training error for nabla u_0 and nabla u on the initial data
                u_pred_var_list1 = list()
                u_train_var_list1 = list()
                u_pred_var_list2 = list()
                u_train_var_list2 = list()
                grad_u_u_x = grad_u_u[:, 1].reshape(-1, )
                grad_u_u_y = grad_u_u[:, 2].reshape(-1, )
                u_pred_var_list1.append(grad_u_u_x)
                u_pred_var_list2.append(grad_u_u_y)
                u_train_var_list1.append(dx_u0(x_time_0))
                u_train_var_list2.append(dy_u0(x_time_0))
                u_pred_var_list1 = torch.cat(u_pred_var_list1, 0)
                u_train_var_list1 = torch.cat(u_train_var_list1, 0)
                u_pred_var_list2 = torch.cat(u_pred_var_list2, 0)
                u_train_var_list2 = torch.cat(u_train_var_list2, 0)
                if torch.cuda.is_available():
                    u_pred_var_list1=u_pred_var_list.cuda()
                    u_train_var_list1=u_train_var_list.cuda()
                    u_pred_var_list2=u_pred_var_list.cuda()
                    u_train_var_list2=u_train_var_list.cuda()
                loss_nablau=round(float(torch.sqrt(torch.mean(abs(u_pred_var_list1 - u_train_var_list1) ** 2 + abs(u_pred_var_list2 - u_train_var_list2) ** 2))), 6)
                models["loss_nablau"] = loss_nablau
                



                ########################

            models_list.append(models)
            # print(models)

        else:
            print("No File Found")

    retraining_prop = pd.concat(models_list, ignore_index=True)
    return retraining_prop


#compute the PDE residual
def compute_residual(network, x_f_train):
    if torch.cuda.is_available():
        x_f_train=x_f_train.cuda()
    x_f_train.requires_grad = True
    u = (network(x_f_train)).reshape(-1, )

    inputs = torch.ones(x_f_train.shape[0], )

    if torch.cuda.is_available():
        inputs = inputs.cuda()

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=inputs, create_graph=True)[0]
    grad_u_t = grad_u[:, 0].reshape(-1, )
    grad_u_x = grad_u[:, 1].reshape(-1, )
    grad_u_y = grad_u[:, 2].reshape(-1, )

    grad_u_tt = torch.autograd.grad(grad_u_t, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 0]
    grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 1]
    grad_u_yy = torch.autograd.grad(grad_u_y, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 2]

    residual = grad_u_tt - grad_u_xx - grad_u_yy + a*grad_u_t

    return residual

#compute points from midpoint rule
def generator_points(samples, dim):
    h=round(samples ** (1/dim))
    midpoint_1D=np.arange(1/(2*h),1,(1/h))
    midpoint_nD=list(it.product(midpoint_1D, repeat=dim))
    return torch.FloatTensor(midpoint_nD)


