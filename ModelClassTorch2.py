from ImportFile import *

#sets up PINN, training process and loss function

pi = math.pi


class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        # pre-activation
        return x * torch.sigmoid(x)


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['swish']:
        return Swish()
    else:
        raise ValueError('Unknown activation function')


class Pinns(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_properties, additional_models=None,
                 solid_object=None):
        super(Pinns, self).__init__()
        #set up the Neural network
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(network_properties["hidden_layers"])
        self.neurons = int(network_properties["neurons"])
        self.lambda_residual = float(network_properties["residual_parameter"])
        self.kernel_regularizer = int(network_properties["kernel_regularizer"])
        self.regularization_param = float(network_properties["regularization_parameter"])
        self.num_epochs = int(network_properties["epochs"])
        self.act_string = str(network_properties["activation"])

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.batch_input = nn.BatchNorm1d(num_features=self.neurons, momentum=0.05)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.batch_layers = nn.ModuleList([nn.BatchNorm1d(num_features=self.neurons, momentum=0.05) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.solid_object = solid_object
        self.additional_models = additional_models

        self.activation = activation(self.act_string)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for l in self.hidden_layers:
            x = self.activation(l(x))
        return self.output_layer(x)


#fit function for training the neural network with a custom loss
def fit(model, optimizer_ADAM, optimizer_LBFGS, training_set_class, validation_set_clsss=None, verbose=False, training_ic=False):
    num_epochs = model.num_epochs

    train_losses = list([np.NAN])
    val_losses = list()
    freq = 50

    training_coll = training_set_class.data_coll
    training_boundary = training_set_class.data_boundary
    training_initial_internal = training_set_class.data_initial_internal
    epoch_LSBGF = num_epochs

    model.train()
    for epoch in range(num_epochs): 
        if epoch < epoch_LSBGF:
            print("Using LSBGF")
            optimizer = optimizer_LBFGS
        else:
            print("Using Adam")
            optimizer = optimizer_ADAM
        if verbose and epoch % freq == 0:
            print("################################ ", epoch, " ################################")

        #training procedue if  we have initial training points and boundary training points
        #which is the case for the semilinear wave equation
        if len(training_boundary) != 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_), (x_u_train_, u_train_)) in enumerate(
                    zip(training_coll, training_boundary, training_initial_internal)):
                if verbose and epoch % freq == 0:
                    print("Batch Number:", step)

                x_ob = None
                u_ob = None

                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    x_u_train_ = x_u_train_.cuda()
                    u_train_ = u_train_.cuda()

                #training with VeryCustomLoss which is the loss function for the semilinear wave equation (3.17)
                def closure():
                    optimizer.zero_grad()
                    loss_f = VeryCustomLoss()(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, x_ob, u_ob,
                                          training_set_class, training_ic)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()
        
        #alternative training ig there are no boundary training points
        elif len(training_boundary) == 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_u_train_, u_train_)) in enumerate(zip(training_coll, training_initial_internal)):

                x_ob = None
                u_ob = None

                x_b_train_ = torch.full((4, 1), 0)
                u_b_train_ = torch.full((4, 1), 0)

                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    x_u_train_ = x_u_train_.cuda()
                    u_train_ = u_train_.cuda()

                def closure():
                    optimizer.zero_grad()
                    loss_f = VeryCustomLoss()(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, x_ob, u_ob,
                                          training_set_class, training_ic)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    # print(train_losses[0])
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

        #alternative training if there are no initial training points
        elif len(training_boundary) != 0 and len(training_initial_internal) == 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_boundary)):

                x_ob = None
                u_ob = None

                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    x_u_train_ = x_u_train_.cuda()
                    u_train_ = u_train_.cuda()

                def closure():
                    optimizer.zero_grad()
                    loss_f = VeryCustomLoss()(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, x_ob, u_ob,
                                          training_set_class, training_ic)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    # print(train_losses[0])
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

        if validation_set_clsss is not None:

            N_coll_val = validation_set_clsss.n_collocation
            N_b_val = validation_set_clsss.n_boundary
            validation_set = validation_set_clsss.data_no_batches

            for x_val, y_val in validation_set:
                model.eval()

                x_coll_val = x_val[:N_coll_val, :]
                x_b_val = x_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val, :]
                u_b_val = y_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val]
                x_u_val = x_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val:, :]
                u_val = y_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val:, :]

                if torch.cuda.is_available():
                    x_coll_val = x_coll_val.cuda()
                    x_b_val = x_b_val.cuda()
                    u_b_val = u_b_val.cuda()
                    x_u_val = x_u_val.cuda()
                    u_val = u_val.cuda()

                loss_val = VeryCustomLoss()(model, x_u_val, u_val, x_b_val, u_b_val, x_coll_val, validation_set_clsss)

                if torch.cuda.is_available():
                    del x_coll_val
                    del x_b_val
                    del u_b_val
                    del x_u_val
                    del u_val
                    torch.cuda.empty_cache()

                    # val_losses.append(loss_val)
                if verbose and epoch % 100 == 0:
                    print("Validation Loss:", loss_val)

    history = [train_losses, val_losses] if validation_set_clsss is not None else [train_losses]

    return train_losses[0]


class VeryCustomLoss(torch.nn.Module):
    """
    The loss for the semi-linear wave equation with the specific boundary conditions
    """
    def __init__(self):
        super(VeryCustomLoss, self).__init__()

    def forward(self, network, x_u_train, u_train, x_b_train, u_b_train, x_f_train, x_obj, u_obj, dataclass,
                training_ic, computing_error=False):
        space_dimensions = dataclass.space_dimensions
        solid_object = dataclass.obj

        # Residual #1 is the loss for solving the PDE
        res = Ec.compute_res(network, x_f_train, space_dimensions, solid_object, computing_error)
        if not computing_error and torch.cuda.is_available():
            res=res.cuda()
        loss_1 = (torch.mean(abs(res) ** 2))

        # Residual #2 is boundary for u
        u_pred_var_list = list()
        u_pred_var_list.append(network(x_b_train))
        u_pred_var_list = torch.cat(u_pred_var_list, 0)
        if not computing_error and torch.cuda.is_available():
            u_pred_var_list = u_pred_var_list.cuda()
        loss_2=(torch.mean(abs(u_pred_var_list) ** 2))


        # Residual #3 is boundary for u_t
        u_pred_var_list = list()
        x_b_train.requires_grad = True
        u_b=network(x_b_train).reshape(-1, )
        inputs = torch.ones(x_b_train.shape[0], )
        if not computing_error and torch.cuda.is_available():
            inputs = inputs.cuda()
        grad_u_b = torch.autograd.grad(u_b, x_b_train, grad_outputs=inputs, create_graph=True)[0]
        grad_u_b_t = grad_u_b[:, 0].reshape(-1, )
        u_pred_var_list.append(grad_u_b_t)
        u_pred_var_list = torch.cat(u_pred_var_list, 0)
        if not computing_error and False:
            u_pred_var_list = u_pred_var_list.cuda()
        loss_3=(torch.mean(abs(u_pred_var_list) ** 2))
            

        # Residual #4 is initial data for u
        u_pred_var_list = list()
        u_train_var_list = list()
        u_pred_var_list.append(network(x_u_train))
        u_train_var_list.append(Ec.u0(x_u_train))
        u_pred_var_list = torch.cat(u_pred_var_list, 0)
        u_train_var_list = torch.cat(u_train_var_list, 0)
        if not computing_error and torch.cuda.is_available():
            u_pred_var_list = u_pred_var_list.cuda()
            u_train_var_list = u_train_var_list.cuda()
        loss_4=(torch.mean(abs(u_pred_var_list - u_train_var_list) ** 2))

        # Residual #5 is initial data for u_t
        u_pred_var_list = list()
        u_train_var_list = list()
        x_u_train.requires_grad = True
        u_u=network(x_u_train).reshape(-1, )
        inputs = torch.ones(x_u_train.shape[0], )
        if not computing_error and torch.cuda.is_available():
            inputs = inputs.cuda()
        grad_u_u = torch.autograd.grad(u_u, x_u_train, grad_outputs=inputs, create_graph=True)[0]
        grad_u_u_t = grad_u_u[:, 0].reshape(-1, )
        u_pred_var_list.append(grad_u_u_t)
        u_train_var_list.append(Ec.u1(x_u_train))
        u_pred_var_list = torch.cat(u_pred_var_list, 0)
        u_train_var_list = torch.cat(u_train_var_list, 0)
        if not computing_error and torch.cuda.is_available():
            u_pred_var_list = u_pred_var_list.cuda()
            u_train_var_list = u_train_var_list.cuda()
        loss_5=(torch.mean(abs(u_pred_var_list - u_train_var_list) ** 2))

        # Residual #6 is initial data for nabla u
        u_pred_var_list1 = list()
        u_train_var_list1 = list()
        u_pred_var_list2 = list()
        u_train_var_list2 = list()
        grad_u_u_x = grad_u_u[:, 1].reshape(-1, )
        grad_u_u_y = grad_u_u[:, 2].reshape(-1, )
        u_pred_var_list1.append(grad_u_u_x)
        u_pred_var_list2.append(grad_u_u_y)
        u_train_var_list1.append(Ec.dx_u0(x_u_train))
        u_train_var_list2.append(Ec.dy_u0(x_u_train))
        u_pred_var_list1 = torch.cat(u_pred_var_list1, 0)
        u_train_var_list1 = torch.cat(u_train_var_list1, 0)
        u_pred_var_list2 = torch.cat(u_pred_var_list2, 0)
        u_train_var_list2 = torch.cat(u_train_var_list2, 0)
        if not computing_error and torch.cuda.is_available():
            u_pred_var_list1 = u_pred_var_list.cuda()
            u_train_var_list1 = u_train_var_list.cuda()
            u_pred_var_list2 = u_pred_var_list.cuda()
            u_train_var_list2 = u_train_var_list.cuda()
        loss_6=(torch.mean(abs(u_pred_var_list1 - u_train_var_list1) ** 2 + abs(u_pred_var_list2 - u_train_var_list2) ** 2))
           
        return loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6

#original Custom loss (nothing was changed here)
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()


    def forward(self, network, x_u_train, u_train, x_b_train, u_b_train, x_f_train, x_obj, u_obj, dataclass,
                training_ic, computing_error=False):

        lambda_residual = network.lambda_residual
        lambda_reg = network.regularization_param
        order_regularizer = network.kernel_regularizer
        space_dimensions = dataclass.space_dimensions
        BC = dataclass.BC
        solid_object = dataclass.obj

        # print(x_b_train.shape)

        if x_b_train.shape[0] <= 1:
            space_dimensions = 0

        u_pred_var_list = list()
        u_train_var_list = list()
        for j in range(dataclass.output_dimension):
            #print(dataclass.output_dimension)
            # Space dimensions
            if not training_ic and Ec.extrema_values is not None:
                for i in range(space_dimensions):
                    half_len_x_b_train_i = int(x_b_train.shape[0] / (2 * space_dimensions))

                    x_b_train_i = x_b_train[i * int(x_b_train.shape[0] / space_dimensions):(i + 1) * int(
                        x_b_train.shape[0] / space_dimensions), :]
                    u_b_train_i = u_b_train[i * int(x_b_train.shape[0] / space_dimensions):(i + 1) * int(
                        x_b_train.shape[0] / space_dimensions), :]
                    boundary = 0
                    while boundary < 2:

                        x_b_train_i_half = x_b_train_i[
                                           half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]
                        u_b_train_i_half = u_b_train_i[
                                           half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]
                        #We have functional boundary
                        if BC[i][boundary][j] == "func":

                            u_pred_var_list.append(network(x_b_train_i_half)[:, j])
                            u_train_var_list.append(u_b_train_i_half[:, j])
                        if BC[i][boundary][j] == "der":
                            x_b_train_i_half.requires_grad = True
                            f_val = network(x_b_train_i_half)[:, j]
                            inputs = torch.ones(x_b_train_i_half.shape[0], )
                            if not computing_error and False:#and torch.cuda.is_available():
                                inputs = inputs.cuda()
                            der_f_vals = \
                                torch.autograd.grad(f_val, x_b_train_i_half, grad_outputs=inputs, create_graph=True)[0][:, i]
                            u_pred_var_list.append(der_f_vals)
                            u_train_var_list.append(u_b_train_i_half[:, j])
                        elif BC[i][boundary][j] == "periodic":
                            x_half_1 = x_b_train_i_half
                            x_half_2 = x_b_train_i[
                                       half_len_x_b_train_i * (boundary + 1):half_len_x_b_train_i * (boundary + 2), :]
                            x_half_1.requires_grad = True
                            x_half_2.requires_grad = True
                            inputs = torch.ones(x_half_1.shape[0], )
                            if not computing_error and False:#and torch.cuda.is_available():
                                inputs = inputs.cuda()
                            pred_first_half = network(x_half_1)[:, j]
                            pred_second_half = network(x_half_2)[:, j]
                            der_pred_first_half = \
                                torch.autograd.grad(pred_first_half, x_half_1, grad_outputs=inputs, create_graph=True)[
                                    0]
                            der_pred_first_half_i = der_pred_first_half[:, i]
                            der_pred_second_half = \
                                torch.autograd.grad(pred_second_half, x_half_2, grad_outputs=inputs, create_graph=True)[
                                    0]
                            der_pred_second_half_i = der_pred_second_half[:, i]

                            u_pred_var_list.append(pred_second_half)
                            u_train_var_list.append(pred_first_half)

                            u_pred_var_list.append(der_pred_second_half_i)
                            u_train_var_list.append(der_pred_first_half_i)

                            boundary = boundary + 1

                        boundary = boundary + 1
            else:
                u_pred_b, u_train_b = Ec.apply_BC(x_b_train, u_b_train, network)
                u_pred_var_list.append(u_pred_b)
                u_train_var_list.append(u_train_b)
            if x_u_train.shape[0] != 0 and j != 2:
                u_pred_var_list.append(network(x_b_train)[:, j])
                u_train_var_list.append(u_b_train[:, j])
            # Time Dimension
            if x_u_train.shape[0] != 0 and j != 2:
                u_pred_var_list.append(network(x_u_train)[:, j])
                u_train_var_list.append(u_train[:, j])

            if x_obj is not None and not training_ic:
                if BC[-1][j] == "func":
                    u_pred_var_list.append(network(x_obj)[:, j])
                    u_train_var_list.append(u_obj[:, j])

                if BC[-1][j] == "der":
                    x_obj_grad = x_obj.clone()
                    x_obj_transl = x_obj_grad[(np.arange(0, x_obj_grad.shape[0]) + 1) % (x_obj_grad.shape[0]), :]
                    x_obj_mean = (x_obj_grad + x_obj_transl) / 2
                    x_obj_mean.requires_grad = True
                    f_val = network(x_obj_mean)[:, j]
                    inputs = torch.ones(x_obj_mean.shape[0], )
                    if not computing_error and False:#and torch.cuda.is_available():
                        inputs = inputs.cuda()
                    der_f_vals_x = torch.autograd.grad(f_val, x_obj_mean, grad_outputs=inputs, create_graph=True)[0][:,
                                   0]
                    der_f_vals_y = torch.autograd.grad(f_val, x_obj_mean, grad_outputs=inputs, create_graph=True)[0][:,
                                   1]
                    t = (x_obj_grad - x_obj_transl)

                    nx = t[:, 1] / torch.sqrt(t[:, 1] ** 2 + t[:, 0] ** 2)
                    ny = -t[:, 0] / torch.sqrt(t[:, 1] ** 2 + t[:, 0] ** 2)
                    der_n = der_f_vals_x * nx + der_f_vals_y * ny
                    u_pred_var_list.append(der_n)
                    u_train_var_list.append(u_obj[:, j])

        u_pred_tot_vars = torch.cat(u_pred_var_list, 0)
        u_train_tot_vars = torch.cat(u_train_var_list, 0)

        if not computing_error and False:#and torch.cuda.is_available():
            u_pred_tot_vars = u_pred_tot_vars.cuda()
            u_train_tot_vars = u_train_tot_vars.cuda()

        assert not torch.isnan(u_pred_tot_vars).any()

        loss_vars = (torch.mean(abs(u_pred_tot_vars - u_train_tot_vars) ** 2))

        if not training_ic:

            res = Ec.compute_res(network, x_f_train, space_dimensions, solid_object, computing_error)
            res_train = torch.tensor(()).new_full(size=(res.shape[0],), fill_value=0.0)

            if not computing_error and False:#and torch.cuda.is_available():
                res = res.cuda()
                res_train = res_train.cuda()

            loss_res = (torch.mean(abs(res) ** 2))

            u_pred_var_list.append(res)
            u_train_var_list.append(res_train)

        loss_reg = regularization(network, order_regularizer)
        if not training_ic:
            loss_v =  loss_vars + lambda_residual * loss_res + lambda_reg * loss_reg  # + lambda_reg/loss_reg
        else:
            loss_v = torch.log10(loss_vars + lambda_reg * loss_reg)
        #print("final loss:", loss_v.detach().cpu().numpy().round(4), " ", torch.log10(loss_vars).detach().cpu().numpy().round(4), " ",
        #      torch.log10(loss_res).detach().cpu().numpy().round(4))
        return loss_v

#supplement to CustomLoss
def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss

def init_xavier(model):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            gain = nn.init.calculate_gain('tanh')
            # gain = 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.0)

    model.apply(init_weights)
