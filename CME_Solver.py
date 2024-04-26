import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_saving as dts

from torch.optim.lr_scheduler import MultiStepLR


class CMESolver(object):
    """The fully connected neural network model."""

    def __init__(self, network, config_data):
        super(CMESolver, self).__init__()
        self.network = network
        self.config_data = config_data

        # create data for training and validation..unless it is already available
        if config_data['net_config']['training_samples_needed'] == "True":
            start_time = time.time()
            self.training_data = self.network.generate_sampled_rtc_trajectories(
                self.config_data['reaction_network_config']['final_time'],
                self.config_data['reaction_network_config']['num_time_interval'],
                self.config_data['net_config']['batch_size'])
            self.training_data_cpu_time = (time.time() - start_time)
            print("Time needed to generate training trajectories: %3u" % self.training_data_cpu_time)
            dts.save_sampled_trajectories(config_data['reaction_network_config']['output_folder'] + "/",
                                          self.training_data, sample_type="training")
            dts.save_cpu_time(config_data['reaction_network_config']['output_folder'] + "/", self.training_data_cpu_time
                              , training=True)
        else:
            self.training_data = dts.load_save_sampled_trajectories(
                config_data['reaction_network_config']['output_folder']
                + "/", sample_type="training")
            self.training_data_cpu_time = dts.load_cpu_time(config_data['reaction_network_config']['output_folder']
                                                            + "/", training=True)
        if config_data['net_config']['validation_samples_needed'] == "True":
            start_time = time.time()
            self.valid_data = self.network.generate_sampled_rtc_trajectories(
                self.config_data['reaction_network_config']['final_time'],
                self.config_data['reaction_network_config']['num_time_interval'],
                self.config_data['net_config']['valid_size'])
            self.validation_data_cpu_time = (time.time() - start_time)
            self.total_num_simulated_trajectories = self.config_data['net_config']['valid_size'] + \
                                                    self.config_data['net_config']['batch_size']
            print("Time needed to generate validation trajectories: %3u" % (time.time() - start_time))
            dts.save_sampled_trajectories(config_data['reaction_network_config']['output_folder'] + "/",
                                          self.training_data,
                                          sample_type="validation")
            dts.save_cpu_time(config_data['reaction_network_config']['output_folder'] + "/",
                              self.validation_data_cpu_time, training=False)
        else:
            self.valid_data = dts.load_save_sampled_trajectories(config_data['reaction_network_config']['output_folder']
                                                                 + "/", sample_type="validation")
            self.validation_data_cpu_time = dts.load_cpu_time(config_data['reaction_network_config']['output_folder']
                                                            + "/", training=False)

        # set initial values for functions
        times, states_trajectories, martingale_trajectories = self.training_data
        yvals = torch.from_numpy(self.network.output_function(states_trajectories[:, -1, :]))  # convert to tensor
        y0 = torch.mean(yvals, dim=0)
        # set func_clipping_thresholds
        self.delta_clip = torch.ones_like(y0) + torch.mean(yvals, dim=0) + 2 * torch.std(yvals, dim=0).detach().numpy()
        #self.delta_clip = np.ones(shape=[self.network.output_function_size], dtype="float64") + torch.mean(yvals, dim=0) + 2 * torch.std(yvals, dim=0)
        self.model = NonsharedModel(network, config_data, y0, self.delta_clip)
        if config_data['net_config']['use_previous_training_weights'] == "True":
            filename = config_data['reaction_network_config']['output_folder'] + "/" + "trained_weights"
            self.model.load_state_dict(torch.load(filename))
        self.y_init = self.model.y_init

        # set up optimizer and learning rate scheduler
        lr_boundaries = config_data['net_config']['lr_boundaries']
        lr_values = config_data['net_config']['lr_values']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_values[0], eps=1e-8)
        self.scheduler = MultiStepLR(self.optimizer, milestones=lr_boundaries, gamma=lr_values[1] / lr_values[0])
        self.total_num_simulated_trajectories = self.config_data['net_config']['valid_size'] + self.config_data['net_config']['batch_size']

    def train(self):
        start_time = time.time()
        training_history = []
        function_value_data = []
        num_iterations = self.config_data['net_config']['num_iterations']
        # num_batch_size = self.config_data['net_config']['batch_size']
        logging_frequency = self.config_data['net_config']['logging_frequency']
        # training_data_reset_frequency = self.config_data['net_config']['training_data_reset_frequency']

        # begin sgd iteration
        for step in range(1, num_iterations + 1):
            self.train_step(self.training_data)
            if step % logging_frequency == 0:
                loss = self.loss_fn(self.valid_data, training=False).detach().numpy()
                y_init = self.y_init.detach().numpy()
                elapsed_time = time.time() - start_time + self.validation_data_cpu_time + self.training_data_cpu_time
                training_history.append([step, loss, elapsed_time])
                function_value_data.append(y_init)
                print("step: %5u, loss: %.4e, elapsed time: %3u" % (
                    step, loss, elapsed_time))
                print_array_nicely(y_init, "Estimated Value")
            # if self.config_data['net_config']['allow_batch_reset'] == "True" and \
            #         step % training_data_reset_frequency == 0:
            #     self.training_data = self.network.generate_sampled_rtc_trajectories(
            #         self.config_data['reaction_network_config']['final_time'],
            #         self.config_data['reaction_network_config']['num_time_interval'],
            #         num_batch_size)
            #     print('New training data generated!')
            #     self.total_num_simulated_trajectories += num_batch_size
        return np.array(training_history), np.array(function_value_data), self.total_num_simulated_trajectories

    def loss_fn(self, inputs, training):
        times, states_trajectories, martingale_trajectories = inputs
        y_terminal = self.model(inputs, training)
        y_comp = torch.from_numpy(self.network.output_function(states_trajectories[:, -1, :]))  # convert to tensor
        delta = (y_terminal - y_comp) / self.delta_clip
        loss = torch.mean(torch.where(torch.abs(delta) < 1, torch.square(delta), 2 * torch.abs(delta) - 1), dim=0)
        return torch.sum(loss)

    # def grad(self, inputs, training):
    #     with tf.GradientTape(persistent=True) as tape:
    #         loss = self.loss_fn(inputs, training)
    #     grad = tape.gradient(loss, self.model.trainable_variables)
    #     del tape
    #     return grad

    def train_step(self, train_data):
        self.optimizer.zero_grad()
        loss = self.loss_fn(train_data, training=True)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def estimate_parameter_sensitivities(self):
        times, states_trajectories, martingale_trajectories = self.training_data
        return self.model.compute_parameter_jacobian(states_trajectories, times, len(self.network.parameter_dict),
                                                     training=False)


class NonsharedModel(nn.Module):

    def __init__(self, network, config_data, y0, delta_clip):
        super(NonsharedModel, self).__init__()
        self.network = network
        self.delta_clip = delta_clip
        self.stop_time = config_data['reaction_network_config']['final_time']
        self.num_exponential_features = config_data['net_config']['num_exponential_features']
        self.num_temporal_dnns = config_data['net_config']['num_temporal_dnns']
        self.num_time_samples = config_data['reaction_network_config']['num_time_interval']
        self.y_init = nn.Parameter(y0)
        self.eigval_real = nn.Parameter(torch.rand(1, self.num_exponential_features, dtype=torch.float64))
        self.eigval_imag = nn.Parameter(torch.zeros(1, self.num_exponential_features, dtype=torch.float64))
        self.eigval_phase = nn.Parameter(torch.zeros(1, self.num_exponential_features, dtype=torch.float64))
        self.subnet = [FeedForwardSubNet(self.network.num_reactions, self.network.output_function_size, config_data)
                       for _ in range(self.num_temporal_dnns)]

    def forward(self, inputs, training):
        times, states_trajectories, martingale_trajectories = inputs
        states_trajectories = torch.from_numpy(states_trajectories)
        martingale_trajectories = torch.from_numpy(martingale_trajectories)
        batch_size = martingale_trajectories.shape[0]
        all_one_vec = torch.ones(batch_size, 1, dtype=torch.float64)
        y = torch.matmul(all_one_vec, self.y_init.unsqueeze(0))
        for t in range(0, self.num_time_samples - 1):
            time_left = self.stop_time - times[t]
            temporal_dnn = int(t * self.num_temporal_dnns / self.num_time_samples)
            features_real = torch.tile(torch.exp(self.eigval_real * time_left), [batch_size, 1])
            features_imag = torch.sin(self.eigval_imag * time_left + self.eigval_phase).repeat(batch_size, 1)
            inputs = torch.stack(torch.unbind(states_trajectories[:, t, :], dim=-1) + torch.unbind(features_real, dim=-1) + torch.unbind(features_imag, dim=-1), dim=1)
            z = self.subnet[temporal_dnn](inputs, training)
            z = z.view(batch_size, self.network.output_function_size, self.network.num_reactions)
            martingale_increment = martingale_trajectories[:, t + 1, :] - martingale_trajectories[:, t, :]
            martingale_increment = martingale_increment.unsqueeze(1)
            y = y + torch.sum(z * martingale_increment, dim=2)

        return y

    def compute_parameter_jacobian(self, states_trajectories, times, num_params, training):
        states_trajectories = torch.from_numpy(states_trajectories)
        batch_size = states_trajectories.shape[0]
        jacobian = torch.zeros(batch_size, num_params, self.network.output_function_size)
        for t in range(0, self.num_time_samples - 1):
            time_left = self.stop_time - times[t]
            temporal_dnn = int(t * self.num_temporal_dnns / self.num_time_samples)
            features_real = torch.tile(torch.exp(self.eigval_real * time_left), [batch_size, 1])
            features_imag = torch.sin(self.eigval_imag * time_left + self.eigval_phase).repeat(batch_size, 1)
            inputs = torch.stack(torch.unbind(states_trajectories[:, t, :], dim=-1) + torch.unbind(features_real, dim=-1) + torch.unbind(features_imag, dim=-1), dim=1)
            z = self.subnet[temporal_dnn](inputs, training)
            z = z.view(batch_size, self.network.output_function_size, self.network.num_reactions)
            propensity_jacobian = torch.stack([torch.from_numpy(self.network.propensity_sensitivity_matrix(states_trajectories[i, t, :]))
                                               for i in range(states_trajectories[:, t, :].size(0))], dim=0)
            jacobian = jacobian + torch.matmul(propensity_jacobian, z.transpose(1, 2)) * (times[t + 1] - times[t])
        return torch.mean(jacobian, dim=0)


class FeedForwardSubNet(nn.Module):
    def __init__(self, num_reactions, output_function_size, config_data):
        super(FeedForwardSubNet, self).__init__()
        num_hiddens = config_data['net_config']['num_nodes_per_layer']  # 4
        num_layers = config_data['net_config']['num_hidden_layers']  # 2
        num_species = config_data['reaction_network_config']['num_species']  # 10
        num_exp_features = config_data['net_config']['num_exponential_features']  # 1

        # Define the hidden layers and add them to a ModuleList
        self.layers = nn.ModuleList([nn.Linear(num_species + 2*num_exp_features, num_hiddens)])  # put first layer into module list
        self.layers.extend([nn.Linear(num_hiddens, num_hiddens) for _ in range(num_layers - 1)])  # add the hidden layers
        self.final_layer = nn.Linear(num_hiddens, num_reactions * output_function_size)

        # Initialize weights and biases
        #for layer in self.layers:
        #    nn.init.zeros_(layer.weight)
        #    nn.init.zeros_(layer.bias)
        #nn.init.zeros_(self.final_layer.weight)
        #nn.init.zeros_(self.final_layer.bias)

    def forward(self, x, training):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.final_layer(x)
        return x


def print_array_nicely(y, name):
    size_input = y.size
    y = y.reshape(size_input, )
    print(name, ":", end=' (')
    for i in range(y.size - 1):
        print("%.3f" % y[i], end=', ')
    print("%.3f" % y[y.size - 1], end=')\n')
