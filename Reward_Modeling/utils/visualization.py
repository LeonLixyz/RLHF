import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

def plot_training_log(metric, reward, config):

    if config['project'] == 'reward-enn':

        plot_dir = config['plot_dir'].format(
            config['user'],
            config['project'],
            config['dataset_name'],
            config['lr'],
            config['ref_size'],
            config['hidden_size'],
            config['num_epochs']
            )    
        
    elif config['project'] == 'vanilla-reward':

        plot_dir = config['plot_dir'].format(
            config['user'],
            config['project'],
            config['dataset_name'],
            config['lr'],
            config['num_epochs']
            )  

    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    metrics = [
        ('Training loss', metric['training_loss'], 'Loss', False),
        ('Evaluation accuracy', metric['accuracy'], 'Accuracy', True),
        ('Learning rate', metric['learning_rate'], 'Learning rate', False),
        ('Evaluation Loss', metric['eval_loss'], 'Loss', True),
    ]

    # Plot metrics
    for title, data, ylabel, scale in metrics:
        if scale:
            x_values = [i * config['eval_steps'] for i in range(len(data))]
        else:
            x_values = [i for i in range(len(data))]
        plt.plot(x_values, data)
        plt.title(title)
        plt.xlabel('Training steps')
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(plot_dir, f'{title.replace(" ", "_").lower()}.png'), dpi=300)
        plt.close()

    # Group rewards for combined plot
    grouped_rewards_combined = [
        ('r_win_average', reward['r_win_average'], 'Reward', True, 'blue', '-'),
        ('r_win_min', reward['r_win_min'], 'Reward', True, 'blue', '--'),
        ('r_win_max', reward['r_win_max'], 'Reward', True, 'blue', ':'),
        
        ('r_lose_average', reward['r_lose_average'], 'Reward', True, 'red', '-'),
        ('r_lose_min', reward['r_lose_min'], 'Reward', True, 'red', '--'),
        ('r_lose_max', reward['r_lose_max'], 'Reward', True, 'red', ':'),
        
        ('r_total_average', reward['r_total_average'], 'Reward', True, 'green', '-'),
        ('r_total_min', reward['r_total_min'], 'Reward', True, 'green', '--'),
        ('r_total_max', reward['r_total_max'], 'Reward', True, 'green', ':')
    ]

    # Plot combined rewards
    plt.figure(figsize=(10, 6))
    for title, data, ylabel, scale, color, linestyle in grouped_rewards_combined:
        if scale:
            x_values = [i * config['eval_steps'] for i in range(len(data))]
        else:
            x_values = [i for i in range(len(data))]
        plt.plot(x_values, data, color=color, linestyle=linestyle, label=title)
        plt.xlabel('Training steps')
        plt.ylabel(ylabel)

    plt.title("Combined Rewards (Average, Min, Max)")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(plot_dir, 'combined_rewards.png'), dpi=300)
    plt.close()

def plot_reward_diff(reward_diff_list, model_out_diff_list, label_list, config):

    if config['project'] == 'reward-enn':

        plot_dir = config['plot_dir'].format(
            config['user'],
            config['project'],
            config['dataset_name'],
            config['lr'],
            config['ref_size'],
            config['hidden_size'],
            config['num_epochs']
            )    
        
    elif config['project'] == 'vanilla-reward':

        plot_dir = config['plot_dir'].format(
            config['user'],
            config['project'],
            config['dataset_name'],
            config['lr'],
            config['num_epochs']
            )  
        
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    n = len(reward_diff_list)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot difference points without connecting them
    ax.plot(range(n), reward_diff_list, marker='o', linestyle='None', color='red', label='Reward Difference')
    ax.plot(range(n), model_out_diff_list, marker='o', linestyle='None', color='blue', label='Model Output Difference')

    # Draw a horizontal line at y=0
    ax.axhline(0, color='black', linestyle='--')

    # Set labels and title
    ax.set_xlabel('Preference')
    ax.set_ylabel('Difference')
    ax.set_title('reward enn diff vs model diff')

    # Set x-ticks to be the labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(label_list)

    # Show a legend
    ax.legend()

    # Save the plot
    plt.savefig(plot_dir + '/reward_diff.png', dpi=300)
    plt.close(fig)

    
def plot_components(reward_1_list, reward_2_list, model_out_1_list, model_out_2_list, Eta_out_1_list, Eta_out_2_list, P_out_1_list, P_out_2_list, label_list, config):
    
    if config['project'] == 'reward-enn':

        plot_dir = config['plot_dir'].format(
            config['user'],
            config['project'],
            config['dataset_name'],
            config['lr'],
            config['ref_size'],
            config['hidden_size'],
            config['num_epochs']
            )    
        
    elif config['project'] == 'vanilla-reward':

        plot_dir = config['plot_dir'].format(
            config['user'],
            config['project'],
            config['dataset_name'],
            config['lr'],
            config['num_epochs']
            )  
    
    n = len(reward_1_list)

    # Create an array for x-axis
    x = np.arange(n)

    # Define colors and markers
    color_dict = {'reward': 'red', 'model_out': 'green', 'Eta_out': 'blue', 'P_out': 'purple'}
    marker_dict = {1: 'o', 2: 's'}  # 'o' for circle, 's' for square

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot each line with markers and no lines
    ax.plot(x, reward_1_list, color=color_dict['reward'], marker=marker_dict[1], linestyle='None', label='reward_1')
    ax.plot(x, reward_2_list, color=color_dict['reward'], marker=marker_dict[2], linestyle='None', label='reward_2')
    ax.plot(x, model_out_1_list, color=color_dict['model_out'], marker=marker_dict[1], linestyle='None', label='model_out_1')
    ax.plot(x, model_out_2_list, color=color_dict['model_out'], marker=marker_dict[2], linestyle='None', label='model_out_2')
    ax.plot(x, Eta_out_1_list, color=color_dict['Eta_out'], marker=marker_dict[1], linestyle='None', label='Eta_out_1')
    ax.plot(x, Eta_out_2_list, color=color_dict['Eta_out'], marker=marker_dict[2], linestyle='None', label='Eta_out_2')
    ax.plot(x, P_out_1_list, color=color_dict['P_out'], marker=marker_dict[1], linestyle='None', label='P_out_1')
    ax.plot(x, P_out_2_list, color=color_dict['P_out'], marker=marker_dict[2], linestyle='None', label='P_out_2')

    # Show the legend at the right corner with small font size
    ax.legend(loc='upper right', fontsize='small')



    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Model Outout Components')


    plt.savefig(plot_dir + '/components.png', dpi=300)
    plt.close(fig)

def test_logger(reward_model, test_dataloader, device):

    reward_1_list = []
    reward_2_list = []
    model_out_1_list = []
    model_out_2_list = []
    Eta_out_1_list = []
    Eta_out_2_list = []
    P_out_1_list = []
    P_out_2_list = []
    reward_diff_list = []
    model_out_diff_list = []
    label_list = []

    with torch.no_grad():

        for i, batch in enumerate(test_dataloader):
            print('ith batch', i)
            

            data, z_samples = batch

            p_1, p_2, p_1_att, p_2_att, label = data
            p_1 = p_1.to(device)
            p_2 = p_2.to(device)
            p_1_att = p_1_att.to(device)
            p_2_att = p_2_att.to(device)
            label = label.to(device)
            z = z_samples.to(device)

            reward_1, model_out_1, Eta_out_1, P_out_1 = reward_model(p_1, p_1_att, z, log = True)
            reward_2, model_out_2, Eta_out_2, P_out_2 = reward_model(p_2, p_2_att, z, log = True)
            reward_1 = reward_1.item()
            reward_2 = reward_2.item()
            model_out_1 = model_out_1.item()
            model_out_2 = model_out_2.item()
            Eta_out_1 = Eta_out_1.item()
            Eta_out_2 = Eta_out_2.item()
            P_out_1 = P_out_1.item()
            P_out_2 = P_out_2.item()


            reward_diff = reward_1 - reward_2
            model_out_diff = model_out_1 - model_out_2

            reward_1_list.append(reward_1)
            reward_2_list.append(reward_2)
            model_out_1_list.append(model_out_1)
            model_out_2_list.append(model_out_2)
            Eta_out_1_list.append(Eta_out_1)
            Eta_out_2_list.append(Eta_out_2)
            P_out_1_list.append(P_out_1)
            P_out_2_list.append(P_out_2)
            reward_diff_list.append(reward_diff)
            model_out_diff_list.append(model_out_diff)
            label_list.append(label)

            print('preference is', label)
            print('reward_1 is', reward_1)
            print('reward_2 is', reward_2)
            print('model_out_1 is', model_out_1)
            print('model_out_2 is', model_out_2)
            print('Eta_out_1 is', Eta_out_1)
            print('Eta_out_2 is', Eta_out_2)
            print('P_out_1 is', P_out_1)
            print('P_out_2 is', P_out_2)
            print('reward_diff is', reward_diff)
            print('model_out_diff is', model_out_diff)

            print('-------------------------------------')

    return reward_1_list, reward_2_list, model_out_1_list, model_out_2_list, Eta_out_1_list, Eta_out_2_list, P_out_1_list, P_out_2_list, reward_diff_list, model_out_diff_list, label_list


def enn_test_logger(reward_model, test_dataloader, device):

    with torch.no_grad():

        for i, batch in enumerate(test_dataloader):
            print('Data', i)
            

            data, z_samples = batch

            p_1, p_2, p_1_att, p_2_att, label = data
            p_1 = p_1.to(device)
            p_2 = p_2.to(device)
            p_1_att = p_1_att.to(device)
            p_2_att = p_2_att.to(device)
            label = label.to(device)
            z_s = z_samples.to(device)


            # z_s has shape (batch_size, num_Z_size, ref_size), we want to extract every z from num_z_size dimension, result in (batch_size, 1, ref_size)

            num_Z_size = z_s.shape[1]

            reward_1_list = []
            reward_2_list = []
            Eta_out_1_list = []
            Eta_out_2_list = []
            P_out_1_list = []
            P_out_2_list = []
            reward_diff_list = []


            for idx in range(num_Z_size):

                z = z_s[:, idx, :].unsqueeze(1)
                reward_1, model_out_1, Eta_out_1, P_out_1 = reward_model(p_1, p_1_att, z, log = True)
                reward_2, model_out_2, Eta_out_2, P_out_2 = reward_model(p_2, p_2_att, z, log = True)
                reward_1 = reward_1.item()
                reward_2 = reward_2.item()
                model_out_1 = model_out_1.item()
                model_out_2 = model_out_2.item()
                Eta_out_1 = Eta_out_1.item()
                Eta_out_2 = Eta_out_2.item()
                P_out_1 = P_out_1.item()
                P_out_2 = P_out_2.item()

                reward_diff = reward_1 - reward_2
                model_out_diff = model_out_1 - model_out_2

                reward_1_list.append(reward_1)
                reward_2_list.append(reward_2)
                Eta_out_1_list.append(Eta_out_1)
                Eta_out_2_list.append(Eta_out_2)
                P_out_1_list.append(P_out_1)
                P_out_2_list.append(P_out_2)
                reward_diff_list.append(reward_diff)

            print('preference is', label)
            print('model_out_1 is', model_out_1)
            print('model_out_2 is', model_out_2)
            print('model_out_diff is', model_out_diff)
            print('Eta_out_1_list is', Eta_out_1_list)
            print('Eta_out_2_list is', Eta_out_2_list)
            print('P_out_1_list is', P_out_1_list)
            print('P_out_2_list is', P_out_2_list)
            print('reward_1_list is', reward_1_list)
            print('reward 1 std', np.std(np.array(reward_1_list)))
            print('reward_2_list is', reward_2_list)
            print('reward 2 std', np.std(np.array(reward_2_list)))
            print('reward_diff_list is', reward_diff_list)
            print('reward diff std', np.std(np.array(reward_diff_list)))


            print('-------------------------------------')
