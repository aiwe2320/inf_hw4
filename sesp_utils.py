import numpy as np
import numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt


def read_single_column_data(input_file):
    '''
    Read data from a file with a single column of floats and store as a list
    '''
    data = []
    with open(input_file, 'r') as f:
        if not f:
            return None  # File empty
        else:
            for line in f:
                data.append(float(line))
    
    return data


def plot_data_ctrls(neg_ctrl, pos_ctrl, data, alpha, y_label, title, outfile):
    '''
    Take data from the negative and positive controls and the data and plot on a scatter plot
    '''
    fig, ax = plt.subplots()
    
    # Use a random normal distribution to disperse points in a given category
    neg_ctrl_x = rnd.normal(1, 0.1, len(neg_ctrl))
    pos_ctrl_x = rnd.normal(2, 0.1, len(pos_ctrl))
    data_x = rnd.normal(3, 0.1, len(data))
    
    # Plot data using scatter
    ax.scatter(neg_ctrl_x, neg_ctrl, alpha=alpha, color='red')
    ax.scatter(pos_ctrl_x, pos_ctrl, alpha=alpha, color='black')
    ax.scatter(data_x, data, alpha=alpha, color='blue')
    
    # Set labels
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Negative Control', 'Positive Control', 'Data'])
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    plt.savefig(outfile, bbox_inches='tight')
    

def calc_se(pos_ctrl, c):
    '''
    Calculate the sensitivity of the test using positive control data
    '''
    # Sensitivity = Pr(test + given truly +) = True positive rate = TP/(TP+FN)
    TP = 0  # Count number of true positives in positive control set
    FN = 0  # Count number of false negatives
    for val in pos_ctrl:
        if (val >= c):
            TP += 1
        else:
            FN += 1
    
    se = TP / (TP + FN)
    
    return se


def calc_sp(neg_ctrl, c):
    '''
    Calculate the specificity of the test using negative control data
    '''
    # Sensitivity = Pr(test - given truly -) = True negative rate = TN/(TN+FP)
    FP = 0  # Count false positives
    TN = 0  # Count true negatives
    for val in neg_ctrl:
        if (val >= c):
            FP += 1
        else:
            TN += 1
    
    sp = TN / (TN + FP)
    
    return sp


def calc_raw_prevalence(data, c):
    '''
    Calculate the raw prevalence from testing data
    '''
    # Raw prevalence = npos / ntot
    npos = 0  # Count positive tests based on cutoff (c)
    for val in data:
        if (val >= c):
            npos += 1
    
    phi = npos / len(data)
    
    return phi

def calc_corrected_prevalence(neg_ctrl, pos_ctrl, data, c):
    '''
    Calculate the corrected prevalence from testing data and controls
    '''
    # Corrected prevalence = theta = (phi - (1 - sp)) / (se + sp - 1)
    phi = calc_raw_prevalence(data, c)
    se = calc_se(pos_ctrl, c)
    sp = calc_sp(neg_ctrl, c)
    
    if (se + sp - 1 == 0):
        theta = None
    else:
        theta = (phi - (1 - sp)) / (se + sp - 1)
    
    return theta

def generate_c_list(c0, dc, cmax):
    '''
    Generate list of cutoffs for evaluation
    INCLUSIVE OF CMAX
    '''
    c_list = []
    # If 0<dc<1, can't use range function
    if (dc > 0 and dc < 1):
        count = c0
        while (count <= cmax):
            c_list.append(count)
            count += dc
    else:  # Get list of times using range
        c_list = list(range(c0, cmax, dc))
        c_list.append(cmax)
    return c_list


def find_total_max(neg_ctrl, pos_ctrl, data):
    '''
    Find the absolute maximum assay value among all sets of data
    '''
    data_max = max(neg_ctrl)
    if (max(pos_ctrl) > data_max):
        data_max = max(pos_ctrl)
    if (max(data) > data_max):
        data_max = max(data)
    
    return data_max


def find_youden_choice(neg_ctrl, pos_ctrl, data, dc):
    '''
    Find Youden choice, the best cutoff to maximize test specificity/sensitivity
    '''
    cmax = find_total_max(neg_ctrl, pos_ctrl, data)  # Find absolute maximum of data to bound search
    c_list = generate_c_list(c0=0, dc=dc, cmax=cmax)  # Generate list of cutoffs (inclusive of cmax)
    
    # Iterate over potential cutoff values to find Youden choice
    J_max = -1
    c_choice = -1
    se_choice = -1
    sp_choice = -1
    for c in c_list:
        se = calc_se(pos_ctrl, c)
        sp = calc_sp(neg_ctrl, c)
        youden_ndx = se + sp - 1
        if (youden_ndx > J_max):  # If Youden index is greater than previous max, store index, cutoff, se, sp
            J_max = youden_ndx
            c_choice = c
            se_choice = se
            sp_choice = sp
    
    return se_choice, sp_choice, c_choice, J_max


def plot_data_ctrls_cutoff(neg_ctrl, pos_ctrl, data, c, alpha, y_label, title, outfile):
    '''
    Take data from the negative and positive controls and the data and plot on a scatter plot, with a cutoff (c)
    '''
    fig, ax = plt.subplots()
    
    # Use a random normal distribution to disperse points in a given category
    neg_ctrl_x = rnd.normal(1, 0.1, len(neg_ctrl))
    pos_ctrl_x = rnd.normal(2, 0.1, len(pos_ctrl))
    data_x = rnd.normal(3, 0.1, len(data))
    cx = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    c_list = [c]*len(cx)
    
    # Plot data using scatter
    ax.scatter(neg_ctrl_x, neg_ctrl, alpha=alpha, color='red')
    ax.scatter(pos_ctrl_x, pos_ctrl, alpha=alpha, color='black')
    ax.scatter(data_x, data, alpha=alpha, color='blue')
    ax.plot(cx, c_list, linestyle='--', color='magenta', label='Cutoff Value')
    
    # Set labels
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Negative Control', 'Positive Control', 'Data'])
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    
    plt.savefig(outfile, bbox_inches='tight')


def plot_theta_by_c(neg_ctrl, pos_ctrl, data, dc, x_label, y_label, title, outfile):
    '''
    For a range of values of cutoff (c), calculate corrected prevalence (theta)
    '''
    cmax = find_total_max(neg_ctrl, pos_ctrl, data)  # Find absolute maximum of data to bound search
    c_list = generate_c_list(c0=0, dc=dc, cmax=cmax)  # Generate list of cutoffs (inclusive of cmax)
    
    # Calculate theta for each value of c
    theta_list = []
    for c in c_list:
        theta_tmp = calc_corrected_prevalence(neg_ctrl, pos_ctrl, data, c)
        theta_list.append(theta_tmp)
    
    # Get Youden's choice
    se_choice, sp_choice, c_choice, J_max = find_youden_choice(neg_ctrl, pos_ctrl, data, dc)
    youden_theta = calc_corrected_prevalence(neg_ctrl, pos_ctrl, data, c_choice)
    
    # Plot ROC + Youden's point
    fig, ax = plt.subplots()
    
    ax.plot(c_list, theta_list, color='black')
    ax.scatter(c_choice, youden_theta, color='blue', label="Youden's choice")
    
    # Set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    plt.savefig(outfile, bbox_inches='tight')


def plot_ROC(neg_ctrl, pos_ctrl, data, dc, x_label, y_label, title, outfile):
    '''
    For a range of cutoff values, plot the receiver operator curve (ROC) and put a point at the Youden's choice
    '''
    cmax = find_total_max(neg_ctrl, pos_ctrl, data)  # Find absolute maximum of data to bound search
    c_list = generate_c_list(c0=0, dc=dc, cmax=cmax)  # Generate list of cutoffs (inclusive of cmax)
    
    # Get sensitivity, specificity for each cutoff value to construct plot
    # Note that the TP rate = se while the FP rate = 1 - sp
    TP_list = []
    FP_list = []
    for c in c_list:
        TP_list.append(calc_se(pos_ctrl, c))
        FP_list.append(1 - calc_sp(neg_ctrl, c))
    
    # Get Youden's choice
    se_choice, sp_choice, c_choice, J_max = find_youden_choice(neg_ctrl, pos_ctrl, data, dc)
    
    # Plot ROC + Youden's point
    fig, ax = plt.subplots()
    
    ax.plot(FP_list, TP_list, color='black')
    ax.scatter((1 - sp_choice), se_choice, color='blue', label="Youden's choice")
    
    # Set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    
    plt.savefig(outfile, bbox_inches='tight')