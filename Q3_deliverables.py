import sesp_utils as sesp

# Read data from csvs
neg_ctrl = sesp.read_single_column_data('data/HW4_Q3_neg.csv')
pos_ctrl = sesp.read_single_column_data('data/HW4_Q3_pos.csv')
data = sesp.read_single_column_data('data/HW4_Q3_data.csv')

# Plot the controls and data as a scatter plot
alpha = 0.25
y_label = 'Assay Value'
title = 'Testing Assay Values for Control and Sample Populations'
outfile = 'assay_plots.png'

sesp.plot_data_ctrls(neg_ctrl, pos_ctrl, data, alpha, y_label, title, outfile)

# Calculate Youden choice
dc = 0.25
se_choice, sp_choice, c_choice, J_max = sesp.find_youden_choice(neg_ctrl, pos_ctrl, data, dc)
print('Youden\'s Choice Cutoff: ' + str(c_choice))

# Plot scatter with cutoff
alpha = 0.25
y_label = 'Assay Value'
title = 'Testing Assay Values for Control and Sample Populations With Cutoff'
outfile = 'assay_plots_cutoff.png'
sesp.plot_data_ctrls_cutoff(neg_ctrl, pos_ctrl, data, c_choice, alpha, y_label, title, outfile)

# Generate ROC curve and plot
x_label = 'False Positive Rate (1-sp)'
y_label = 'True Positive Rate (se)'
title = 'Receiver Operator Curve'
outfile = 'assay_ROC.png'
sesp.plot_ROC(neg_ctrl, pos_ctrl, data, dc, x_label, y_label, title, outfile)

# Generate theta(c) plot
x_label = 'Cutoff Value'
y_label = 'Corrected Prevalence (theta)'
title = 'Corrected Prevalence vs Cutoff'
outfile = 'theta_v_cutoff.png'
sesp.plot_theta_by_c(neg_ctrl, pos_ctrl, data, dc, x_label, y_label, title, outfile)