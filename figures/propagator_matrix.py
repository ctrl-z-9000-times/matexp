"""
Plot the propagator matrix function
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matexp import main, LinearInput, LogarithmicInput
from matexp.lti_model import LTI_Model

time_step   = 0.1
temperature = 37
model       = "mod/src/NMDA.mod"
model       = "mod/src/Balbi2017/Nav11.mod"
v_input     = LinearInput("v", -100, 100)
g_input     = LogarithmicInput('C', 0, 1e3)
all_inputs  = [v_input, g_input]
self        = LTI_Model(model, all_inputs, time_step, temperature)
# 
fig_title = self.name + " Propagator Matrix Function"
plt.figure(fig_title)

if self.num_inputs == 1:
    # Sample the inputs.
    input1 = self.inputs[0]
    input1.set_num_buckets(1)
    input1 = input1.sample_space(50)
    # Compute the exact propagator matrix.
    exact = self.make_matrix(input1.reshape(1, -1))
    # 
    for row_idx, row in enumerate(self.state_names):
        for col_idx, col in enumerate(self.state_names):
            plt.subplot(self.num_states, self.num_states, row_idx*self.num_states + col_idx + 1)
            plt.title(col + " ➜ " + row)
            if isinstance(self.input1, LinearInput):
                plt.plot(input1, exact[:, row_idx, col_idx], color='k')
            elif isinstance(self.input1, LogarithmicInput):
                plt.semilogx(input1, exact[:, row_idx, col_idx], color='k')
            if self.num_states < 10: # Otherwise there is not enough room on the figure.
                plt.xlabel(self.input1.name, labelpad=1.0)
    x = .05
    plt.subplots_adjust(left=x, bottom=x, right=1-x, top=1-x, wspace=0.6, hspace=1.0)

elif self.num_inputs == 2:
    input1, input2 = self.inputs
    if isinstance(input1, LogarithmicInput): input1.scale = 0.01
    if isinstance(input2, LogarithmicInput): input2.scale = 0.01
    input1.set_num_buckets(100)
    input2.set_num_buckets(100)
    # Sample the inputs.
    pixels = 100
    input1_values = input1.sample_space(pixels)
    input2_values = input2.sample_space(pixels)
    # Make the coordinates of every point in the grid.
    inputs = np.stack(np.meshgrid(input1_values, input2_values)).reshape(2, -1)
    # Compute the exact propagator matrix.
    exact = self.make_matrix(inputs)
    # 
    for row_idx, row in enumerate(self.state_names):
        for col_idx, col in enumerate(self.state_names):
            ax = plt.subplot(self.num_states, self.num_states, row_idx*self.num_states + col_idx + 1)
            plt.title(col + " ➜ " + row)
            input1_buckets = np.array(self.input1.get_bucket_value(input1_values), dtype=int)
            input2_buckets = np.array(self.input2.get_bucket_value(input2_values), dtype=int)
            heatmap        = exact[:, row_idx, col_idx].reshape(pixels, pixels)
            imdata         = ax.imshow(heatmap, interpolation='bilinear')
            plt.colorbar(imdata, ax=ax, format='%g')
            plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.ylabel(self.input1.name)
            plt.xlabel(self.input2.name)
    x = .05
    plt.subplots_adjust(left=x, bottom=x, right=1-x, top=1-x, wspace=0.25, hspace=0.5)

plt.show()
