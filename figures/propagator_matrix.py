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
model       = "mod/NMDA.mod"
model       = "mod/Nav11_6state.mod"
v_input     = LinearInput("v", -100, 100)
g_input     = LogarithmicInput('C', 0, 1e3)
all_inputs  = [v_input, g_input]
self        = LTI_Model(model, all_inputs, time_step, temperature)
# 
fig = plt.figure(figsize=(7.5, 7.5))
fig_title = self.name + " Propagator Matrix Function"
# fig.suptitle(fig_title)
gs = fig.add_gridspec(self.num_states, self.num_states,
                      hspace=0, wspace=0)
axes = gs.subplots(sharex='col', sharey='row')
line_params = dict(color='r')

if self.num_inputs == 1:
    # Sample the inputs.
    input1 = self.inputs[0]
    input1.set_num_buckets(1)
    input1 = input1.sample_space(100)
    # Compute the exact propagator matrix.
    exact = self.make_matrix(input1.reshape(1, -1))
    # Setup each subplot.
    for row_idx, row in enumerate(self.state_names):
        for col_idx, col in enumerate(self.state_names):
            # plt.title(col + " ➜ " + row)
            box = axes[row_idx, col_idx]
            if row_idx == 0:
                box.set_title("From " + col)
            if col_idx == 0:
                box.annotate("To " + row, size=12,
                    xy=(-.05, .5), xycoords='axes fraction', 
                    verticalalignment='center',
                    horizontalalignment='right',
                    rotation=90)
            if row_idx == 5 and col_idx == 2:
                box.annotate("mV", size=10,
                    xy=(1, -.2), xycoords='axes fraction', 
                    verticalalignment='top',
                    horizontalalignment='center')
            # 
            if isinstance(self.input1, LinearInput):
                box.plot(input1, exact[:, row_idx, col_idx], **line_params)
            elif isinstance(self.input1, LogarithmicInput):
                box.semilogx(input1, exact[:, row_idx, col_idx], **line_params)
            # Setup X axis.
            box.xaxis.set_ticks([-70,0,40])
            box.xaxis.set_tick_params(direction="in", top=True)
            # Setup Y axis.
            box.set_ybound(-.1, 1.1)
            box.yaxis.set_ticks([0,1])
            box.yaxis.set_tick_params(direction="in", right=True,
                labelleft=False, labelright=True)

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

fig.savefig(self.name + ".png", dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()
