import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

data_dict = io.loadmat('Time_Data')
maps, times = data_dict['data'], data_dict['times'][0]
S_active = data_dict['S_active']

model = 12
fig, axs = plt.subplots(2,4)
fig2 = plt.figure()
fig.canvas.set_window_title('Model {}'.format(model))
fig2.canvas.set_window_title('Model {}, S_active'.format(model))
axs = axs.reshape(-1)

fig2.gca()
img = plt.imshow(S_active[model,:,:], origin='lower', cmap='gray')
fig2.subplots_adjust(right=0.9)
cbar_ax2 = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
plt.colorbar(img, cax=cbar_ax2)

for idx_ax, ax in enumerate(axs):
    im = ax.imshow(maps[model,:,:,idx_ax], cmap='jet', origin='lower')
    ax.set_title(str(times[idx_ax])+'d')
    ax.set_xticks(np.arange(0, 16, step=2))
    ax.set_yticks(np.arange(0, 16, step=2))
    ax.set_xticklabels([str(e) for e in np.arange(-7.5, 8.5, 2.0)])
    ax.set_yticklabels(np.arange(-7.5, 8.5, 2.0))

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)


plt.show()
