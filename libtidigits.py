__author__ = 'sbraun'
import numpy as np
#import matplotlib.pyplot as plt

verbose_mode = 0

def train_normalizer(X_train):
    msg = 0
    # get 2d array for mean and std calculation
    X_train_mat = np.hstack(X_train)

    # mean and std calculation
    mean_vec = np.mean(X_train_mat, axis=1)
    std_vec = np.std(X_train_mat, axis=1)

    # zero mean and unit variance normalization

    X_train_n1 = X_train.copy()
    X_train_n2 = X_train.copy()
    for i in range(0, len(X_train)):
        X_train_n1[i] = X_train[i] - mean_vec[:,
                                     np.newaxis]  # zero mean, http://stackoverflow.com/questions/8423051/remove-mean-from-numpy-matrix
        X_train_n2[i] = X_train_n1[i] / std_vec[:, np.newaxis]  # unit variance

    for i in range(0, len(X_train)):
        if np.isnan(X_train_n2[i]).any():
            X_train_n2[i][np.isnan(X_train_n2[i])] = 0
            msg = 1
    if msg == 1:
        print('Divide by zero solved by replacing NaNs with zeros.')

    return X_train_n2

def normalizer_params(X_train):
    # get 2d array for mean and std calculation
    X_train_mat = np.hstack(X_train)

    # mean and std calculation
    mean_vec = np.mean(X_train_mat, axis=1)
    std_vec = np.std(X_train_mat, axis=1)

    return mean_vec, std_vec

def test_normalizer(X_test, mean_vec, std_vec):
    # get 2d array for mean and std calculation

    # mean and std calculation
    #mean_vec = np.mean(X_train_mat, axis=1)
    #std_vec = np.std(X_train_mat, axis=1)

    # zero mean and unit variance normalization

    X_test_n1 = X_test.copy()
    X_test_n2 = X_test.copy()
    for i in range(0, len(X_test)):
        X_test_n1[i] = X_test[i] - mean_vec[:,
                                     np.newaxis]  # zero mean, http://stackoverflow.com/questions/8423051/remove-mean-from-numpy-matrix
        X_test_n2[i] = X_test_n1[i] / std_vec[:, np.newaxis]  # unit variance

    return X_test_n2

# Convert TI-Digits into actual digits
def char_to_dig(char_dig):
    if char_dig == 'O' or char_dig == 'Z':
        return 0
    else:
        return int(char_dig)


# Define a function to zero-pad data
def pad_sequences(sequences, max_len, dtype='float32', padding='pre', truncating='pre', transpose=True, value=0.):
    # (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

    nb_samples = len(sequences)
    x = (np.ones((nb_samples, max_len, sequences[0].shape[0])) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[:, -max_len:]
        elif truncating == 'post':
            trunc = s[:, :max_len]
        if transpose:
            trunc = trunc.T
        if padding == 'post':
            x[idx, :len(trunc), :] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):, :] = trunc
    return x


#def plot_confusion_matrix(cm, normalize=1, title='Confusion matrix', cmap=plt.cm.viridis):
#    if normalize == 1:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)
#    plt.tight_layout()
#    plt.clim(0,1)
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')

def vprint(string):
    global verbose_mode
    if verbose_mode == 1:
        print(string)

#def tetris(diff_array, sample_range, SNR, y_true):
   # fig = plt.figure()
   # ax = fig.add_subplot(111)
   # plt.imshow(diff_array[:, sample_range], interpolation='none', cmap='viridis', aspect='auto')
   # cb = plt.colorbar()
   # plt.clim(-1*np.max(y_true), 0)
   # cb.set_label('Label difference')
   # locs = np.arange(len(SNR))
   # for axis in [ax.yaxis]:
   #     axis.set_ticks(locs + 0.5, minor=True)
   #     axis.set(ticks=locs, ticklabels=SNR)
   # locs = np.arange(len(sample_range))
   # ax.grid(True, which='minor')

   # if len(sample_range)<=50:
   #     for axis in [ax.xaxis]:
   #         axis.set_ticks(locs + 0.5, minor=True)
   #         axis.set(ticks=locs, ticklabels=sample_range)
   #     # Turn on the grid for the minor ticks
   #     ax.grid(True, which='minor')
   #     plt.ylabel('SNR [dB]')
   #     plt.xlabel('Sample #')
   #     ax2 = fig.add_axes(ax.get_position(), frameon=False)
   #     ax2.tick_params(labelbottom='off', labeltop='on', labelleft="off", labelright='off',
   #                     bottom='off', left='off', right='off')
   #     ax2.set_xlim(ax.get_xlim())
   #     ax2.set_xticks(np.arange(len(sample_range)))
   #     ax2.set_xticklabels(y_true[sample_range], minor=False)

#def tetris_proba(heat_array, sample_range, SNR, y_true):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.imshow(heat_array[:, sample_range], interpolation='none', cmap='viridis', aspect='auto')
    # plt.pcolor(np.flipud(heat_array[:, sample_range]), cmap='viridis', edgecolor='black', linestyle='-', lw=1)
    #cb = plt.colorbar()
    #plt.clim(0, 1)
    #cb.set_label('Class probability')
    #locs = np.arange(len(SNR))
    #for axis in [ax.yaxis]:
    #    axis.set_ticks(locs + 0.5, minor=True)
    #    axis.set(ticks=locs, ticklabels=SNR)
    #locs = np.arange(len(sample_range))
    #for axis in [ax.xaxis]:
    #    axis.set_ticks(locs + 0.5, minor=True)
    #    axis.set(ticks=locs, ticklabels=sample_range)
    # Turn on the grid for the minor ticks
    #ax.grid(True, which='minor')
    #plt.ylabel('SNR [dB]')
    #plt.xlabel('Sample #')

    #ax2 = fig.add_axes(ax.get_position(), frameon=False)
    #ax2.tick_params(labelbottom='off', labeltop='on', labelleft="off", labelright='off',
    #                bottom='off', left='off', right='off')
    #ax2.set_xlim(ax.get_xlim())
    #ax2.set_xticks(np.arange(len(sample_range)))
    #ax2.set_xticklabels(y_true[sample_range], minor=False)
