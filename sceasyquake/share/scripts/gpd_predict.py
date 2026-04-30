#! /usr/bin/env python3

# Automatic picking of seismic waves using Generalized Phase Detection 
# See http://scedc.caltech.edu/research-tools/deeplearning.html for more info
#
# Ross et al. (2018), Generalized Seismic Phase Detection with Deep Learning,
#                     Bull. Seismol. Soc. Am., doi:10.1785/0120180080
#                                              
# Author: Zachary E. Ross (2018)                
# Contact: zross@gps.caltech.edu                        
# Website: http://www.seismolab.caltech.edu/ross_z.html         

import argparse as ap
import os

import numpy as np
import obspy.core as oc
import pylab as plt

try:
    import tensorflow as tf
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
    except Exception:
        pass
except Exception:
    raise
#####################
# Hyperparameters
min_proba = 0.994 # Minimum softmax probability for phase detection
freq_min = 3.0
freq_max = 20.0
filter_data = False  # disabled: SeisBench GPD applies no pre-filter; both implement max-norm inside forward()
decimate_data = True # If false, assumes data is already 100 Hz samprate
n_shift = 10 # Number of samples to shift the sliding window at a time
n_gpu = 1 # Number of GPUs to use (if any)
min_distance_sec = 0.2  # Minimum time separation between picks (seconds) - matches SeisBench default
#####################
batch_size = 1000*3

half_dur = 2.00
only_dt = 0.01
n_win = int(half_dur/only_dt)
n_feat = 2*n_win

#-------------------------------------------------------------

def filter_picks_min_distance(picks_with_probs, min_distance_samples):
    """Filter picks to enforce minimum distance, keeping highest probability picks.
    
    Parameters
    ----------
    picks_with_probs : list of tuples
        List of (sample_index, probability) tuples
    min_distance_samples : int
        Minimum distance in samples between picks
        
    Returns
    -------
    list of tuples
        Filtered list of (sample_index, probability) tuples
    """
    if len(picks_with_probs) <= 1:
        return picks_with_probs
    
    # Sort by sample index
    sorted_picks = sorted(picks_with_probs, key=lambda x: x[0])
    
    # Filter keeping highest probability when picks are too close
    filtered = []
    i = 0
    while i < len(sorted_picks):
        current_idx, current_prob = sorted_picks[i]
        
        # Find all picks within min_distance
        group = [(current_idx, current_prob)]
        j = i + 1
        while j < len(sorted_picks) and sorted_picks[j][0] - current_idx < min_distance_samples:
            group.append(sorted_picks[j])
            j += 1
        
        # Keep the one with highest probability
        best_pick = max(group, key=lambda x: x[1])
        filtered.append(best_pick)
        
        # Skip to next group
        i = j
    
    return filtered


def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided


def main():
    parser = ap.ArgumentParser(
        prog='gpd_predict.py',
        description='Automatic picking of seismic waves using'
                    'Generalized Phase Detection')
    parser.add_argument(
        '-I',
        type=str,
        default=None,
        help='Input file')
    parser.add_argument(
        '-O',
        type=str,
        default=None,
        help='Output file')
    parser.add_argument(
        '-P',
        default=True,
        action='store_false',
        help='Suppress plotting output')
    parser.add_argument(
        '-V',
        default=False,
        action='store_true',
        help='verbose')
    parser.add_argument(
        '-F',
        type=str,
        default=None,
        help='path where GPD lives')
    args = parser.parse_args()

    plot = args.P
    # Delegate to reusable function
    process_dayfile(args.I, args.O, base_dir=args.F, verbose=args.V, plot=plot)


# module-level model cache (per-process) to avoid reloading on every call
_CACHED_MODEL = None


def process_dayfile(infile, outfile, base_dir=None, verbose=False, plot=False):
    """Process infile and write picks (with probability) to outfile.

    Callable programmatically; caches the loaded model per process.
    """
    global _CACHED_MODEL

    fdir = []
    with open(infile) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
    nsta = len(fdir)

    import keras

    model = _CACHED_MODEL
    base_dir = base_dir if base_dir else os.path.dirname(__file__)

    if model is None:
        # Try model files in order of preference (easyQuake 2.0 Keras 3 formats first,
        # then fall back to the original Keras 1/2 JSON + HDF5 format)
        candidates = [
            os.path.join(base_dir, 'model_pol_optimized_converted.keras'),
            os.path.join(base_dir, 'model_pol_final_converted.keras'),
            os.path.join(base_dir, 'model_pol_gpd_calibrated_F80.h5'),
            os.path.join(base_dir, 'model_pol_properly_converted.keras'),
            os.path.join(base_dir, 'model_pol_fixed.h5'),
            os.path.join(base_dir, 'model_pol_new.keras'),
            os.path.join(base_dir, 'model_pol_legacy.h5'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    model = keras.models.load_model(path)
                    print(f"Loaded GPD model from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {e}")

        # Legacy fallback: model_pol.json + model_pol_best.hdf5 (easyQuake < 2.0)
        if model is None:
            json_path = os.path.join(base_dir, 'model_pol.json')
            hdf5_path = os.path.join(base_dir, 'model_pol_best.hdf5')
            if os.path.isfile(json_path) and os.path.isfile(hdf5_path):
                try:
                    from keras.models import model_from_json
                    with open(json_path) as jf:
                        model = model_from_json(jf.read())
                    model.load_weights(hdf5_path)
                    print(f"Loaded GPD model (legacy JSON+HDF5) from: {base_dir}")
                except Exception as e:
                    print(f"Failed to load legacy model: {e}")

        if model is None:
            raise RuntimeError(
                "Failed to load any GPD model variant. "
                "Run easyQuake model conversion or reinstall easyQuake."
            )
        _CACHED_MODEL = model

    if n_gpu > 1:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=n_gpu)

    ofile = open(outfile, 'w')

    for i in range(nsta):
        try:
            fname = fdir[i][0].split("/")
            if not os.path.isfile(fdir[i][0]):
                print("%s doesn't exist, skipping" % fdir[i][0])
                continue
            if not os.path.isfile(fdir[i][1]):
                print("%s doesn't exist, skipping" % fdir[i][1])
                continue
            if not os.path.isfile(fdir[i][2]):
                print("%s doesn't exist, skipping" % fdir[i][2])
                continue
            st = oc.Stream()
            st += oc.read(fdir[i][0])  # N
            st += oc.read(fdir[i][1])  # E
            st += oc.read(fdir[i][2])  # Z
            st.sort(['channel'])
            st.detrend(type='linear')
            # Bandpass filter and taper are disabled so GPD-easyquake and
            # GPD-seisbench receive identically preprocessed data.
            # SeisBench GPD applies no stream-level filter; max-normalization
            # is done inside the model forward() for both implementations.
            if decimate_data:
                st.resample(100.0)
            st.merge(fill_value='interpolate')
            print(st)
            for tr in st:
                if isinstance(tr.data, np.ma.masked_array):
                    tr.data = tr.data.filled()

            chan = st[0].stats.channel
            sr = st[0].stats.sampling_rate
            dt = st[0].stats.delta
            net = st[0].stats.network
            sta = st[0].stats.station
            chan = st[0].stats.channel
            latest_start = np.max([x.stats.starttime for x in st])
            earliest_stop = np.min([x.stats.endtime for x in st])
            if (earliest_stop > latest_start):
                st.trim(latest_start, earliest_stop)
                if verbose:
                    print("Reshaping data matrix for sliding window")
                tt = (np.arange(0, st[0].data.size, n_shift) + n_win) * dt
                tt_i = np.arange(0, st[0].data.size, n_shift) + n_feat
                #tr_win = np.zeros((tt.size, n_feat, 3))
                sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)
                sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)
                sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)
                tr_win = np.zeros((sliding_N.shape[0], n_feat, 3))
                tr_win[:,:,0] = sliding_N
                tr_win[:,:,1] = sliding_E
                tr_win[:,:,2] = sliding_Z
                tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
                tt = tt[:tr_win.shape[0]]
                tt_i = tt_i[:tr_win.shape[0]]
        
                if verbose:
                    ts = model.predict(tr_win, verbose=True, batch_size=batch_size)
                else:
                    ts = model.predict(tr_win, verbose=False, batch_size=batch_size)
        
                prob_S = ts[:,1]
                prob_P = ts[:,0]
                prob_N = ts[:,2]
        
                # Calculate minimum distance in samples
                min_distance_samples = int(min_distance_sec / dt)
        
                from obspy.signal.trigger import trigger_onset
                
                # Extract P picks
                trigs = trigger_onset(prob_P, min_proba, 0.1)
                p_picks = []
                p_picks_with_probs = []  # Store (index, prob) for filtering
                for trig in trigs:
                    if trig[1] == trig[0]:
                        continue
                    pick = np.argmax(ts[trig[0]:trig[1], 0])+trig[0]
                    p_picks_with_probs.append((pick, float(prob_P[pick])))
                
                # Filter P picks to enforce minimum distance
                p_picks_filtered = filter_picks_min_distance(p_picks_with_probs, min_distance_samples)
                for pick, pick_prob in p_picks_filtered:
                    stamp_pick = st[0].stats.starttime + tt[pick]
                    chan_pick = st[0].stats.channel[0:2]+'Z'
                    p_picks.append(stamp_pick)
                    ofile.write("%s %s %s P %s %.6f\n" % (net, sta, chan_pick, stamp_pick.isoformat(), pick_prob))
        
                # Extract S picks
                trigs = trigger_onset(prob_S, min_proba, 0.1)
                s_picks = []
                s_picks_with_probs = []  # Store (index, prob) for filtering
                for trig in trigs:
                    if trig[1] == trig[0]:
                        continue
                    pick = np.argmax(ts[trig[0]:trig[1], 1])+trig[0]
                    s_picks_with_probs.append((pick, float(prob_S[pick])))
                
                # Filter S picks to enforce minimum distance
                s_picks_filtered = filter_picks_min_distance(s_picks_with_probs, min_distance_samples)
                for pick, pick_prob in s_picks_filtered:
                    stamp_pick = st[0].stats.starttime + tt[pick]
                    chan_pick_s = st[0].stats.channel[0:2]+'E'
                    s_picks.append(stamp_pick)
                    ofile.write("%s %s %s S %s %.6f\n" % (net, sta, chan_pick_s, stamp_pick.isoformat(), pick_prob))
        
                if plot:
                    fig = plt.figure(figsize=(8, 12))
                    ax = []
                    ax.append(fig.add_subplot(4,1,1))
                    ax.append(fig.add_subplot(4,1,2,sharex=ax[0],sharey=ax[0]))
                    ax.append(fig.add_subplot(4,1,3,sharex=ax[0],sharey=ax[0]))
                    ax.append(fig.add_subplot(4,1,4,sharex=ax[0]))
                    for i in range(3):
                        ax[i].plot(np.arange(st[i].data.size)*dt, st[i].data, c='k', \
                                   lw=0.5)
                    ax[3].plot(tt, ts[:,0], c='r', lw=0.5)
                    ax[3].plot(tt, ts[:,1], c='b', lw=0.5)
                    for p_pick in p_picks:
                        for i in range(3):
                            ax[i].axvline(p_pick-st[0].stats.starttime, c='r', lw=0.5)
                    for s_pick in s_picks:
                        for i in range(3):
                            ax[i].axvline(s_pick-st[0].stats.starttime, c='b', lw=0.5)
                    plt.tight_layout()
                    plt.show()
        except Exception as e:
            print(f"Exception processing station {i}: {e}")
            import traceback
            traceback.print_exc()
    ofile.close()


if __name__ == "__main__":
    main()
