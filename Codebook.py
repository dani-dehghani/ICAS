#%%
import numpy as np
import matplotlib.pyplot as plt

class Codebook:

    def __init__(self ,L_max , N1, O1 = 4, az_min = -60, az_max = 60 ):        
        self.c0 = 299792458.0   # Speed of light in m/s
        self.L_max = L_max      # Max. number of beams
        self.N1 = N1            # Number of antennas in hor. direction
        self.O1 = O1            # Oversampling
        self.az_min = az_min
        self.az_max = az_max

    def steer_vec(self, n_elements, thetas_rad, electric_length=0.5, centered=False):
        phase_delta = electric_length * 2 * np.pi * np.sin(thetas_rad)
        el_array = np.arange(n_elements, dtype=float)
        if centered:
            el_array -= (n_elements - 1) / 2
        phase = el_array[:, np.newaxis] * np.expand_dims(phase_delta, axis=0)
        return np.exp(1j * phase)

    def mag2db(self, mag_value):
        return 20 * np.log10(mag_value)

    def dft_beamforming(self, steer_index, n_antennas, oversampling=4, normalize=True):
        phases = 2 * np.pi * steer_index * np.arange(n_antennas) / (n_antennas * oversampling)
        vec = np.exp(1j * phases)
        if normalize:
            vec /= np.sqrt(n_antennas)
        return vec

    def array_factor(self, antenna_weights, thetas_degrees, n_antennas=None):
        if n_antennas is None:
            n_antennas = len(antenna_weights)

        def sv(rad, n_ant):
            return 1j * np.pi * np.arange(n_ant) * np.sin(rad)

        aw = antenna_weights.reshape(n_antennas, 1)
        af = np.array([self.steer_vec(n_antennas, d).reshape(1, n_antennas).conj() @ antenna_weights
                       for d in np.deg2rad(thetas_degrees)]).squeeze()
        return af

    def beamforming_vectors(self):
        worst_beam_width = np.rad2deg(2 / (self.N1 * np.cos(np.deg2rad(self.az_max))))
        beam_degs = np.linspace(self.az_min + worst_beam_width / 2, self.az_max - worst_beam_width / 2, self.L_max)
        beam_ang_freq = np.pi * np.sin(np.deg2rad(beam_degs))
        l_indices = np.round(beam_ang_freq * self.N1 * self.O1 / (2 * np.pi)).astype(int)
        dft_vectors = [self.dft_beamforming(l, self.N1, oversampling=self.O1) for l in l_indices]
        return dft_vectors ,beam_degs

    def plot_beamforming_polar(self, dft_vectors):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location("N")
        ax.set_thetalim(-np.pi/2, np.pi/2)
        ax.set_xlabel("Directivity [dBi]")
        thetas_eval = np.linspace(-90, 90, 1000)

        for dft_vec in dft_vectors:
            bf = self.array_factor(dft_vec, thetas_eval)
            bf_db = self.mag2db(np.abs(bf))
            ax.plot(np.deg2rad(thetas_eval), bf_db)

        ax.set_rlim(-10, None)
        plt.show()


#codebook_instance = Codebook()
#beamforming_vectors = codebook_instance.beamforming_vectors()
#codebook_instance.plot_beamforming_polar(beamforming_vectors)

# %%
