# Demonstrates a method for computation of resolution and noise in MR images by 
# non-linear and non-static methods. Simulates data and then applies simple 
# Fourier transform reconstruction, for which resolution and g-factors are then estimated.

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

############################################# Define reconstruction functions and load data
def model_recofunction(kspace):
    # This should be implemented by the user. 
    # should return a reconstruction of the given k-space as numpy.ndarray
    
    # in this simple example we will use just an inverse fft with RSSQ-coil combination
    imagespace = np.fft.ifft2(kspace, axes=(-2,-1))
    return np.sqrt(np.square(np.absolute(imagespace)).sum(0))

def load_data(imsize, coils, acs_size, R):
    # This should be implemented by the user. 
 
    ### Load data
    # in this method, the user should load: 
    #   fully sampled multi-coil k-space data
    #   undersampling mask
    
    # for this demo, we will generate a simple phantom:
    im = np.ones((coils,)+imsize, dtype=np.complex64)
    for c in range(coils):
        xcenter = np.random.randint(0, imsize[0])
        ycenter= np.random.randint(0, imsize[1])
        rad =  np.random.randint(imsize[0])
        for x in range(imsize[0]):
            for y in range(imsize[1]):
                if ((x-xcenter)**2 + (y-ycenter)**2) < rad**2:
                    im[c,x,y] = 1 + 1j

    gtkspace = inverse_recofunction_coilwise(im) # fully sampled multi-coil k-space data
        
    # for this demo, we will use a regular undersampling pattern with fully sampled center region
    mask = np.repeat(np.expand_dims(np.arange(imsize[0]) % R == 0,0), imsize[1], axis=0) 
    mask[:,:acs_size] = 1
    mask[:,-acs_size:] = 1
    
    # for the measured noise, we just use uncorrelated gaussian noise in this example
    gaussnoise = np.reshape(np.random.standard_normal(coils*imsize[0]*imsize[1]),[coils,imsize[0]*imsize[1]]).astype(np.complex64)
    gaussnoise += 1j*np.reshape(np.random.standard_normal(coils*imsize[0]*imsize[1]),[coils,imsize[0]*imsize[1]]).astype(np.complex64)
    
    return gtkspace, mask, gaussnoise
#############################################

def fully_sampled_recofunction_coilwise(kspace):
    return  np.fft.ifft2(kspace, axes=(-2,-1))

def inverse_recofunction_coilwise(image):
    return  np.fft.fft2(image, axes=(-2,-1))

def fftinterp(signal, num_interp_times):
    # Interpolates a 1-dimensional signal via the Fourier method to num_interp_times-fold resolution
    signal_len = signal.shape[0]
    num_interp = num_interp_times*signal_len
    energy_ratio = int((num_interp*2 + signal_len) / signal_len) # factor by which to multiply the interpolated signal, to preserve energy, and thus peak height
    ffted = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(signal)))
    expanded = np.zeros(signal_len + num_interp*2,np.complex64)
    expanded[num_interp:num_interp+signal_len] = ffted
    interpolated = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(expanded)))
    interpolated = interpolated * energy_ratio
    return np.real(interpolated)

def get_peak_width(signal, at_height = 2/math.pi, position=None, interval_width = 5): # at_height is the height at which to measure the width of the mainlobe. the rayleigh criterion gives 2/pi.
    # Given a 1-dimensional signal, determines the width of the main maximum
    
    # To avoid finding unwanted peaks, only search for the peak in a certain interval around the position where the mainlobe is expected
    if position != None:
        interval = signal[max(0,position-interval_width):min(signal.shape[0],position+interval_width +1)]
    else:
        interval = signal
    peak_pos = np.argmax(np.abs(interval)) # find the position of the peak
    if position != None:
        peak_pos = position - interval_width + peak_pos 
    
    peak_height = signal[peak_pos]
    eval_height = peak_height * at_height
    
    b1 = 0
    b2 = signal.size-1
    for i in range(1,signal.size):
        point1 = peak_pos - i
        point2 = peak_pos + i
        if point1 >= 0:
            if signal[point1] <= eval_height and b1 == 0:
                b1 = point1
        if point2 < signal.size:
            if signal[point2] <= eval_height and b2 == signal.size-1:
                b2 = point2
        if b1 != 0 and b2 != signal.size-1: 
            break
            
    border1 = (eval_height - signal[b1] + (signal[b1+1] - signal[b1])*b1)/(signal[b1+1] - signal[b1])
    border2 = (eval_height - signal[b2-1] + (signal[b2]- signal[b2-1])*(b2-1))/(signal[b2] - signal[b2-1])
    
    width = border2-border1
    width = min(max(width,0),signal.shape[0])
    
    return width

def compute_res_map(gtkspace, mask):
    
    ### Compute ground truth and unperturbed undersampled reconstruction

    gtcw = fully_sampled_recofunction_coilwise(gtkspace) # coil-wise fully sampled ground truth reconstruction
    gt = np.sqrt(np.square(np.absolute(fully_sampled_recofunction_coilwise(gtkspace))).sum(0)) # ground-truth RSSQ-reconstruction
    unperturbed_model_reco = model_recofunction(gtkspace*mask) # compute unperturbed model reconstruction

    ### Compute resolution maps
    
    num_interp_times = 5  # by how many times the resolution should be increased through Fourier interpolation before measure main lobe width
    
    # set the amplitude of perturbations to 0.1% of global maximum pixel value of the root-sum-of-squares fully sampled reconstruction
    # experimentally, we found that our methods behave linearly for such perturbations
    # this has to be ensured for any reconstruction method the method should be applied to
    perturbation_ampl = np.abs(gt).max()*0.001 
    
    res_map = np.zeros([gt.shape[0]-2, gt.shape[1]-2, 2], dtype = np.float64) # do not compute the resolution maps at the edge pixels, since there are problems there
    
    # loop over all pixels for which the resolution map should be computed
    for x in range(0,res_map.shape[0],1): 
        for y in range(0,res_map.shape[1],1):
            pixel = (y+1,x+1)
            
            perturbed_image_coilwise = np.copy(gtcw)
            proportion = (perturbation_ampl + gt[pixel]) / gt[pixel] + 0j
            
            for coil in range(perturbed_image_coilwise.shape[0]):
                perturbed_image_coilwise[(coil,)+pixel] = perturbed_image_coilwise[(coil,)+pixel] * proportion # apply perturbation; the argument of the perturbed pixels are kept 
            perturbed_kspace = inverse_recofunction_coilwise(perturbed_image_coilwise).astype(np.complex64)    #  transform perturbed image to k-space and apply undersampling
            perturbed_kspace = perturbed_kspace*mask
            perturbed_modelreco = model_recofunction(perturbed_kspace) # apply model-based reconstruction
            
            lpsf2d = np.abs(unperturbed_model_reco) - np.abs(perturbed_modelreco) #  lpsf is the difference of the perturbed and unperturbed reconstruction
            
            for dimension in range(2):
                # extract row or column, depending on which dimension the resolution should be computed in
                if dimension == 0: # horizontal
                    lpsf = lpsf2d[pixel[0],:]
                elif dimension == 1: # vertical
                    lpsf = lpsf2d[:, pixel[1]]
                
                interpolated = np.abs(fftinterp(lpsf,num_interp_times=num_interp_times)) # Fourier-interpolate lpsf to better measure mainlobe width
                peak_width = get_peak_width(interpolated, position=(pixel[1-dimension]*(num_interp_times*2 +1))) # measure mainlobe width, which is a measure of spatial resolution
                res_map[y, x, dimension] = peak_width / (num_interp_times*2 +1) # scale, to compensate for interpolation
                
            print("\rComputing resolution map: {:.2f}%".format(100*((y+1)+(x)*res_map.shape[0])/(res_map.shape[0]*res_map.shape[1])), end="")
    print("")
    
    return res_map, gt, unperturbed_model_reco

def compute_noise_cov(noise):
    # expects measured noise as  coils x ...
    N =  np.prod(np.array(noise[0].shape))
    noise = np.reshape(noise,[noise.shape[0],-1],order="F")
    noise_cov = noise @ np.transpose(np.conjugate(noise))  / (N) 
    eigVal, eigVec =  np.linalg.eig(noise_cov)
    eigVal = np.sqrt(eigVal)
    eigVal = np.diag(eigVal)
    eigVecInv = np.linalg.inv(eigVec)
    noiseCovSqrt = eigVec @ eigVal @ eigVecInv
    return noiseCovSqrt.astype(np.complex64)

def compute_gfmap(numrepl, nconv, gtkspace, mask, R):
    nc, ny, nx = gtkspace.shape
    noise_model_recos = np.zeros((numrepl,ny,nx))
    noise_full_rss_recos = np.zeros((numrepl,ny,nx))
    
    for nr in range(numrepl):
    
        # generate complex gaussian noise
        gaussnoise = np.reshape(np.random.standard_normal(nc*ny*nx),[nc,ny*nx]).astype(np.complex64)
        gaussnoise += 1j*np.reshape(np.random.standard_normal(nc*ny*nx),[nc,ny*nx]).astype(np.complex64) 
        
        # use measured covariance to generate synthetic noise
        synnoise = nconv @ gaussnoise
        synnoise = np.reshape(synnoise,[nc,ny,nx],order = 'F')
    
        # add noise to kspace and do fully sampled rss recostruction
        noisy_ksp = (gtkspace+synnoise)
        ffted = np.fft.ifft2(noisy_ksp, axes=[-2,-1])
        noise_full_rss_recos[nr] =  np.sqrt(np.sum(np.square(np.absolute(ffted)),0))
        
        # apply undersampling and do model reconstruction
        noisy_usksp = noisy_ksp*mask
        noise_model_recos[nr] = model_recofunction(noisy_usksp)
        
        print("\rComputing g-factor map: {:.2f}%".format(100*((nr+1)/numrepl)), end="")
    
    # the factor scal is chosen such that |usreco_rss-scal*usreco_model|_2^2 is minimal
    # we multiply this factor to the model reconstructions to give then the same scale as the rss reconstructions, because the models usually return images in arbitrary scale. 
    usreco_rss = np.sqrt(np.sum(np.square(np.absolute(np.fft.ifft2(gtkspace*mask, axes=[-2,-1]))),0))
    usreco_model = model_recofunction(gtkspace*mask)
    scal = np.dot(usreco_model.flatten(), usreco_rss.flatten()) / np.dot(usreco_model.flatten(), usreco_model.flatten())
    
    gfmap = np.std(noise_model_recos*scal, axis=0) / (np.std(noise_full_rss_recos, axis=0)*math.sqrt(R))
    
    return gfmap

if __name__ == "__main__":
    
    ### Config
    imsize = (100,100)
    coils = 10 #  number of simulated coils
    R = 3 # acceleration factor for simulated data
    acs_size = 6 #  size of fully sampled center region for simulated data
    numrepl = 10000 # number of replicas for g-factor map computation
    np.random.seed(173)
    
    ### Compute resolution and g-factor maps
    # compute resolution maps
    gtkspace, mask, noise = load_data(imsize, coils, acs_size, R)
    res_map, ground_truth, unperturbed_model_reco = compute_res_map(gtkspace, mask)

    # compute g-factor map
    nconv = compute_noise_cov(noise)
    gfmap = compute_gfmap(numrepl, nconv, gtkspace, mask, R)
        
    ### Visualization
    fig = plt.figure(figsize=(10,6.5), dpi = 300)
    gs = matplotlib.gridspec.GridSpec(2, 3, height_ratios=[1,1.4], hspace=0.0)
    
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(ground_truth, cmap="gray")
    ax.set_title("Ground truth")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(unperturbed_model_reco, cmap="gray")
    ax.set_title("Undersampled FFT reconstruction")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(np.fft.fftshift(mask, axes=[-2,-1]), cmap="gray")
    ax.set_title("Undersampling mask")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    s = 0.63
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(res_map[:,:,0], cmap="jet")
    ax.set_title("Horizontal resolution map")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(im, ax = ax, format="%.3f", shrink=s)
    
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(res_map[:,:,1], cmap="jet")
    ax.set_title("Vertical resolution map")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(im, ax = ax, format="%.3f", shrink=s)
    
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(gfmap, cmap="viridis") #, vmin=0, vmax=1.5
    ax.set_title("g-factor map")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(im, ax = ax, format="%.3f", shrink=s)
    
    plt.show()
    
    # Both resolution maps are (almost) constant, since the fft is shift-invariant
    # The vertical resolution map is constant 1, since no undersampling is applied in this direction, and the fft reconstruction acts independently on the dimensions
    # The horizontal resolution map is constant at a value larger than 1, since the undersampling applies blurring, which is not corrected by the fft reconstruction
    # The g-factor map is constant at a value below 1, since the blurring introduced in the undersampling suppresses noise
    