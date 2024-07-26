import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.size'] = 18
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")


def load_data(filename, unpack=True):
    return np.loadtxt(filename, unpack=unpack)

def plot(x, y, xlabel=None, ylabel=None, label=None):
    plt.plot(x, y, label=label)

    if label is not(None):
        plt.legend()

    if xlabel is not(None):
        plt.xlabel(xlabel)
    
    if ylabel is not(None):
        plt.ylabel(ylabel)

def xlim(xmin, xmax):
    plt.xlim([xmin, xmax])

def ylim(ymin, ymax):
    plt.ylim([ymin, ymax])

def gaussian(x, mu, std_dev, amplitude):
    return amplitude*np.exp(-1/2*((x - mu)/std_dev)**2)

def beerlambert(wavelength, k, n, l=100, r=1000):
    l = l # переводим метры в км
    tau = np.exp(-k*n*l)
    if r is None:
        return wavelength, tau
    diff = np.diff(wavelength)
    if np.sum(np.abs((diff - diff[0]))>0.01)>1:
        raise ValueError('Сетка по длине волны должна быть однородная')
    delta_wv = diff[0]
    wv_psf = np.arange(-5000*delta_wv,5001*delta_wv, delta_wv)
    psf = gaussian(wv_psf, mu=0, std_dev=np.mean(wavelength)/r, amplitude=1)
    t = np.convolve(tau, psf, mode='valid')/np.sum(psf)
    wvl_valid = np.convolve(wavelength, psf, mode='valid')/np.sum(psf)
    return wvl_valid, t

def convolve(wavelength, tau, r=1000):
    diff = np.diff(wavelength)
    if np.sum(np.abs((diff - diff[0]))>0.01)>1:
        raise ValueError('Сетка по длине волны должна быть однородная')
    delta_wv = diff[0]
    wv_psf = np.arange(-5000*delta_wv,5001*delta_wv, delta_wv)
    psf = gaussian(wv_psf, mu=0, std_dev=np.mean(wavelength)/r, amplitude=1)
    t = np.convolve(tau, psf, mode='valid')/np.sum(psf)
    wvl_valid = np.convolve(wavelength, psf, mode='valid')/np.sum(psf)
    return wvl_valid, t
def transmission(location, CO2=0.96, H2O=1000*10**(-6), CH4=0.0 , CO=0.0, HDO=0.0, HCl=0.0, O2=0.0, O3=0.0, r=1000):
    if location not in ['Mars2020', 'Zhurong', 'Curiosity']:
        raise ValueError('Проверьте чтовы вы правильно указали локацию. Возможные локации: Mars2020, Zhurong и Curiosity.')
    mixing_ratios = {'CO2': CO2, 'H2O': H2O, 'CH4': CH4, 'CO':CO, 'HDO':HDO, 'HCl':HCl, 'O2':O2, 'O3':O3}
    counter = 0  
    ts = []
    for gas in mixing_ratios.keys():
        if mixing_ratios[gas] != 0.0:
            counter += 1
            wvl, k = np.loadtxt('data/'+location+'/absorption-coefficients/'+gas+'.txt', unpack=True)
            wvl_valid, t_tmp = beerlambert(wvl, k, mixing_ratios[gas], l=100, r=None)
            ts.append(t_tmp)

    t = ts[0]
    for i in range(1,counter):
        t = t*ts[i]

    diff = np.diff(wvl_valid)
    if np.sum(np.abs((diff - diff[0]))>0.01)>1:
        raise ValueError('Сетка по длине волны должна быть однородная')
    delta_wv = diff[0]
    wv_psf = np.arange(-5000*delta_wv,5001*delta_wv, delta_wv)
    psf = gaussian(wv_psf, mu=0, std_dev=np.mean(wvl_valid)/r, amplitude=1)
    t = np.convolve(t, psf, mode='valid')/np.sum(psf)
    wvl_valid = np.convolve(wvl_valid, psf, mode='valid')/np.sum(psf)

    return wvl_valid, t

def solar_xray(days=30):
    amp = 1.25
    k = 1
    theta = 2.25
    time = np.arange(0, days*24*60, 1)
    events = np.zeros_like(time).astype(float)
    time_events = []
    lat = []
    longtitude = []
    amplitude = []
    for i in range(len(time)):
        if np.random.randn() > 2.35:
            p = amp**np.random.gamma(k, theta) + 1e-100
            event = p* ( (1+np.tanh((time-time[i])/(2*p))/2)**0.5) / (np.cosh(-((time-time[i])/(4*p))**2))
            events += event
            if np.max(event)>15:
                time_events.append(time[i]/24/60)
                lat.append(np.random.rand()*100-50)
                longtitude.append(np.random.rand()*180-90)
                amplitude.append(np.max(event))
            # events += gaussian(time, time[i], 20, np.random.gamma(0.5, 1))
    
    return time/24/60, events, np.array(time_events)/24/60, np.array(lat), np.array(longtitude), np.array(amplitude)

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
    
def solar_flares(lat, lon, amplitude, scale=2):

    scale = np.arange(0,scale,0.05*scale)
    lat = lat/180*np.pi
    lon = lon/180*np.pi

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(projection='3d')
    r = 1
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    
    my_col = cm.gray(np.ones_like(z)*0.98)

    ax.plot_surface(x, y, z, cmap=None, facecolors=my_col, alpha=0.12, shade=0.02)

    ax.view_init(0, 0)
    plt.gca().set_axis_off()
    set_axes_equal(ax)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    for i in range(len(lat)):
        theta = np.pi/2*np.tanh(amplitude[i]/50)
        
        # print(theta)
        
        xx = np.cos(lon[i]) * np.cos(lat[i])
        yy = np.sin(lon[i]) * np.cos(lat[i])
        zz = np.sin(lat[i])

        n1 = np.cross([xx,yy,zz], [0,0,1])
        n1 = n1/np.dot(n1,n1)**0.5
    #     print(n1.shape)
        
        n2 = np.cross(n1, [xx,yy,zz])
        n2 = n2/np.dot(n2,n2)**0.5
    #     print(n2.shape)
        n3 = np.zeros([3, 64])
        n3[:, :63] = n1[:,None]@np.cos(np.arange(0, 2*np.pi, 0.1))[None, :] + n2[:,None]@np.sin(np.arange(0, 2*np.pi, 0.1))[None, :]
        n3[:, 63] = n3[:, 0]
        #     n3 = n3/np.dot(n3,n3)**0.5
        n3 = n3.T
        n3 = n3/np.sqrt(np.sum(n3**2))
        
        # print(n3.shape)
        n3 = n3*np.tan(theta)
        # scale = np.arange(0, 2, 0.1)
        for j in range(len(scale)):
            sc_param = scale[j]
            ax.plot(sc_param*n3[:,0]+(sc_param+1)*xx, sc_param*n3[:,1]+(sc_param+1)*yy, sc_param*n3[:,2]+(sc_param+1)*zz, 
                    color='C'+str(i), alpha=0.7)

        
    plt.show()
