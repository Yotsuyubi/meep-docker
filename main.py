import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

C = 299_792_458

resolution = 1/5 # pixels/um

sx = 270 # um
sy = 270 # um
sz = 1500 # um
cell = mp.Vector3(sx,sy,sz)

dpml = 120 # um
pml_layers = [mp.PML(dpml, direction=mp.Z)]

w = 10 # width of object

fcen = 1/((C/0.7e12)*1e6)  # pulse center frequency
df = 1/((C/1e12)*1e6)    # pulse width (in frequency)
sources = [
    mp.Source(mp.GaussianSource(fcen,fwidth=df,is_integrated=True),
                     component=mp.Ex,
                     center=mp.Vector3(0,0,-200),
                     size=mp.Vector3(x=sx, y=sy)),
]

time = 100e3
nfreq = 300  # number of frequencies at which to compute flux


def load_mask(path, size):
    im = Image.open(path)
    im = im.resize((size[0], size[1]))
    im = im.convert('L')
    array = np.array(im) / 255
    array = (array < 1)
    array = np.abs(array - 1)
    array = np.expand_dims(array, -1)
    array = np.concatenate([array for _ in range(size[2])], -1)
    return array


def get_trans_ref():
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=[],
                        sources=sources,
                        resolution=resolution,
                        k_point=mp.Vector3())


    tran_fr = mp.FluxRegion(center=mp.Vector3(0,0,400), direction=mp.Y)
    tran = sim.add_flux(fcen,df,nfreq,tran_fr)

    pt = mp.Vector3(0,0,400)
    # sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ex,pt,1e-3))
    sim.run(until=time)

    straight_tran_flux = sim.get_flux_data(tran)

    return straight_tran_flux

def get_refl_ref():
    geometry = [
        # mp.Block(
        # size=mp.Vector3(mp.inf, mp.inf, w),
        # center=mp.Vector3(0,0,0),
        # material=mp.perfect_electric_conductor),
    ]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        k_point=mp.Vector3())


    refl_fr = mp.FluxRegion(center=mp.Vector3(0,0,-400), direction=mp.Y, weight=-1) 
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    pt = mp.Vector3(0,0,-400)
    # sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ex,pt,1e-3))
    sim.run(until=time)

    straight_refl_data = sim.get_flux_data(refl)

    return straight_refl_data


def get_signal(straight_refl_data):
    eps = [
        mp.Vector3(1.69**2, 1.54**2, 1.54**2), # 0 deg
        mp.Vector3(1.54**2, 1.69**2, 1.54**2), # 90 deg
    ]

    mask = np.ones((sx, sy, w), dtype=np.int8)
    mask[60:-60,60:-60,:] = 0
    # mask = load_mask('0.png', (sx, sy, w)) # define metal-mesh pattern from image.

    masks = [
        mask, # 0 deg
        np.rot90(mask, axes=(0, 1), k=-1) # 90 deg
    ]

    deg = [0, 90]
    data = []

    for i in range(len(eps)):
        e = eps[i]
        mask = masks[i]
        material = mp.MaterialGrid(
            grid_size=mp.Vector3(sx, sy, w),
            medium1=mp.Medium(epsilon_diag=e),
            medium2=mp.perfect_electric_conductor,
            weights=mask
        )
        geometry = [
            mp.Block(
                size=mp.Vector3(sx, sy, w),
                center=mp.Vector3(0,0,0),
                material=material
            ),
        ]

        sim = mp.Simulation(cell_size=cell,
                            boundary_layers=pml_layers,
                            geometry=geometry,
                            sources=sources,
                            resolution=resolution,
                            k_point=mp.Vector3())


        tran_fr = mp.FluxRegion(center=mp.Vector3(0,0,400), direction=mp.Y)
        refl_fr = mp.FluxRegion(center=mp.Vector3(0,0,-400), direction=mp.Y, weight=-1) 
        nfreq = 300  # number of frequencies at which to compute flux
        refl = sim.add_flux(fcen, df, nfreq, refl_fr)
        tran = sim.add_flux(fcen, df, nfreq, tran_fr)

        sim.load_minus_flux_data(refl, straight_refl_data)

        pt = mp.Vector3(0,0,-400)
        # sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ex,pt,1e-3))
        sim.run(until=time)

        tran_flux = sim.get_flux_data(tran)
        refl_flux = sim.get_flux_data(refl)

        data.append([tran_flux, refl_flux])

        sim.plot2D(fields=mp.Ex, output_plane=mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(sx, 0, sz)))
        plt.savefig('Ex_xz_{}.png'.format(deg[i]))

        sim.plot2D(fields=mp.Ex, output_plane=mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(0, sy, sz)))
        plt.savefig('Ex_yz_{}.png'.format(deg[i]))

        sim.plot2D(fields=mp.Ex, output_plane=mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(sx, sy, 0)))
        plt.savefig('Ex_xy_{}.png'.format(deg[i]))

    return data, mp.get_flux_freqs(tran)


tran_ref = get_trans_ref()
refl_ref = get_refl_ref()

data, freq = get_signal(refl_ref)

f = []
Rs = []
Ts = []
for i in range(nfreq):
    f.append(C/(1/freq[i]*1e-6))
    Ts.append([
        data[0][0].E[i]/tran_ref.E[i],
        data[1][0].E[i]/tran_ref.E[i],
    ])
    Rs.append([
        data[0][1].E[i]/refl_ref.E[i],
        data[1][1].E[i]/refl_ref.E[i],
    ])

trans_ref = np.array(tran_ref.E).T
refl_ref = np.array(refl_ref.E).T
trans_sig = np.array(data[0][0].E).T
refl_sig = np.array(data[0][1].E).T

Rs = np.array(Rs).T
Ts = np.array(Ts).T
f = np.array(f)

if mp.am_master():
    plt.figure()
    plt.plot(f*1e-12,np.abs(Ts[0,:])**2,'r-',label='trans0')
    plt.plot(f*1e-12,np.abs(Rs[0,:])**2,'b-',label='refl0')
    plt.plot(f*1e-12,np.abs(Ts[1,:])**2,'r--',label='trans90')
    plt.plot(f*1e-12,np.abs(Rs[1,:])**2,'b--',label='refl90')
    # plt.plot(f*1e-12,np.abs(Ts[0,:])**2 + np.abs(Rs[0,:])**2,'g-',label='loss0')
    # plt.plot(f*1e-12,np.abs(Ts[1,:])**2 + np.abs(Rs[1,:])**2,'g--',label='loss90')
    plt.ylim(0,1)
    plt.xlim(0.5, 1)
    plt.xlabel("Freq (THz)")
    plt.ylabel("Transmittance/Reflectance (-)")
    plt.legend()
    plt.savefig('TsRs_amp.png')
    plt.close()

if mp.am_master():
    plt.figure()
    plt.plot(f*1e-12,np.unwrap(np.angle(Ts[0,:])),'r-',label='t0')
    plt.plot(f*1e-12,np.unwrap(np.angle(Rs[0,:])),'b-',label='r0')
    plt.plot(f*1e-12,np.unwrap(np.angle(Ts[1,:])),'r--',label='t90')
    plt.plot(f*1e-12,np.unwrap(np.angle(Rs[1,:])),'b--',label='r90')
    # plt.ylim(0,1)
    plt.xlim(0.5, 1)
    plt.xlabel("Freq (THz)")
    plt.ylabel("Phase shift (rad)")
    plt.legend()
    plt.savefig('TsRs_phase.png')
    plt.close()

wl = (C/np.array(f))
k = 2*np.pi / wl
Ne = 1/(k*w*1e-6)*np.arccos( (1 - Rs[0]**2 + Ts[0]**2) / 2*Ts[0] )
No = 1/(k*w*1e-6)*np.arccos( (1 - Rs[1]**2 + Ts[1]**2) / 2*Ts[1] )
N = Ne.real - No.real
if mp.am_master():
    plt.figure()
    plt.plot(f*1e-12,N,'b-')
    # plt.ylim(-5,5)
    plt.xlim(0.5, 1)
    plt.xlabel("Freq (THz)")
    plt.ylabel("$\Delta n$ (-)")
    plt.legend()
    plt.savefig('N_delta.png')
    plt.close()
print(np.max(N))