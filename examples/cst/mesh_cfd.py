'''
generate mesh with the CRM geometry but the surface mesh for database simulation

- this results first use pointwise to get surface mesh with correct direction and format from lyu-3m mesh
- difference from reconstruct:
    - use real dihedral values at sections + spline
    - twist modified according to the xRef=0.25

'''
import os, sys, random, time
import numpy as np
import json
# from pyhyp import pyHyp
# from cgnsutilities.cgnsutilities import readGrid
from cst_modeling.basic import rotation_3d
from cst_modeling.io import read_plot3d, output_plot3d_concat
from cst_modeling.foil import cst_foil_fit, cst_foil
import copy
from mpi4py import MPI
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from functools import reduce

# comm = MPI.COMM_WORLD

class SurfaceMeshingParams:

    def __init__(self, tip_path: str, twists0: float = None, mesh_point_numbers: dict = None, xx0: list = None, control_points: list = None):
        self.twists0 = twists0 if twists0 is not None else 6.7166
        self.mesh_point_numbers = mesh_point_numbers if mesh_point_numbers is not None else {
            'nFoil':    117,
            'nX':       113,
            'nY':       [41, 125],
            'nZ':       9,
            'nFar':     81
        }
        self.base_tip_blocks = read_plot3d(tip_path, verbose=0) 
        self.xx0 = xx0 if xx0 is not None else [
            0., 5.05256963e-04, 1.29435990e-03, 2.30710683e-03, 3.47136074e-03, 4.61166766e-03, 5.96008856e-03,
            7.55045344e-03, 9.42059830e-03, 1.16133031e-02, 1.41777060e-02, 1.71692487e-02, 2.06514955e-02, 2.46963502e-02, 2.93869138e-02,
            3.48180094e-02, 4.10985870e-02, 4.83540454e-02, 5.67280608e-02, 6.63881031e-02, 7.69451593e-02, 8.75207785e-02, 9.81108987e-02,
            1.08712986e-01, 1.19325489e-01, 1.29947375e-01, 1.40577934e-01, 1.51216707e-01, 1.61863310e-01, 1.72517384e-01, 1.83178632e-01,
            1.93846825e-01, 2.04521670e-01, 2.15202919e-01, 2.25890219e-01, 2.36583338e-01, 2.47281970e-01, 2.57985787e-01, 2.68694449e-01,
            2.79407717e-01, 2.90125279e-01, 3.00846740e-01, 3.11571696e-01, 3.22299795e-01, 3.33030723e-01, 3.43764039e-01, 3.54499277e-01,
            3.65235991e-01, 3.75973806e-01, 3.86712274e-01, 3.97450937e-01, 4.08189379e-01, 4.18927144e-01, 4.29663868e-01, 4.40399177e-01,
            4.51132661e-01, 4.61863901e-01, 4.72592596e-01, 4.83318512e-01, 4.94041363e-01, 5.04760823e-01, 5.15383453e-01, 5.26002270e-01,
            5.36617182e-01, 5.47228029e-01, 5.57834613e-01, 5.68436802e-01, 5.79034560e-01, 5.89627936e-01, 6.00216864e-01, 6.10801313e-01,
            6.21381327e-01, 6.31956985e-01, 6.42528420e-01, 6.53095762e-01, 6.63659148e-01, 6.74218723e-01, 6.84774729e-01, 6.95327470e-01,
            7.05877214e-01, 7.16424300e-01, 7.26969057e-01, 7.37511925e-01, 7.48053471e-01, 7.58594170e-01, 7.69134556e-01, 7.79675279e-01,
            7.90217107e-01, 8.00760711e-01, 8.11306563e-01, 8.21855670e-01, 8.32408873e-01, 8.42966811e-01, 8.53529992e-01, 8.64099098e-01,
            8.74674556e-01, 8.85256582e-01, 8.95845260e-01, 9.06440753e-01, 9.17042790e-01, 9.27651068e-01, 9.38265455e-01, 9.48885481e-01,
            9.59569457e-01, 9.68120400e-01, 9.74963459e-01, 9.80439342e-01, 9.84820908e-01, 9.88326688e-01, 9.91131624e-01, 9.93375756e-01,
            9.95171174e-01, 9.96607568e-01, 9.97756716e-01, 9.98676054e-01, 9.99411534e-01, 1.
        ]
        self.control_points = control_points if control_points is not None else [0.09999976658796647, 0.23391927215290928, 0.36783877771785206, 
                    0.5258790832883891, 0.683919388858926, 0.841959694429463, 1.0]

        self.mesh_generation_index = 0

log_file = ''

def log(*string):
    if log_file is not None:
        if log_file == '':
            print(*string)

def linear_smooth(block1, block2):
    
    nsm = 3
    # print(block1.shape, block2.shape)
    new_block = np.linspace(block1[:, -nsm], block2[:, nsm-1], 2*nsm-1, axis=1)
    block1[:, -nsm:] = new_block[:, :nsm]
    block2[:, :nsm] = new_block[:, -nsm:]




def scaling_tip(xx1, zz1, wingCoef, generator_config: SurfaceMeshingParams, rdx=None, rdz=None, tip_twist=None):
    '''
    Linear interpolate for new tip airfoil
    
    
    '''
    blocks = copy.deepcopy(generator_config.base_tip_blocks)
    # tip_lower_new, tip_front_new, tip_upper_new, tip_back_new, tip_side
    
    nn0 = len(blocks[0]) + int(len(blocks[1]) / 2)
    nn1 = len(blocks[0])
    nn2 = len(blocks[1])
    
    assert generator_config.mesh_point_numbers['nFoil'] == nn0, 'Foil direction %d != %d' % (generator_config.mesh_point_numbers['nFoil'], nn0) 
    assert generator_config.mesh_point_numbers['nX'] == nn1, 'X direction %d != %d' % (generator_config.mesh_point_numbers['nX'], nn1)
    assert generator_config.mesh_point_numbers['nZ'] == nn2, 'Y direction %d != %d' % (generator_config.mesh_point_numbers['nZ'], nn2)
    
    # target airfoil
    # xu1, yu1, yl1, _, _ = cst_foil(nn=nn0, cst_u=cst_u_tip, cst_l=cst_l_tip, t=t_tip, a1=0.985)
    # xx1 = np.concatenate((np.flip(xu1[1:]), xu1, np.ones((nn2-1,))), axis=0)
    yy1 = np.zeros_like(xx1)
    # zz1 = np.concatenate((np.flip(yl1[1:]), yu1, np.linspace(yu1[-1], yl1[-1], nn2)[1:]), axis=0)
    
    # modify endwall block position based on sweep and dihedral angle 
    tail_avg = np.mean(blocks[4][:, -1, 0], axis=0)
    tip_tail_avg = np.mean(blocks[3][:, 0, 0], axis=0)
    # print(tail_avg, tip_tail_avg)
    dx0, dy, dz0 = tail_avg - tip_tail_avg
    if rdx is None:
        dx = (np.tan(min(35,0.8*wingCoef['SA'])/180*np.pi) + (wingCoef['TR'] - 1) / wingCoef['half_span']) * dy - (tail_avg[0] - 1)
    else:
        dx = rdx * dy * 0.5

    if rdz is None:
        dz = -np.tan(wingCoef['DA']/180*np.pi) * dy * 0.5
    else:
        dz = rdz * dy * np.linspace(1.5, 0.8, blocks[4].shape[1])

    # log(np.arctan([rdx, rdz])/np.pi*180)
    blocks[4][:, :, :, 0] += (dx - dx0)
    blocks[4][:, :, :, 2] += (dz - dz0).reshape(1, -1, 1)
    
    if tip_twist is not None:
        shape = blocks[4].shape
        blocks[4] = rotation_3d(blocks[4].reshape(-1, 3), np.array([0, 0, 0]), np.array([0, 1, 0]), tip_twist - 10.46).reshape(*shape)
    
    # modify endwall block thickness
    iLE = int(len(zz1) / 2)
    tt1 = zz1[iLE:-8] - np.flip(zz1[:iLE+1-8])
    oritt = blocks[2][:, 0, 0, 2] - np.flip(blocks[0][:, 0, 0, 2])
    scale_r = tt1 / oritt 
    b4mid   = 0.5 * (blocks[4][-1, :, 0, 2] + blocks[4][0, :, 0, 2])
    b4tt     = scale_r * (blocks[4][-1, :, 0, 2] - blocks[4][0, :, 0, 2])

    blocks[4][:, :, 0, 2] = np.linspace(b4mid - 0.5*b4tt*np.linspace(0.8, 1.4, blocks[4].shape[1]), b4mid + 0.5*b4tt, blocks[4].shape[0])
    
    # low_r = min(zz1) / min(blocks[0][:, 0, 0, 2])
    # upp_r = max(zz1) / max(blocks[2][:, 0, 0, 2])
    
    ## modify circum blocks to blend
    
    surface_blocks1 = np.concatenate((blocks[0][:-1], blocks[1][:-1], blocks[2][:-1], blocks[3]), axis=0)   # circum blocks of tip
    tip_airfoils1 = surface_blocks1[:, 0, 0]   # airfoil at wing side
    tip_airfoils2 = surface_blocks1[:, -1, 0]   # airfoil at tip side
    side_block_airfoil = np.concatenate((np.flip(blocks[4][0, 1:, 0], axis=0), blocks[4][:-1, 0, 0], blocks[4][-1, :-1, 0], np.flip(blocks[4][:, -1, 0], axis=0)), axis=0)  # endwall block of tip

    diff1 = np.array([xx1, yy1, zz1]).transpose(1, 0) - tip_airfoils1
    diff2 = side_block_airfoil - tip_airfoils2
    
    for block, st, ed in zip(blocks[:-1], [None, nn1-1, nn1+nn2-2, -nn2], [nn1, nn1+nn2-1, -nn2+1, None]):
        block[:, :, 0, :] += (np.linspace(1, 0, block.shape[1])[None, :, None] * diff1[st:ed, None, :] +\
            np.linspace(0, 1, block.shape[1])[None, :, None] * diff2[st:ed, None, :])

    return blocks

def move_block(block, translate, scale, angle, axis):
    
    block_shape = block.shape
    new_block = rotation_3d(block.reshape(-1, 3), origin=np.array([0, 0, 0]), axis=axis, angle=angle)
    # new_block[:, 1] = 0.5 * new_block[:, 1]
    new_block = new_block * scale + translate[None, :]  
    
    return new_block.reshape(*block_shape)

def power_growth0(dx0, L, n):
    '''
    integer ,intent(in ):: n
    real*8  ,intent(in ):: dx0, L
    real*8  ,intent(out):: dxs(n)
    real*8  :: f, a, a1, aa
    integer :: i, is
    '''
    dxs = np.zeros((n,))

    f = L / dx0
    a = 1.2
    a1= 1.0
    
    while abs((a-a1)/a) > 2e-7:
        a1= a
        a =(a*f-f+1.)**(1./n)
    aa = 0.

    for i in range(n):
        dxs[i] = aa+a**i*dx0
        aa = dxs[i]

    return dxs

def power_growth(dx0, dx1, L, n):
    '''

    '''
    k0 = 0
    k = int(n / 2)

    while abs(k0 - k) > 0.5:
        k0 = k
        r = (dx0 / dx1)**(1 / (k-2))
        k = int(n + 1 - (L - dx1 * (1 - r**(k0-1)) / (1 - r)) / dx0)

    yLEs1 = np.concatenate((np.arange(0, n - k) * dx0, np.flip(L - dx1 * (1 - r**np.arange(0, k)) / (1 - r))), axis=0)
    yLEs1[n - k - 3: n - k + 3] = np.linspace(yLEs1[n - k - 3], yLEs1[n - k + 2], 6)    # smoothing
    
    return yLEs1

def surface_meshing(wingCoefs: dict, generator_config: SurfaceMeshingParams = None, config: dict = None, save_surface: bool = False, output_folder: str = 'output'):

    # print('Generate surface mesh for CFD with given parameters')
    
    #* surface meshing
    nn0 = generator_config.mesh_point_numbers['nFoil']
    nn1 = generator_config.mesh_point_numbers['nX']
    nn2 = generator_config.mesh_point_numbers['nZ']
    nny = generator_config.mesh_point_numbers['nY']
    
    # get distributed values
    cabin_ratio = 0.1
    half_span = wingCoefs['half_span'] 
    kink = half_span * generator_config.control_points[2]
    cabin = half_span * cabin_ratio
    
    _control_points = [half_span * eta for eta in generator_config.control_points]
    
    # get spanwise eta distribution
    # inner section
    yLEs0 = np.linspace(cabin, kink, nny[0])
    # outer section (linear) -> 0.0006
    yLEs1 = power_growth(dx0=yLEs0[-1] - yLEs0[-2], dx1=0.002*0.275, L=half_span - kink, n=nny[1]) + kink
    yLEs = np.concatenate((yLEs0, yLEs1), axis=0)   # Caution! There is a relunctate point when concatenate, 
                                                    # this is dealed when seperating it to two blocks at writing out
    
    # import matplotlib.pyplot as plt
    # plt.plot(range(nny[0]), yLEs0, '-o', c='r')
    # plt.plot(range(nny[1]), yLEs1, '-o')
    # plt.show()
    
    # leading edge parameters    
    xLEs = np.tan(wingCoefs['SA']/180*np.pi) * yLEs     # sweep direction
    
    # chord
    chord_cabin, chord_kink, chord_tip = wingCoefs["chords"]

    chords0 = chord_cabin + (chord_kink - chord_cabin) * (yLEs0 - cabin) / (kink - cabin)
    chords1 = chord_kink + (chord_tip  - chord_kink) * (yLEs1 - kink) / (half_span - kink)
    chords  = np.concatenate((chords0, chords1), axis=0)
    
    # twist
    twists = [wingCoefs['root_twist']] + wingCoefs['twists'].tolist()
    tws_cs = CubicSpline(_control_points, [reduce(lambda x, y: x + y, twists[:i+1]) + generator_config.twists0 for i in range(len(wingCoefs['twists'])+1)], extrapolate=True)
    tws = tws_cs(yLEs)
    # dihedral
    zLE_cs = CubicSpline([0] + _control_points, [0] + [reduce(lambda x, y: x + y, wingCoefs['DAs'][:i+1]) for i in range(len(wingCoefs['DAs']))], extrapolate=False)
    zLEs = zLE_cs(yLEs) + 0.25 * chords * np.sin(tws/180*np.pi)

    wingCoefs['surface_area'] = (chord_cabin + chord_kink) * (kink - cabin) * 0.5 + (chord_kink + chord_tip) * (half_span - kink) * 0.5
        
    # reconstruct airfoils with CSTs and store them in zzss
    zzss = []
    cst_u = np.array(wingCoefs['cst_u']).reshape(len(_control_points), -1)
    cst_l = np.array(wingCoefs['cst_l']).reshape(len(_control_points), -1)

    if cst_u.shape[1] == cst_l.shape[1] + 1:
        cst_l = np.concatenate((-cst_u[:, [0]], cst_l), axis=1)
    for i in range(len(_control_points)):
        xx1, yus, yls, _, _ = cst_foil(nn=nn0, x=np.array(generator_config.xx0), cst_u=cst_u[i], cst_l=cst_l[i], t=None, tail=0.008)
        zzss.append(np.concatenate((np.flip(yls[1:], axis=0), yus[:-1], np.linspace(yus[-1], yls[-1], nn2)), axis=0))
    
    zzs = []
    for yy in yLEs:
        if yy < _control_points[0]:
            zzs.append(zzss[0] + (yy - _control_points[0]) / (_control_points[1] - _control_points[0]) * (zzss[1] - zzss[0]))
            log(f'[warning] extrapolate for y = {yy:.2f}')
        elif yy - _control_points[-1] > 0:
            raise RuntimeError()
        else:
            for i in range(len(_control_points) - 1):
                if yy >= _control_points[i] and yy <= _control_points[i+1]:
                    zzs.append(zzss[i] + (yy - _control_points[i]) / (_control_points[i+1] - _control_points[i]) * (zzss[i+1] - zzss[i]))
                    break
            else:
                raise RuntimeError
    zzs = np.array(zzs).transpose()
        
    # rotation
    xxs = np.concatenate((np.flip(xx1[1:], axis=0), xx1[:-1], np.ones((nn2,))), axis=0)
    # wrt 0.25 chord is considered in adding a dz to dihedrals
    tws_rat = tws / 180 * np.pi
    xxs1  = np.cos(tws_rat[None, :]) * xxs[:, None] +  np.sin(tws_rat[None, :]) * zzs
    zzs1 = -np.sin(tws_rat[None, :]) * xxs[:, None] +  np.cos(tws_rat[None, :]) * zzs
    # log(np.isnan(xxs1).any(), np.isnan(zzs1).any())

    xxs1 = xxs1 * chords[None, :] + xLEs[None, :]
    zzs1 = zzs1 * chords[None, :] + zLEs[None, :]
    # log(np.isnan(xxs1).any(), np.isnan(zzs1).any())

    # print(xxs.shape, np.repeat(yLEs[None, :], 2*nn0-2+nn2, axis=0).shape, zzs.shape)
    block = np.stack((xxs1, np.repeat(yLEs[None, :], 2*nn0-2+nn2, axis=0), zzs1)).transpose(1, 2, 0)[:, :, None, :]
    
    nny2 = nny[1] + 1
    output_blocks = [block[:nn1, :nny[0]], block[nn1-1:nn1+nn2-1, :nny[0]], block[nn1+nn2-2: -nn2+1, :nny[0]], block[-nn2:, :nny[0]],
                     block[:nn1, nny[0]:nny[0]+nny2], block[nn1-1:nn1+nn2-1, nny[0]:nny[0]+nny2], block[nn1+nn2-2: -nn2+1, nny[0]:nny[0]+nny2], block[-nn2:, nny[0]:nny[0]+nny2],
                    ] 

    # print(np.mean(output_blocks[7][:, -2, 0], axis=0), np.mean(output_blocks[7][:, -1, 0], axis=0))
    tip_tail_dr = np.mean(output_blocks[7][:, -2, 0] - output_blocks[7][:, -1, 0], axis=0)
    tip_blocks = scaling_tip(xxs, zzs[:, -1], wingCoefs, generator_config, rdx=tip_tail_dr[0]/tip_tail_dr[1],  rdz=tip_tail_dr[2]/tip_tail_dr[1])
    ntip_blocks = [move_block(blk, translate=np.array([xLEs[-1], yLEs[-1], zLEs[-1]]), scale=chords[-1], angle=tws[-1], axis=np.array([0, 1, 0])) for blk in tip_blocks]


    # rotate block (airfoil at x-z to airfoil at x-y)
    output_blocks = [block[:, :, :, [0, 2, 1]].transpose(1, 0, 2, 3) for block in output_blocks + ntip_blocks]

    
    for i in range(len(output_blocks)):

        if np.isnan(output_blocks[i]).any():
            log(f'nan in block {i}')

    if save_surface:
        output_dict = {}
        for k in wingCoefs.keys():
            if isinstance(wingCoefs[k], np.ndarray):
                output_dict[k] = wingCoefs[k].tolist()
            else:
                output_dict[k] = wingCoefs[k]
        with open(os.path.join(output_folder, f'input_{generator_config.mesh_generation_index:d}.json'), 'w') as f:
            json.dump(output_dict, f)
        output_plot3d_concat(output_blocks, fname=os.path.join(output_folder, f'wing_cfd_{generator_config.mesh_generation_index:d}.xyz'), order='ij')
        generator_config.mesh_generation_index += 1
    return np.concatenate([block.reshape(-1, 3) for block in output_blocks], axis=0)


    