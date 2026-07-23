'''
use FFD for deformation, use ML model for prediction

'''


# ======================================================================
#         Import modules
# ======================================================================
# rst Imports (beg)
import os
import json
import argparse
import ast
from mpi4py import MPI
from baseclasses import AeroProblem
from adflow import ADFLOW
from pygeo import DVConstraints, DVGeometry
from pyoptsparse import Optimization, OPT
from idwarp import USMesh
from multipoint import multiPointSparse
import numpy as np

from pyoptsparse_ml.mlsolver import MLSolver
from pyoptsparse_ml.combine import SolverCombined
from flowvae.app.wing.api import SuperWingAPI, surface_meshing_ML, SurfaceMeshingParams_ML

# Optional ADflow verification handles; populated when CFD sampling is enabled
CFDSolver = None
_cfd_mesh = None

source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

# rst Imports (end)
# rst args (beg)
# Use Python's built-in Argument parser to get commandline options
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default="output")
parser.add_argument("--opt", type=str, default="IPOPT", choices=["SLSQP", "IPOPT", "SNOPT"])
parser.add_argument("--optOptions", type=ast.literal_eval, default={}, help="additional optimizer options to be added")
parser.add_argument("-i", "--cfdFrequency", type=int, default=0, help="In Ni+Nj iterations, not call CFD at first Ni iterations (0 disables).")
parser.add_argument("-j", "--cfdIter", type=int, default=1, help="In Ni+Nj iterations, call ADflow every Nj iterations (0 stands for switch to CFD after Ni iterations).")
parser.add_argument("-s", "--cfdInclude", type=int, default=0, choices=list(SolverCombined.include_modes.keys()), 
                    help="CFD + ML mode. Available modes: " + ", ".join([f"{k}: {v}" for k, v in SolverCombined.include_modes.items()]))
parser.add_argument("-d", "--mlDevice", type=str, default='cuda:0', help="Device for CUDA")
parser.add_argument("--cfdOptions", type=ast.literal_eval, default={}, help="additional ADflow options to be merged with defaults when CFD sampling is enabled")
parser.add_argument("--mutecfdsave", action="store_false", help="Store the ADflow volume solution whenever a CFD run is performed.")
parser.add_argument("--mlModel", type=str, default='finetune20', choices=SuperWingAPI.version_folder.keys(), help="ML model to use")
parser.add_argument("--fd", action="store_true", help="use FD to get gradient of ML model, otherwise use BP.")
parser.add_argument("--printlevel", type=int, default=0, help="print level for ML model, 0: no print, 1: print debug.")
args = parser.parse_args()
# rst args (end)
# ======================================================================
#         Create multipoint communication object
# ======================================================================
# rst multipoint (beg)
MP = multiPointSparse(MPI.COMM_WORLD)
MP.addProcessorSet("cruise", nMembers=1, memberSizes=MPI.COMM_WORLD.size)
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()

if not os.path.exists(args.output):
    if comm.rank == 0:
        os.mkdir(args.output)

# rst multipoint (end)
# ======================================================================
#         Set up flow conditions with AeroProblem
# ======================================================================
# rst aeroproblem (beg)
scale = 1.408244549835913
ap = AeroProblem(name="wing", alpha=1.5, mach=0.85, reynolds=20000000.0, reynoldsLength=1., T=300, areaRef=3.407014/(scale**2), 
                 chordRef=1./scale, evalFuncs=["cl", "cd", "cmz"], xRef=1.2077/scale+0.21792, yRef=0.007669/scale+0.00408, zRef=0.)

# Add angle of attack variable
ap.addDV("alpha", value=1.5, lower=0, upper=8.0, scale=0.1)

# rst aeroproblem (end)
# ======================================================================
#         ML model Set-up
# ======================================================================
# rst mlmodel (beg)
mlSolver = MLSolver(output_keys=['cl', 'cd', 'cmz'],
                    condition_keys={
                        'alpha': [0, 5],
                        'mach': [0.75, 0.90],
                        'reynolds': 20000000,
                    },
                    options={
                        'output_dir': args.output,
                        'write_surface_tecplot': False, 
                        "sens_mode": "FD" if args.fd else "BP",
                        "fd_step": 1e-5, 
                    },
                    device=args.mlDevice,
                    comm=comm)

mlSolver.setModel(SuperWingAPI(model_version=args.mlModel, device=args.mlDevice) if comm.rank == 0 else None, ap=ap)

# rst mlmodel (end)


# ======================================================================
#         Geometric Design Variable Set-up
# ======================================================================
# rst dvgeo (beg)

# Create DVGeometry object (FFD-based)
FFDFile = os.path.join(source_dir, "ffd.xyz")
DVGeo = DVGeometry(FFDFile)

# Create reference axis
nRefAxPts = DVGeo.addRefAxis("wing", xFraction=0.25, alignIndex="k")
nTwist = nRefAxPts - 1

# Set up global design variables
def twist(val, geo):
    for i in range(1, nRefAxPts):
        geo.rot_z["wing"].coef[i] = val[i - 1]

DVGeo.addGlobalDV(dvName="twist", value=[0] * nTwist, func=twist, lower=-10, upper=10, scale=0.01)

# Set up local design variables
DVGeo.addLocalDV("local", lower=-0.05, upper=0.05, axis="y", scale=1)

# Baseline surface mesh for ML model input
with open(os.path.join(source_dir, "input.json"), "r", encoding="ascii") as f:
    wing_coefs = json.load(f)

mlSolver.setSurfaceMesh(surface_meshing_ML(wing_coefs, generator_config=SurfaceMeshingParams_ML(), save_surface=False))
mlSolver.setDVGeo(DVGeo, useBaseline=-1)

# rst dvgeo (end)

# ======================================================================
#         Optional ADflow CFD Verification Set-up
# ======================================================================
# rst adflow (beg)
cfd_frequency = max(0, args.cfdFrequency)

if cfd_frequency > 0:

    cfd_options = {
        # I/O Parameters
        "gridFile": os.path.join(source_dir, "wing_vol.cgns"),
        "outputDirectory": args.output,
        "monitorvariables": ["resrho", "cl", "cd", "cmz"],
        "surfaceVariables": ['cp', 'rho', 'temp', 'ptloss', 'vx', 'vy', 'vz', 'mach', 'cfx', 'cfy', 'cfz', 'ch', 'yplus'],
        "writeTecplotSurfaceSolution": False,
    #     "restartFile": "output/wing_000_vol.cgns",
        # Physics Parameters
        "equationType": "RANS",
        # Solver Parameters
        "smoother": "DADI",
        "MGCycle": "sg",
        "infchangecorrection": True,
        # ANK Solver Parameters
        "useANKSolver": True,
        # NK Solver Parameters
        "useNKSolver": True,
        "nkswitchtol": 1e-6,
        # Termination Criteria
        "L2Convergence": 1e-8,
        "L2ConvergenceCoarse": 1e-2,
        "nCycles": 4000,
        # Adjoint Parameters
        "adjointL2Convergence": 1e-8,
        "adjointMaxIter": 1000,
    }

    if args.cfdOptions:
        cfd_options.update(args.cfdOptions)

    if args.mutecfdsave:
        cfd_options.setdefault("writevolumesolution", True)
        cfd_options.setdefault("writesurfacesolution", True)

    CFDSolver = ADFLOW(options=cfd_options, comm=comm)
    _cfd_mesh = USMesh(options={"gridFile": os.path.join(source_dir, "wing_vol.cgns")}, comm=comm)

    CFDSolver.setDVGeo(DVGeo)
    CFDSolver.setMesh(_cfd_mesh)

    CFDSolver.addLiftDistribution(150, "z")
    CFDSolver.addSlices("z", np.linspace(0.3, 2.9, 10))

# rst adflow (end)
# ======================================================================
#         DVConstraint Setup, and Thickness and Volume Constraints
# ======================================================================
# rst dvconVolThick (beg)
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)

if cfd_frequency > 0:
    DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface())
else:
    raise NotImplementedError()
    DVCon.setSurface(mlSolver.getTriangulatedMeshSurface())

# Surface definition is assigned after optional CFD setup
# Volume constraints
leList = [[0.25, 0.02, 0.32], [0.92, 0.11, 1.08819], [2.2, 0.46, 2.8]]
teList = [[1.25, 0.02, 0.32], [1.38, 0.13, 1.08819], [2.36, 0.48, 2.8]]
DVCon.addVolumeConstraint(leList, teList, nSpan=20, nChord=20, lower=1.0, scaled=True)

# Thickness constraints
DVCon.addThicknessConstraints2D(leList, teList, nSpan=10, nChord=10, lower=0.5, scaled=True)

if comm.rank == 0:
    # Only make one processor do this
    DVCon.writeTecplot(os.path.join(args.output, "constraints.dat"))
    DVCon.writeSurfaceTecplot(os.path.join(args.output, "trisurface.dat"))

# rst dvconVolThick (end)
# ======================================================================
#         Functions:
# ======================================================================
# rst funcs (beg)

def cruiseFuncs_pre(ap, x):

    # Set design vars
    DVGeo.setDesignVars(x)

def cruiseFuncs_ml(ap):

    # Run ML
    mlSolver(ap)

    # Evaluate functions
    funcs = {}
    DVCon.evalFunctions(funcs)
    mlSolver.evalFunctions(ap, funcs)
    mlSolver.checkSolutionFailure(ap, funcs)

    if comm.rank == 0 and args.printlevel > 0:  print('current funcs', funcs)

    return funcs

def cruiseFuncs_cfd(ap):
    cfd_funcs = {}
    CFDSolver(ap)
    CFDSolver.evalFunctions(ap, cfd_funcs)
    CFDSolver.checkSolutionFailure(ap, cfd_funcs)

    return cfd_funcs

def cruiseFuncsSens_ml(ap, x, funcs):

    funcsSens = {}
    DVCon.evalFunctionsSens(funcsSens)

    mlSolver.evalFunctionsSens(ap, funcsSens)
    mlSolver.checkAdjointFailure(ap, funcsSens)

    return funcsSens

def cruiseFuncsSens_cfd(ap, x, funcs):

    cfd_funcsSens = {}
    CFDSolver.evalFunctionsSens(ap, cfd_funcsSens)
    CFDSolver.checkAdjointFailure(ap, cfd_funcsSens)

    if comm.rank == 0: print("cfd_funcsSens keys:", list(cfd_funcsSens.keys()))

    return cfd_funcsSens

def objCon(funcs, printOK):
    # Assemble the objective and any additional constraints (ML only for opt.hst)
    funcs["obj"] = funcs[ap["cd"]]
    funcs["cl_con_" + ap.name] = funcs[ap["cl"]]
    funcs["cm_con_" + ap.name] = -funcs[ap["cmz"]]
    if printOK:
        print("funcs in obj:", funcs)
    return funcs

# rst funcs (end)
# ======================================================================
#         Optimization Problem Set-up
# ======================================================================
# rst optimizer
# Set up optimizer
if args.opt == "SLSQP":
    optOptions = {"IFILE": os.path.join(args.output, "SLSQP.out")}
elif args.opt == "SNOPT":
    optOptions = {
        "Major feasibility tolerance": 1e-4,
        "Major optimality tolerance": 1e-4,
        "Hessian full memory": None,
        "Function precision": 1e-8,
        "Print file": os.path.join(args.output, "SNOPT_print.out"),
        "Summary file": os.path.join(args.output, "SNOPT_summary.out"),
        "Major iterations limit": 1000,
    }
elif args.opt == "IPOPT":
    optOptions = {
        "output_file": os.path.join(args.output, "IPOPT.out"),
        "limited_memory_max_history": 200,
        "print_level": 5,
        "tol": 1e-6,
        "acceptable_tol": 1e-5,
        "max_iter": 300,
    }
optOptions.update(args.optOptions)
opt = OPT(args.opt, options=optOptions)
# rst optimizer (end)

# rst optprob (beg)
# Create optimization problem
optProb = Optimization("opt", MP.obj, comm=comm)

# Add objective
optProb.addObj("obj", scale=1e2)

# Add variables from the AeroProblem
ap.addVariablesPyOpt(optProb)

# Add DVGeo variables
DVGeo.addVariablesPyOpt(optProb)

# Add constraints
DVCon.addConstraintsPyOpt(optProb)
optProb.addCon("cl_con_" + ap.name, lower=0.5, upper=0.5, scale=10.0)
optProb.addCon("cm_con_" + ap.name, lower=-0.18, upper=10.0, scale=10.0)

# The MP object needs the 'obj' and 'sens' function for each proc set,
# the optimization problem and what the objcon function is:
combinedSolver = SolverCombined(opt=opt, ap=ap, comm=comm, output_dir=args.output, 
                                cfd_frequency=args.cfdFrequency, cfd_include_mode=args.cfdInclude, cfd_iter=args.cfdIter)

MP.setProcSetObjFunc("cruise", combinedSolver.wrap_cruiseFuncs(cruiseFuncs_ml, cruiseFuncs_cfd, _pre=cruiseFuncs_pre))
MP.setProcSetSensFunc("cruise", combinedSolver.wrap_cruiseFuncsSens(cruiseFuncsSens_ml, cruiseFuncsSens_cfd)    )
MP.setObjCon(objCon) # must have this empty function
MP.setOptProb(optProb)
optProb.printSparsity()
# rst optprob (end)


# Run Optimization
sol = opt(optProb, sens=MP.sens, storeHistory=os.path.join(args.output, "opt.hst"))
if comm.rank == 0:
    print(sol)
