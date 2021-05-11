#!/usr/bin/env python

import argparse
import numpy as np
import xarray as xr


def reshape_var(v, dims, newdim_name):
    v = v.stack(newdim=dims)
    v = v.reset_index("newdim")
    v = v.drop(v.coords.keys())
    v = v.rename({"newdim": newdim_name})

    return v


description = """Convert a ROMS file to a UGRID compliant netCDF"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("roms", type=str, help="input filepath")
parser.add_argument("ugrid", type=str, help="output filepath")
parser.add_argument("--vars", type=str, nargs="*", help="variables to process (only supports RHO vars atm)")
args = parser.parse_args()

# By default, we always process the bathymetry and mask
vars = ["h", "mask_rho"]
if args.vars is not None:
    vars = vars + args.vars

roms = xr.open_dataset(args.roms)

##############################
# Process the mesh variables #
##############################

# The rho points are considered to to be the face points of the output mesh,
# with the boundary points excluded (since they don't have bounding psi-points)
face_dims = ("xi_rho", "eta_rho")
lat_rho = reshape_var(roms.lat_rho[1:-1, 1:-1], face_dims, "nFace")
lon_rho = reshape_var(roms.lat_rho[1:-1, 1:-1], face_dims, "nFace")

lat_rho.attrs = {"standard_name": "latitude",
                 "long_name": "Characteristics latitude of 2D mesh face.",
                 "units": "degrees_north",
                 "bounds": "face_ybnds"}
lon_rho.attrs = {"standard_name": "longitude",
                 "long_name": "Characteristics longitude of 2D mesh face.",
                 "units": "degrees_east",
                 "bounds": "face_xbnds"}

# The psi points are considered to be the mesh nodes
node_dims = ("xi_psi", "eta_psi")
lat_psi = reshape_var(roms.lat_psi, node_dims, "nNode")
lon_psi = reshape_var(roms.lon_psi, node_dims, "nNode")

lat_psi.attrs = {"standard_name": "latitude",
                 "long_name": "Latitude of 2D mesh nodes.",
                 "units": "degrees_north"}
lon_psi.attrs = {"standard_name": "longitude",
                 "long_name": "Longitude of 2D mesh nodes.",
                 "units": "degrees_east"}

# Map out the nodes to their associated faces
face_nodes = xr.DataArray(np.empty((lat_rho.size, 4), dtype=int), dims=["nFace", "nFaceMax"])
face_nodes[:, 0] = np.array([n + np.floor(n/(roms.dims["eta_rho"] - 2))
                             for n in range(face_nodes.shape[0])])
face_nodes[:, 1] = face_nodes[:, 0] + roms.dims["eta_psi"]
face_nodes[:, 2] = face_nodes[:, 1] + 1
face_nodes[:, 3] = face_nodes[:, 0] + 1

face_nodes.attrs = {"cf_role": "face_node_connectivity",
                    "long_name": "Maps every face to its corner nodes.",
                    "start_index": 0,
                    "standard_name": "face_node_connectivity",
                    "units": "nondimensional"}

# # The U and V points are considered to be the mesh edges
lat_u = roms.lat_u[1:-1, :].stack(nEdge=("xi_u", "eta_u"))
lon_u = roms.lon_u[1:-1, :].stack(nEdge=("xi_u", "eta_u"))

lat_u.attrs = {"standard_name": "latitude",
               "units": "degrees_north"}
lon_u.attrs = {"standard_name": "longitude",
               "units": "degrees_east"}

lat_v = roms.lat_v[:, 1:-1].stack(nEdge=("xi_v", "eta_v"))
lon_v = roms.lon_v[:, 1:-1].stack(nEdge=("xi_v", "eta_v"))

lat_v.attrs = {"standard_name": "latitude",
               "units": "degrees_north"}
lon_v.attrs = {"standard_name": "longitude",
               "units": "degrees_east"}

# Create the UGRID mesh topology object
mesh = xr.DataArray(np.zeros((1), dtype=np.int32), dims=["One"],
                    attrs={"cf_role": "mesh_topology",
                           "long_name": "Topology data of 2D mesh.",
                           "topology_dimension": 2.0,
                           "node_coordinates": "lon_psi lat_psi",
                           "face_coordinates": "lon_rho lat_rho",
                           "face_node_connectivity": "face_nodes",
                           "face_dimension": "nFace"})

########################################
# Process the requested data variables #
########################################

data_vars = {}
for v in vars:
    if v not in roms.variables.keys():
        print("Error: {0} not found in input dataset! Skipping...".format(v))
    else:
        print("Processing {0}...".format(v))
        if ("eta_rho" in roms[v].dims) and ("xi_rho" in roms[v].dims):  # Face var
            data_vars[v] = reshape_var(roms[v].isel(eta_rho=roms[v].eta_rho[1:-1],
                                                    xi_rho=roms[v].xi_rho[1:-1]),
                                       face_dims, "nFace")

            data_vars[v].attrs["mesh"] = "mesh"
            data_vars[v].attrs["location"] = "face"
            data_vars[v].attrs["coordinates"] = "lon_rho lat_rho"

            if "coordinates" in data_vars[v].encoding.keys():
                data_vars[v].encoding.pop("coordinates")


# Construct the output dataset
data_vars.update({"lat_rho": lat_rho,
                  "lon_rho": lon_rho,
                  "lat_psi": lat_psi,
                  "lon_psi": lon_psi,
                  "face_nodes": face_nodes,
                  "mesh": mesh})
ugrid = xr.Dataset(data_vars)

# If any DataArrays with a time dimension were introduced, then add the time
# coord to the output dataset, and give it a more "standard" name
if "ocean_time" in ugrid.dims.keys():
    ugrid["ocean_time"] = roms.ocean_time
    ugrid = ugrid.rename({"ocean_time": "time"})

ugrid.attrs["Conventions"] = "CF-1.6, UGRID-1.0"

ugrid.to_netcdf(args.ugrid)
ugrid.close()
