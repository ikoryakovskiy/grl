--[[
--  Mountain car RBDL model, modelled as a pendulum without friction
--
--  Author:
--    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
--]]

r = 100
m = 1
g = 9.81
J = m*r*r

wr = 0.02
wd = 0.01
pr = r+wr
pd = 0.001

joints = {
  hinge = {
    { 1., 0., 0., 0., 0., 0.}
  }
}

visuals = {
  pendulum = {
    {
      name = "plate",
      dimensions = { pd, 2*pr, 2*pr },
      color = {0., 0., 1.},
      mesh_center = { 0., 0., 0. },
      src = "meshes/unit_cylinder_medres_z.obj"
    },
    {
      name = "weight",
      dimensions = { wd, 2*wr, 2*wr },
      color = {.8, .8, .8},
      mesh_center = { (pd+wd)/2, 0., l },
      src = "meshes/unit_cylinder_medres_z.obj"
    }
  }
}

-- **************************************************************************
-- *** CONTROLLER ***********************************************************
-- **************************************************************************

function control(state, action)
  return {action[0] * r}
end

-- **************************************************************************
-- *** MODEL ****************************************************************
-- **************************************************************************

model = {
  gravity = {0., 0., -g},
  
  frames = {
    {
      name = "pendulum",
      parent = "ROOT",
      body = {
        mass = m,
        com = {0., 0., r},
        inertia = {{ J, 0., 0.},
                   {0., 0., 0.},
                   {0., 0., 0.}}
      },
      joint = joints.hinge,
      visuals = visuals.pendulum
    }
  },
}

return model
