--[[
--  Squatting task for RBDL Leo simulation
--  using MPRL-style observations and rewards.
--
--  Authors:
--    Wouter Caarls <wouter@caarls.org>
--    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
--]]

-- Helper functions
math.randomseed( os.time() )
pi = math.pi

reward_shaping = false
terminate_on_success = false
maxvoltage = 10.7

function failed(state)
  if state[0] < -1.4 or state[0] > 1.4 then
    return true
  else
    return false
  end
end

function succeeded(state)
  return true
end

-- Exported functions

function configure(argstr)
  T = 10
  return {observation_dims = 8,
          -- ankle.angle, knee.angle, hip.angle, shouler.angle, ankle.anglerate, knee.anglerate, hip.anglerate, shouler.anglerate
          observation_min = {-pi, -pi, -pi, -pi, -10*pi, -10*pi, -10*pi, -10*pi},
          observation_max = { pi,  pi,  pi,  pi,  10*pi,  10*pi,  10*pi,  10*pi},
          action_dims = 3,
          action_min = {-maxvoltage, -maxvoltage, -maxvoltage, -maxvoltage},
          action_max = { maxvoltage,  maxvoltage,  maxvoltage,  maxvoltage},
          reward_min = -1000,
          reward_max = 2000
          }
end

function start(test)
  if test == 1 then -- !!! Lua always treats 'numbers' as 'true'. For a correct boolean expression compare numbers !!! --
    return {0, 0, 0, 0, 0, 0, 0, 0}
  else
    C = 1*0.087263889; -- 0.087263889 = +/- 5 deg
    r1 = math.random(-C, C);
    r2 = math.random(-C, C);
    r3 = math.random(-C, C);
    return {0, 0, 0, 0, 0, 0, 0, 0}
  end
end

function observe(state)
  if failed(state) or (succeeded(state) and terminate_on_success) then
    t = 2
  elseif state[4] > T - 1E-3 then
    t = 1
  else
    t = 0
  end
  
  return {state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]}, t
end

function evaluate(state, action, next)
  if failed(next) then
    if not reward_shaping then
      return -1000
    else
      return -1000
    end
  else
    if not reward_shaping then
      return 1
    end
  end
end

function invert(obs)
  return {obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], 0}
end