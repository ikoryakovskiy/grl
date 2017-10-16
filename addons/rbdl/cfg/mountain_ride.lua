--[[
--  Mountain car task, driving from x0 to x1
--  Radius should be same as in the model mountain.lua
--
--  Author:
--    Wouter Caarls <wouter@caarls.org>
--]]

local model = require 'mountain'

if not T then
  T = 10
end

x0 = -1
x1 =  1
r  = 100

function configure()
  return {observation_dims = 2,
          observation_min = {-5, -10},
          observation_max = { 5,  10},
          action_dims = 1,
          action_min = {-10},
          action_max = {10},
          reward_min = -100,
          reward_max = 0
          }
end

function start()
  a = math.asin(x0 / r)
  return {math.pi+a, 0, 0} -- pi is bottom, 0 is top
end

function failed(state)
  return (math.abs(state[0]-math.pi) > 5 or math.abs(state[1]) > 10)
end

function observe(state)
  if state[2] > T-1E-3 then
    t = 1
  elseif failed(state) then
    t = 2
  else
    t = 0
  end
  return {r*math.sin(state[0]-math.pi), r*state[1]*math.cos(state[0]-math.pi)}, t
end

function evaluate(state, action, next)
  if failed(state) then
    return -100
  else
    return 0
  end
end

function invert(obs)
  return {0, 0, 0}
end
