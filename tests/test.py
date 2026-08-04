import numpy as np

from learning_ddpg.swarm.ccpso import CCPSOSwarm


def sphere(positions):
    return np.sum(positions ** 2, axis=1)


swarm = CCPSOSwarm(
    particles=20,
    dimensions=10,
    fun=sphere,
    lower_bound=-100.0,
    upper_bound=100.0,
    max_fe=1000,
)

swarm.reset()

gbest_history = [swarm.gbest_fitness]

while not swarm.done:
    swarm.step(conv=0.75)
    gbest_history.append(swarm.gbest_fitness)

print("FE:", swarm.fe_count)
print("initial gbest:", gbest_history[0])
print("final gbest:", gbest_history[-1])

assert swarm.fe_count == 1000
assert np.all(np.diff(gbest_history) <= 1e-12)
assert swarm.gbest_fitness <= gbest_history[0]