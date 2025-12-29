## Inverted pendulum
- clipping gradients reduced variance
- standardizing adv helped stabilize kl divergence and keeping it low. 
- the variation starts when entropy loss first hits 0. Suggests that exploration is low. So, increasing entropy beta(current .001) might help. 
- Found the cause of instability of training:

$$\mathcal{L} =
\underbrace{-\log \pi(a_t|s_t) \cdot A_t}_{\text{policy loss}}
+ \underbrace{c_v (V - V_{target})^2}_{\text{value loss}} \;-\; \underbrace{\beta \cdot H(\pi)}_{\text{entropy bonus}}$$