# Scenario: Electric Vehicle Charging

This project investigates the use of Q-learning for smart charging of an electric vehicle (EV). The objective is to determine when the vehicle should be charged during a 24-hour period while electricity prices vary throughout the day.

The agent represents a charging controller that must make a decision every hour. At each timestep, the agent can either charge the battery or wait. Charging increases the battery level but incurs a cost that depends on the current electricity price.

The challenge is that the battery must reach a sufficient charge level before the vehicle is needed, while minimizing the total charging cost. Decisions made earlier in the day affect the available battery level later, making this a sequential decision-making problem that is well suited for reinforcement learning.

The environment contains uncertainty through stochastic electricity prices, which vary between episodes according to a random seed.

# Reward Function

The reward function is designed to encourage low charging costs while ensuring that the vehicle has sufficient battery capacity at the end of the day.

For each hour:

- If the agent chooses to charge:
  
  reward = - electricity_price

- If the agent chooses not to charge:
  
  reward = 0

At the end of the episode:

- Battery level ≥ 80%:
  
  reward += 50

- Battery level < 80%:
  
  reward -= 100

This reward structure encourages the agent to delay charging until electricity prices are low, while still ensuring that the vehicle is sufficiently charged before departure.

# Modeling

## States

The state consists of three variables:

- Hour of the day (0–23)
- Battery level (0–10)
- Electricity price level (low, medium, high)

The battery level is discretized into 11 levels, where:

- 0 = 0%
- 1 = 10%
- ...
- 10 = 100%

The electricity price is discretized into three categories:

- 0 = Low
- 1 = Medium
- 2 = High

The resulting state representation is:

(state) = (hour, battery_level, price_level)

This gives a total of:

24 × 11 × 3 = 792 states

which is large enough to create a meaningful learning problem while remaining manageable for tabular Q-learning.

## Actions

The agent can choose between two actions every hour:

- 0 = Do not charge
- 1 = Charge

If the agent chooses to charge, the battery level increases by one battery unit (10%), up to a maximum of 100%.

## Episode

One episode represents a single day.

The episode starts at hour 0 and ends after hour 23.

## Reward Function

The reward function encourages the agent to minimize charging cost while ensuring that the vehicle has sufficient battery capacity at the end of the day.

For each timestep:

- Charge:
  
  reward = - electricity_price

- Do not charge:
  
  reward = 0

At the end of the episode:

- Battery level ≥ 80%:
  
  reward += 50

- Battery level < 80%:
  
  reward -= 100

This reward structure encourages the agent to charge when electricity prices are low while still reaching the required battery level before the end of the day.
