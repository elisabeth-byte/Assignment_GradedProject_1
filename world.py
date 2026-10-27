import random

# ==============================================================
# WORLD
# ==============================================================

MAX_HOUR = 24
MAX_BATTERY = 10

# Electricity price categories
# 0 = low, 1 = medium, 2 = high
PRICES = {
    0: 2,
    1: 5,
    2: 10,
}


class EVChargingEnv:

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def reset(self):

        ## Start a new day.
        self.hour = 0

        # Start with a random battery level between 20% and 80%
        self.battery = self.rng.randint(2, 8)

        # Random electricity price
        self.price = self.rng.randint(0, 2)

        return (self.hour, self.battery, self.price)

    def step(self, action):
        """
        Actions:
            0 = Do not charge
            1 = Charge

        State:
            (hour, battery_level, price_level)
        """

        reward = 0

        # Charge battery
        if action == 1:
            self.battery = min(self.battery + 1, MAX_BATTERY)

            # Pay electricity cost
            reward -= PRICES[self.price]

        # Reward shaping: penalize battery levels below 80%
        if self.battery < 8:
            reward -= (8 - self.battery)

        # Advance time
        self.hour += 1

        done = self.hour >= MAX_HOUR

        # Final reward based on how close the battery is to the target level
        if done:
            if self.battery >= 8:
                reward += 80 + 10 * (self.battery - 8)
            else:
                reward -= 20 * (8 - self.battery)

        # New electricity price for next hour
        if not done:
            self.price = self.rng.randint(0, 2)

        # State = (hour, battery_level, price_level)
        next_state = (
            self.hour,
            self.battery,
            self.price
        )

        return next_state, reward, done