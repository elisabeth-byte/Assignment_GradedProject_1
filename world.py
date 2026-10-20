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
        """
        Start a new day.
        """

        self.hour = 0

        # Start between 20% and 80% charge
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

        # Advance time
        self.hour += 1

        done = self.hour >= MAX_HOUR

        # Final reward at end of day
        if done:
            if self.battery >= 8:
                reward += 50
            else:
                reward -= 100

        # New electricity price for next hour
        if not done:
            self.price = self.rng.randint(0, 2)

        next_state = (
            self.hour,
            self.battery,
            self.price
        )

        return next_state, reward, done