import pandas as pd
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random
import matplotlib.pyplot as plt

# ==========================================
# DATA LOADER (USES PROCESSED DATA)
# ==========================================
class ChicagoDataLoader:
    def __init__(self, crime_file="processed_crime.csv", census_file="processed_census.csv"):
        self.has_data = False
        self.community_stats = {}

        try:
            self.crimes = pd.read_csv(crime_file)
            df_census = pd.read_csv(census_file)

            for _, row in df_census.iterrows():
                ca = row['Community Area Number']
                if pd.notna(ca):
                    self.community_stats[int(ca)] = {
                        "unemployment": row['unemployment'],
                        "hardship": row['hardship']
                    }

            self.has_data = True
            print("✅ Processed data loaded successfully")

        except Exception as e:
            print("⚠️ Error loading processed data:", e)

    def get_area_stats(self, ca):
        return self.community_stats.get(ca, {"unemployment": 0.15, "hardship": 0.5})


# ==========================================
# PERSON AGENT
# ==========================================
class Person(Agent):
    def __init__(self, uid, model, ca):
        super().__init__(uid, model)

        self.community_area_id = ca
        stats = model.loader.get_area_stats(ca)

        self.employment = random.choices(
            ["Employed", "Unemployed"],
            weights=[1 - stats["unemployment"], stats["unemployment"]]
        )[0]

        self.stress = 0.2 + stats["hardship"] + random.uniform(0, 0.2)

        self.home = None
        self.work_zone = (random.randint(5, 10), random.randint(5, 10))
        self.leisure_zone = (random.randint(10, 18), random.randint(10, 18))

        self.past_risk = 0
        self.crime = False

    def move(self):
        hour = self.model.time % 24
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True)

        if self.employment == "Employed" and 9 <= hour <= 17:
            target = self.work_zone
        elif 18 <= hour <= 23:
            target = self.leisure_zone
        else:
            target = self.home

        new_pos = min(neighbors, key=lambda p: (p[0] - target[0])**2 + (p[1] - target[1])**2)
        self.model.grid.move_agent(self, new_pos)

    def escalate(self, neighbors):
        esc = 0
        for n in neighbors:
            if isinstance(n, Person):
                if self.employment != n.employment:
                    esc += 0.1
                esc += abs(self.stress - n.stress) * 0.2
        return esc

    def interact(self):
        x, y = self.pos

        stats = self.model.loader.get_area_stats(self.community_area_id)

        S = self.stress + stats["hardship"]
        T = self.model.risk_grid[x][y]

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        I = self.escalate(neighbors)

        # 🔥 Real-time based on actual data distribution
        hour = random.choice(self.model.loader.crimes['Hour'].values) \
            if self.model.loader.has_data else self.model.time % 24

        time_factor = 1.5 if (hour >= 20 or hour <= 2) else 1

        current = (S * T + I) * time_factor
        final = 0.8 * current + 0.2 * self.past_risk
        self.past_risk = final

        # 🔥 TRUE interaction-based violence
        for n in neighbors:
            if isinstance(n, Person):
                combined = (self.stress + n.stress) / 2
                if combined > 0.7 and random.random() < 0.3:
                    self.crime = True
                    self.model.crime_map[x][y] += 1
                    break

        if final > 1.2:
            self.crime = True
            self.model.crime_map[x][y] += 1
            self.stress *= 0.5

        if self.pos == self.home:
            self.past_risk = 0

    def step(self):
        self.crime = False
        self.move()
        self.interact()


# ==========================================
# POLICE AGENT (GUARDIAN)
# ==========================================
class Police(Agent):
    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        for n in neighbors:
            if isinstance(n, Person):
                n.stress *= 0.7


# ==========================================
# MODEL
# ==========================================
class CrimeCity(Model):
    def __init__(self, N=200, width=20, height=20):
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.time = 0

        self.loader = ChicagoDataLoader()

        self.risk_grid = np.zeros((width, height))
        self.crime_map = np.zeros((width, height))

        # 🔥 Use processed normalized coordinates
        if self.loader.has_data:
            for _, row in self.loader.crimes.iterrows():
                x = int(row['lat_norm'] * (width - 1))
                y = int(row['lon_norm'] * (height - 1))
                self.risk_grid[x][y] += 0.2

        # Agents
        for i in range(N):
            ca = random.choice(list(self.loader.community_stats.keys())) if self.loader.community_stats else 1
            a = Person(i, self, ca)
            self.schedule.add(a)

            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(a, (x, y))
            a.home = (x, y)

        # Police agents
        for i in range(5):
            p = Police(1000 + i, self)
            self.schedule.add(p)
            self.grid.place_agent(p, (random.randrange(width), random.randrange(height)))

    def step(self):
        self.time += 1

        # 🔥 Dynamic environment
        self.risk_grid *= 0.99

        for cell in self.grid.coord_iter():
            contents, x, y = cell
            if contents:
                self.risk_grid[x][y] += 0.01 * len(contents)

        self.schedule.step()


# ==========================================
# HEATMAP
# ==========================================
def plot_heatmap(model):
    plt.imshow(model.crime_map, cmap='hot')
    plt.colorbar(label="Crime Intensity")
    plt.title("Crime Hotspot Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    model = CrimeCity()

    for i in range(200):
        model.step()

    print("✅ Simulation complete")
    plot_heatmap(model)