import pandas as pd
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random

from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer


# ==========================================
# DATA LOADER (PROCESSED DATA)
# ==========================================
class ChicagoDataLoader:
    def __init__(self):
        self.has_data = False
        self.community_stats = {}

        try:
            self.crimes = pd.read_csv("processed_crime.csv")
            df_census = pd.read_csv("processed_census.csv")

            for _, row in df_census.iterrows():
                # Skip the citywide summary row (Community Area Number is NaN)
                if pd.isna(row['Community Area Number']):
                    continue
                self.community_stats[int(row['Community Area Number'])] = {
                    "unemployment": row['unemployment'],
                    "hardship": row['hardship']
                }

            self.has_data = True
            print("✅ Processed data loaded")

        except Exception as e:
            print("⚠️ Error:", e)


# ==========================================
# PERSON AGENT (FULL LOGIC)
# ==========================================
class Person(Agent):
    def __init__(self, uid, model, ca):
        super().__init__(uid, model)

        self.community_area_id = ca
        stats = model.loader.community_stats.get(ca, {"unemployment": 0.15, "hardship": 0.5})

        # Employment (heterogeneous)
        self.employment = random.choices(
            ["Employed", "Unemployed"],
            weights=[1 - stats["unemployment"], stats["unemployment"]]
        )[0]

        # Stress (SDT)
        self.stress = 0.2 + stats["hardship"] + random.uniform(0, 0.2)

        # Zones (RAT movement)
        self.home = None
        self.work_zone = (random.randint(5, 10), random.randint(5, 10))
        self.leisure_zone = (random.randint(10, 18), random.randint(10, 18))

        # Memory
        self.past_risk = 0

        self.crime = False

    # --------------------------------------
    # Movement (RAT)
    # --------------------------------------
    def move(self):
        hour = self.model.time % 24
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

        if self.employment == "Employed" and 9 <= hour <= 17:
            target = self.work_zone
        elif 18 <= hour <= 23:
            target = self.leisure_zone
        else:
            target = self.home

        new_pos = min(neighbors, key=lambda p: (p[0] - target[0])**2 + (p[1] - target[1])**2)
        self.model.grid.move_agent(self, new_pos)

    # --------------------------------------
    # Interaction / Escalation (I)
    # --------------------------------------
    def escalate(self, neighbors):
        esc = 0
        for n in neighbors:
            if isinstance(n, Person):
                if self.employment != n.employment:
                    esc += 0.1
                esc += abs(self.stress - n.stress) * 0.2
        return esc

    # --------------------------------------
    # Risk Engine (S × T + I)
    # --------------------------------------
    def interact(self):
        x, y = self.pos

        stats = self.model.loader.community_stats.get(self.community_area_id, {"hardship": 0.5})

        # S = Stress (SDT)
        S = self.stress + stats["hardship"]

        # T = Environment
        T = self.model.risk_grid[x][y]

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        # I = Interaction
        I = self.escalate(neighbors)

        # Time factor
        hour = self.model.time % 24
        time_factor = 1.5 if (hour >= 20 or hour <= 2) else 1

        # Core equation with tunable interaction multiplier (gamma)
        current = (S * T + self.model.gamma * I) * time_factor

        # Memory
        final = 0.8 * current + 0.2 * self.past_risk
        self.past_risk = final

        # Interaction-based violence
        for n in neighbors:
            if isinstance(n, Person):
                combined = (self.stress + n.stress) / 2
                if combined > 0.7 and random.random() < 0.3:
                    self.crime = True
                    self.model.crime_map[x][y] += 1
                    break

        # Threshold trigger
        if final > 1.2:
            self.crime = True
            self.model.crime_map[x][y] += 1
            self.stress *= 0.5

        # Reset
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
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        for n in neighbors:
            if isinstance(n, Person):
                n.stress *= 0.7


# ==========================================
# MODEL
# ==========================================
class CrimeCity(Model):
    def __init__(self, N=200, width=20, height=20, gamma=0.2):
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.time = 0
        self.grid_width = width
        self.grid_height = height
        self.gamma = float(gamma)

        self.loader = ChicagoDataLoader()

        # Uniform low baseline risk (unseeded): no real-crime coordinates injected.
        self.risk_grid = np.full((self.grid_width, self.grid_height), 0.1)
        self.crime_map = np.zeros((width, height))

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

        # DataCollector (UI fix)
        self.datacollector = DataCollector({
            "Crimes": lambda m: sum(a.crime for a in m.schedule.agents if isinstance(a, Person))
        })

        print("✅ Agents created:", len(self.schedule.agents))

    def step(self):
        self.time += 1

        # Dynamic environment
        self.risk_grid *= 0.99

        for contents, (x, y) in self.grid.coord_iter():
            if contents:
                self.risk_grid[x][y] += 0.01 * len(contents)

        self.datacollector.collect(self)
        self.schedule.step()


# ==========================================
# VISUALIZATION
# ==========================================
def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.6,
        "Layer": 0
    }

    if isinstance(agent, Police):
        portrayal["Color"] = "black"
        portrayal["r"] = 0.7
        portrayal["Layer"] = 2
    elif agent.crime:
        portrayal["Color"] = "red"
        portrayal["r"] = 0.9
        portrayal["Layer"] = 2
    elif agent.employment == "Unemployed":
        portrayal["Color"] = "orange"
        portrayal["Layer"] = 1
    else:
        portrayal["Color"] = "blue"

    return portrayal


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
    chart = ChartModule([{"Label": "Crimes", "Color": "Red"}])

    server = ModularServer(
        CrimeCity,
        [grid, chart],
        "ABM Crime Simulation (FINAL FULL)",
        {"N": 200, "width": 20, "height": 20}
    )

    server.port = 8521
    server.launch()