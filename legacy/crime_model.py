import pandas as pd
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
import random

# ==========================================
# 1. DATA LOADER
# ==========================================
class ChicagoDataLoader:
    def __init__(self, crime_file="crime_data.csv", census_file="census_data.csv"):
        self.has_data = False
        self.community_stats = {}
        
        try:
            df_crime = pd.read_csv(crime_file)
            violent_types = ['BATTERY', 'ASSAULT', 'ROBBERY', 'HOMICIDE', 'CRIM SEXUAL ASSAULT']
            self.crimes = df_crime[df_crime['Primary Type'].isin(violent_types)]
            self.crimes = self.crimes.dropna(subset=['Latitude', 'Longitude', 'Community Area'])
            
            df_census = pd.read_csv(census_file)
            
            for _, row in df_census.iterrows():
                ca_id = row['Community Area Number']
                if pd.notna(ca_id):
                    self.community_stats[int(ca_id)] = {
                        "unemployment": row['PERCENT AGED 16+ UNEMPLOYED'],
                        "hardship": row['HARDSHIP INDEX']
                    }

            print("✅ Data loaded successfully")
            self.has_data = True
            
        except:
            print("⚠️ Using synthetic data mode")
            self.has_data = False

    def get_hotspots(self):
        if not self.has_data:
            return []
        return list(zip(self.crimes['Latitude'], self.crimes['Longitude'], self.crimes['Community Area']))

    def get_area_stats(self, community_area_id):
        if not self.has_data or community_area_id not in self.community_stats:
            return {"unemployment": 15.0, "hardship": 50.0}
        return self.community_stats[community_area_id]


# ==========================================
# 2. AGENT (UPGRADED)
# ==========================================
class Person(Agent):
    def __init__(self, unique_id, model, community_area_id):
        super().__init__(unique_id, model)

        stats = model.loader.get_area_stats(community_area_id)
        local_unemployment = stats['unemployment'] / 100.0

        # --- DEMOGRAPHICS ---
        self.age = random.randint(18, 60)

        self.employment = random.choices(
            ["Employed", "Unemployed"],
            weights=[1 - local_unemployment, local_unemployment]
        )[0]

        # --- ROUTINES ---
        if self.employment == "Employed":
            self.routine = "Work"
        else:
            self.routine = random.choice(["Leisure", "Random"])

        # --- HOME ---
        self.home = None

        # --- STRESS ---
        base = 0.4 if self.employment == "Unemployed" else 0.1
        if stats['hardship'] > 60:
            base += 0.2

        self.stress_level = base + random.uniform(0, 0.1)
        self.crime_committed = False

    def move(self):
        hour = self.model.time % 24
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

        # --- ROUTINE-BASED MOVEMENT ---
        if self.routine == "Work" and 9 <= hour <= 17:
            target = (10, 10)
        elif self.routine == "Leisure" and 18 <= hour <= 23:
            target = (15, 15)
        else:
            target = self.home

        # Move towards target
        new_position = min(
            possible_steps,
            key=lambda pos: (pos[0] - target[0])**2 + (pos[1] - target[1])**2
        )

        self.model.grid.move_agent(self, new_position)

    def interact(self):
        x, y = self.pos

        S = self.stress_level
        T = self.model.risk_grid[x][y]

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        I = len(neighbors) * 0.1

        P = (S * T) + I

        if P > 1.3 and self.random.random() < 0.2:
            self.crime_committed = True
            self.stress_level = 0.2

    def step(self):
        self.crime_committed = False
        self.move()
        self.interact()


# ==========================================
# 3. MODEL
# ==========================================
class CrimeCity(Model):
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True
        self.time = 0  # 🔥 NEW (TIME)

        self.loader = ChicagoDataLoader("crime_data.csv", "census_data.csv")
        hotspots = self.loader.get_hotspots()

        self.risk_grid = np.zeros((width, height))

        if self.loader.has_data and hotspots:
            lats = [x[0] for x in hotspots]
            lons = [x[1] for x in hotspots]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)

            for lat, lon, _ in hotspots:
                gx = int(((lat - min_lat) / (max_lat - min_lat + 1e-5)) * (width - 1))
                gy = int(((lon - min_lon) / (max_lon - min_lon + 1e-5)) * (height - 1))
                self.risk_grid[gx][gy] += 0.1
        else:
            for i in range(8, 12):
                for j in range(8, 12):
                    self.risk_grid[i][j] = 0.8

        # --- CREATE AGENTS ---
        for i in range(self.num_agents):
            random_ca = random.choice(list(self.loader.community_stats.keys())) if self.loader.community_stats else 1

            a = Person(i, self, random_ca)
            self.schedule.add(a)

            x = self.random.randrange(width)
            y = self.random.randrange(height)

            self.grid.place_agent(a, (x, y))

            # 🔥 Assign HOME
            a.home = (x, y)

        self.datacollector = DataCollector({
            "Crimes": lambda m: sum(a.crime_committed for a in m.schedule.agents)
        })

    def step(self):
        self.time += 1  # 🔥 TIME UPDATE
        self.datacollector.collect(self)
        self.schedule.step()


# ==========================================
# 4. VISUALIZATION
# ==========================================
def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.6,
        "Layer": 0
    }

    if agent.crime_committed:
        portrayal["Color"] = "red"
        portrayal["Layer"] = 2
        portrayal["r"] = 0.9

    elif agent.employment == "Unemployed":
        portrayal["Color"] = "orange"
        portrayal["Layer"] = 1

    else:
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 0

    return portrayal


grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
chart = ChartModule([{"Label": "Crimes", "Color": "Red"}])

server = ModularServer(
    CrimeCity,
    [grid, chart],
    "Stage 2 + 3: ABM Simulation",
    {"N": 200, "width": 20, "height": 20}
)

server.port = 8521

if __name__ == "__main__":
    server.launch()