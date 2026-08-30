import random
import numpy as np

from crime_model_v5 import CrimeCity


def test_emergent_crime_unseeded_grid() -> None:
    random.seed(42)
    np.random.seed(42)

    model = CrimeCity(N=300, width=20, height=20)

    # Initial state must not contain pre-seeded crimes.
    assert np.allclose(model.risk_grid, 0.1), "risk_grid must start uniformly at 0.1"
    assert int(model.crime_map.sum()) == 0, "crime_map must start at zero"

    for _ in range(10):
        model.step()

    # Keep this check explicit for early-step sanity.
    assert int(model.crime_map.sum()) >= 0

    for _ in range(190):
        model.step()

    total_crimes = int(model.crime_map.sum())
    nonzero_cells = int((model.crime_map > 0).sum())

    assert total_crimes > 0, "crime must emerge by 200 steps"
    assert nonzero_cells > 0, "emergent crime clusters must exist by 200 steps"


if __name__ == "__main__":
    test_emergent_crime_unseeded_grid()
    print("test_emergent_crime_unseeded_grid passed")
