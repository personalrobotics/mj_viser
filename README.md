# mj-viser

Web-based MuJoCo viewer using [Viser](https://github.com/nerfstudio-project/viser).

## Installation

```bash
pip install mj-viser
```

## Quick Start

```python
import mujoco
from mj_viser import MujocoViewer

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

viewer = MujocoViewer(model, data)
viewer.launch()  # Opens browser at http://localhost:8080
```

## Development

```bash
git clone https://github.com/siddhss5/mj_viser.git
cd mj_viser
uv sync --all-extras
uv run pytest
```

## License

MIT
