# Project RACE: Realtime APF for UAVs in Clutered Environments

## Setup

1. Follow the **Developer install** steps here: https://utiasdsl.github.io/crazyflow/installation/

   1. ```
      git clone --recurse-submodules git@github.com:utiasDSL/crazyflow.git
      cd crazyflow
      ```

   2. ```pixi shell -e gpu``` or `pixi shell` if you don't have an Nvidia GPU

   3. `python examples/hover.py` from inside `crazyflow` directory to confirm install

2. Exit pixi with `exit`

3. Go back to `project_RACE` directory and start pixi

   1. Look inside `pixi.toml` and update the path for crazyflow
   2. Run `pixi shell ` or `pixi shell -e gpu` This starts pixi inside the project_RACE directory
   3. Run `python simpleHover.py` and hopefully that works!

## Usage
positional arguments:
  map_xml                              XML file containing the static environment and static obstacles.

options:
  -h, --help                           show this help message and exit
  --moving-spheres MOVING_SPHERES      Number of moving sphere obstacles to add at runtime.
  --motion {static,circle,random}      Motion model used for all moving spheres.
  --save-video                         Save an MP4 capture of the simulation.
  --no-vis                             Disable live rendering and the final matplotlib plot.
  --timeout TIMEOUT                    Simulation timeout in seconds.

### Example command
`python safeapf.py cave_map.xml --moving-spheres 2 --motion random --save-video --timeout 20`