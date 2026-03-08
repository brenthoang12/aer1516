# Getting crazyflow working

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
   2. Run `pixi shell` This starts pixi inside the project_RACE directory
   3. Run `python simpleHover.py` and hopefully that works!