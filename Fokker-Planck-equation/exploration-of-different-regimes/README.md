The three folders in this sub-repository all contain:
- a *rel_funct_steady_state.py script, which creates a .npz file containing the values of the numerical solution obtained via our scheme at T_inf>T. Booleans allow to obtain (or not) a 2D plot of that state, multiple 3D plots from pre-defined angles, but also relative entropy and Fisher information computed on [0,T].
- a *vid_snaps_std_funct.py script, containing booleans allowing to witness the evolution of a solution in a given regime on [0,T], through a video or snapshots. Graphs displaying the values of functionals of interest through time, such as the all-important mean-voltage X(t), network activity N(t) or entropy and Fisher information can also be obtained.

The outputs can be found inside of an automatically created tree structure in a 'Results' folder, appearing where the script is stored. 

The computation can be long if an important precision is asked for, a boolean 'USE_SOUND' is then implemented (by default at 'True') which plays a different melody in case of success or failure in the execution of the routine.

If the user wants to obtain the video animations, they will need to install FFmpeg. To verify that FFmpeg is correctly installed, run '''bash
ffmpeg -version ``` in a terminal.
Should FFmpeg be missing, it can be installed on:
## Linux
```bash
sudo apt install ffmpeg```
## macOS
```bash
brew install ffmpeg```
## Windows
From the website: ffmpeg.org/download.html.

A simple test a posteriori is to run Python from a terminal, and run the following command:
```python
import shutil
print(shutil.where("ffmpeg"))```
The path to the executable should then appear.

