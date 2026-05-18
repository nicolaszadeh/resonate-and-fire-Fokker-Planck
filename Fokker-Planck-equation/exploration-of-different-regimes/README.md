The three folders in this sub-repository all contain:
- a *rel_funct_steady_state.py script, which creates a .npz file containing the values of the numerical solution obtained via our scheme at T_inf>T. Booleans allow to obtain (or not) a 2D plot of that state, multiple 3D plots from pre-defined angles, but also relative entropy and Fisher information computed on [0,T].
- a *vid_snaps_std_funct.py script, containing booleans allowing to witness the evolution of a solution in a given regime on [0,T], through a video or snapshots. Graphs displaying the values of functionals of interest through time, such as the all-important mean-voltage X(t), network activity N(t) or entropy and Fisher information can also be obtained
- a *3d_vid.py script, which creates a .mp4 file displaying the evolution of a solution in a given regime on [0,T_3d]. There is an important (yet optional) clipping parameter which allows to cut the highest values of the function, and to plot accordingly to them, as well as choosing the colorscale thanks to that clipping.

The outputs can be found inside of an automatically created tree structure in a 'Results' folder, appearing where the script is stored. 

The computation can be long if an important precision is asked for, a boolean 'USE_SOUND' is then implemented (by default at 'True') which plays a different melody in case of success or failure in the execution of the routine.

If the user wants to obtain the video animations, they will need to install FFmpeg. To verify that FFmpeg is correctly installed, run 
```bash
ffmpeg -version 
``` 
in a terminal.
Should FFmpeg be missing, it can be installed on:
## Linux
```bash
sudo apt install ffmpeg
```
## macOS
```bash
brew install ffmpeg
```
## Windows
From the website: ffmpeg.org/download.html.

A simple test a posteriori is to run Python from a terminal, and run the following command:
```python
import shutil
print(shutil.which("ffmpeg"))
```
The path to the executable should then appear.

It should be noted that once the execution of the scheme in itself is successful, the production of the videos usually takes a few minutes because of the format we used: this is not a sign of any malfunction and shouldn't alert the user. 
In the meantime they can consult the .pdf figures which are produced quasi-instantaneously.
