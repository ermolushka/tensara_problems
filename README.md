Here I mostly store my solutions for writing kernels. I am using triton mostly for now.

Some useful triton cli commands

```
tensara init relu -p relu -l python

tensara checker -g H100 -p relu -s relu/sol.py

tensara benchmark -g H100 -p relu -s relu/sol.py

tensara submit -g H100 -p relu -s relu/sol.py

```
