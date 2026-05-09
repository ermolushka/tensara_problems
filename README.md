```
tensara init relu -p relu -l python

tensara checker -g H100 -p relu -s relu/sol.py

tensara benchmark -g H100 -p relu -s relu/sol.py

tensara submit -g H100 -p relu -s relu/sol.py

```
