# Growth_NAS
Tutorial of keeping codes both updated on titanx02 and NewServer but tuning parameter:

Target:

      1. we have 2 server: tatanx02 and newserver.
      
      2. we  want to tune params on both server; 
      but after finding optimal param, turn both server into the optimal one.
      
Methods:

At time=0: 

          let's say both titanx2 and newserver are at version0 (use 'git pull' before tuning param)

At time=1: 

           modified codes on titanx2 to param_setA, run;           
           modified codes on newServer to param_setB, run;             
           
At time=2: I found that param_setB works better than A.

           Thus do: 
           
           In NewServer: git add --all
           
                         git commit -m 'message'   
                         
                         (or use 'git commit -a' instead of above 2 commands.)
                         
                         git push  (now the code is in version1)
                         
           In titanx02:  git reset --hard HEAD  (this revert codes to version0)
           
                         git pull   (now code in version1 is also in titanx02)
                         
                         
