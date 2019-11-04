
run_assignment.m: The main script which runs Task 2-7. Task 6 takes quite alot of time for large n, I'd reduce n or 
comment that section while running the rest of the script.  


Functions: 
Ymodel_i.m: The polynomial model (model (2) in Ning et al. (2014) paper).

Ymodel_ii.m: The combined Hill-based model (model (3) in Ning et al. (2014) paper). 
 
Ymodel_iii.m: The simplified Hill model (model from assignment instructions) 


func_i.m: Function to estimate the parameters of the polynomial model using the ordinary least square fitting 
criterion. This is numerically minimized (using fminunc) in run_assigment.m.

func_ii.m: Function to estimate the parameters of the combined Hill-based model using the ordinary least square fitting
criterion. This is numerically minimized (using fminunc) in run_assigment.m.

func_iii.m: Function to estimate the parameters of the simplified Hill model using the ordinary least square fitting criterion.
This is numerically minimized (using fminunc) in run_assigment.m.
