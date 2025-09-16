#scheduler.py
import pandas as pd
import random

def solve_schedule(avail_df):
    employees = avail_df['Employee'].tolist()
    shifts = avail_df.columns[2:].tolist()  # skip Employee & MaxHoursPerWeek
    max_hours = dict(zip(employees, avail_df['MaxHoursPerWeek']))

    # Initialize schedule
    schedule = pd.DataFrame(0, index=employees, columns=shifts)

    # Shuffle employees to reduce bias from list order
    random.shuffle(employees)

    # Assign shifts
    for s in shifts:
        # Find available employees who haven't reached max_hours
        candidates = [e for e in employees 
                      if avail_df.loc[avail_df['Employee']==e, s].values[0] == 1
                      and schedule.loc[e].sum() < max_hours[e]]
        if candidates:
            # Pick the one with fewest assigned hours so far
            chosen = min(candidates, key=lambda e: schedule.loc[e].sum())
            schedule.loc[chosen, s] = 1

    return schedule
