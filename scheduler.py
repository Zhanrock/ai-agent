#scheduler.py
import pandas as pd
import random

def solve_schedule(avail_df):
    employees = avail_df['Employee'].tolist()
    shifts = avail_df.columns[2:].tolist()  # skip Employee & MaxHoursPerWeek
    max_hours = dict(zip(employees, avail_df['MaxHoursPerWeek']))

    # Initialize schedule
    schedule = pd.DataFrame(0, index=employees, columns=shifts)

    # Shuffle employees to reduce bias
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

def swap_shift(schedule, emp1, emp2, shift):
    """
    Swap shifts between two employees only if both have the shift assigned (1).
    Returns True if swap successful, False otherwise.
    """
    val1 = schedule.loc[emp1, shift]
    val2 = schedule.loc[emp2, shift]

    if val1 == 1 and val2 == 1:
        schedule.loc[emp1, shift], schedule.loc[emp2, shift] = val2, val1
        return True
    else:
        return False
