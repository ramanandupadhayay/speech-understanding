def next_birthday(date, birthdays):
   
    # Sort all birthday dates
    sorted_dates = sorted(birthdays.keys())
    
    # Check for the next date after the given date
    for bday in sorted_dates:
        if bday > date:
            return bday, birthdays[bday]
    
    # If none found, wrap around to the first date
    first_date = sorted_dates[0]
    return first_date, birthdays[first_date]
