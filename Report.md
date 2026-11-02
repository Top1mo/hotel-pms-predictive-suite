# History of findings
-This is a history of findings to make report writing and documentaion easier for the team and help with the flow of the project

## 

### List
    1. Data has 32.65% duplicates dropping them drops sample count from 79,330 to ~53,428 Duplicates: 25902 (32.65%)

    2. removed illogical data (adults + childrens +babies = 0) to become 53,274 not much but noted

    3. company should be neglected as a feature as its 95 percent null but that will be decided by 3ada and omar 
    
    4. company has alot of nulls but it won't be neglected because the nulls means there is no company so it will by converted 
    in to a has_company flag

    5. the nulls in agent feature will be replaced with "no_agent"

    6. the nulls in country it will be replaced with the mode "prt" and the nulls in children will be replaced with the median "0"
    
    7.there are features that will cause leakage so we droped as the prediction point is the booking time

     ReservationStatus , ReservationStatusDate , AssignedRoomType , BookingChanges

    8. Created Arrival_Month_Num, ArrivalDayOfWeek, and Is_Weekend_Arrival, and dropped the original raw year, month, week, and day columns.

    9. combine StaysInWeekendNights + StaysInWeekNights into Total_Stay_Nights, and combine Adults + Children + Babies into Total_Guests. Also add an Is_Family flag.

    10. take care of the target in every pipeline: ADR is the target for regression, IsCanceled for classification, and clustering has no target. The target must never be in the inputs, and the same goes for the other leakage columns: ReservationStatus, ReservationStatusDate, AssignedRoomType, BookingChanges, and the borderline ones (RequiredCarParkingSpaces, DaysInWaitingList, TotalOfSpecialRequests).  3asy , mohamed
    


##