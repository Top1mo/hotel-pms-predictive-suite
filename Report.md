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

## Current preparation protocol

This section supersedes earlier provisional preprocessing notes. The authoritative flow reads raw H2, checks the source schema, normalizes missing markers, removes known zero-guest rows, and removes exact full-row duplicates. Deduplication reduces repeated-record weighting and is a team modeling choice; identical source records may represent real repeated bookings, so results describe the deduplicated population. Different outcomes are never collapsed merely because their predictor values match. ADR extremes are retained because no verified target-based exclusion rule has been established.

Cleaning retains 53,274 rows: 79,330 original rows minus 167 known zero-guest rows and 25,889 duplicate rows among the remainder. Row IDs and source-file SHA-256 are recorded. The fixed-seed grouped random split contains 31,921 training rows, 10,650 validation rows and 10,703 test rows. The requested 60/20/20 fractions are fractions of groups. Groups use the 20 approved booking-time source inputs, excluding outcomes; matching inputs remain together even with distinct outcomes. This is not customer-level grouping or future-period validation. Earlier exploratory work inspected the full dataset; the test partition is reserved from this point onward and is not evaluated by preparation commands.

Prediction is assumed at booking creation. Supervised inputs are lead time; arrival date (year plus engineered month/weekday terms); booked stay and guest counts; meal, country, market segment, distribution channel, reserved room, agent, customer type; company presence; and previous-booking history assumed available before the current booking. The source is not a sequence of historical booking snapshots, so that availability assumption remains a limitation.

Both ADR and IsCanceled are excluded from predictors. ReservationStatus, ReservationStatusDate, AssignedRoomType and BookingChanges are excluded as final/lifecycle information. DaysInWaitingList, RequiredCarParkingSpaces, TotalOfSpecialRequests and DepositType are excluded conservatively until availability at booking creation is justified. ArrivalDateWeekNumber is omitted as redundant. ADR is the regression target; IsCanceled is the classification target. Clustering uses only lead time, total nights, total guests, company presence, repeated-guest status, cyclical month, meal, market segment, customer type and reserved room type, with no supplied outcome.

Numerical median imputation and optional standard scaling are fitted only on training data. Categories use a Missing token and one-hot encoding, with fixed initial min_frequency=5 and max_categories=50 settings learned on training rows; unseen categories are ignored during transform. Month uses sine/cosine terms. Entire preprocessing/model pipelines must be fitted independently in each GroupKFold training fold. No target encoding is used. Sparse outputs are supplied to the model owners. Clustering's categorical/numerical distance weighting remains a baseline choice to assess.

Preparation, preprocessing checks, and training-only EDA are runnable from main.py. Baseline experiments, performance metrics, remaining-milestone plans and the final progress report remain to be completed by the team.
