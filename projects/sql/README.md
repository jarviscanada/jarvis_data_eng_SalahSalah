# Introduction

# SQL Queries

##### Table Setup (DDL)

##### Modifying Data
###### Question 1: insert

```
INSERT INTO cd.facilities (
  facid, name, membercost, guestcost, 
  initialoutlay, monthlymaintenance
) 
VALUES 
  (9, 'Spa', 20, 30, 100000, 800)
```


###### Question 2: Insert 3
```
INSERT INTO cd.facilities (
  facid, name, membercost, guestcost, 
  initialoutlay, monthlymaintenance
) 
VALUES 
  ((SELECT max(facid) FROM cd.facilities) + 1, 'Spa', 20, 30, 100000, 800)
```
###### Question 3: Update
```
UPDATE 
  cd.facilities 
SET 
  initialoutlay = 10000 
WHERE 
  name = 'Tennis Court 2';
```
###### Question 4: Update Calculated
```
UPDATE 
  cd.facilities 
SET 
  membercost = (
    (
      SELECT 
        membercost 
      FROM 
        cd.facilities 
      WHERE 
        name = 'Tennis Court 1'
    ) * 1.1
  ), 
  guestcost = (
    (
      SELECT 
        guestcost 
      FROM 
        cd.facilities 
      WHERE 
        name = 'Tennis Court 1'
    ) * 1.1
  ) 
WHERE 
  name = 'Tennis Court 2';
```
###### Question 5: Delete 
```
DELETE FROM cd.bookings;
```

###### Question 6: Delete Condition
```
DELETE FROM 
  cd.members 
WHERE 
  memid = 37;
```

##### Basics
###### Question 7: Where 2
```
SELECT 
  facid, 
  name, 
  membercost, 
  monthlymaintenance 
FROM 
  cd.facilities 
WHERE 
  (membercost > 0) 
  AND (
    membercost < (monthlymaintenance / 50)
  );
```
###### Question 8: Where 3
```
SELECT 
  * 
FROM 
  cd.facilities 
WHERE 
  name LIKE '%Tennis%';
```
###### Question 9: Where 4
```
SELECT 
	*
FROM 
	cd.facilities
WHERE
	facid IN (1, 5);
```
###### Question 10: Date
```
SELECT 
	memid, 
	surname,
	firstname,
	joindate
FROM
	cd.members
WHERE
	joindate > '2012-09-01';
```
###### Question 11: Union
```
SELECT surname FROM cd.members
UNION
SELECT name FROM cd.facilities;
```
##### Joins
###### Question 12: Join
```
SELECT starttime
FROM   cd.bookings
       LEFT JOIN cd.members
              ON cd.members.memid = cd.bookings.memid
WHERE  surname = 'Farrell'
       AND firstname = 'David
```
###### Question 13: Join 2
```
SELECT starttime,
       NAME
FROM   cd.bookings
       JOIN cd.facilities
         ON cd.bookings.facid = cd.facilities.facid
WHERE  starttime >= '2012-09-21 00:00:00'
       AND starttime < '2012-09-22 00:00:00'
       AND NAME LIKE 'Tennis Court%'
ORDER  BY starttime ASC 
```
###### Question 14:Self 2
```
SELECT members1.firstname AS memfname,
       members1.surname   AS memsname,
       members2.firstname AS recfname,
       members2.surname   AS recsname
FROM   cd.members AS members1
       LEFT JOIN cd.members AS members2
              ON members1.recommendedby = members2.memid
ORDER  BY memsname,
          memfname 
```
###### Question 15: Self
```
SELECT DISTINCT members1.firstname AS firstname,
                members1.surname   AS surname
FROM   cd.members AS members1
       INNER JOIN cd.members AS members2
               ON members2.recommendedby = members1.memid
ORDER  BY surname,
          firstname 
```
###### Question 16: Sub
```
SELECT DISTINCT Concat(mem.firstname, ' ', mem.surname) AS member,
                (SELECT Concat(rec.firstname, ' ', rec.surname)
                 FROM   cd.members rec
                 WHERE  rec.memid = mem.recommendedby)  AS recommender
FROM   cd.members mem 
```
##### Aggregations
###### Question 17: Count 3
```
SELECT 
	recommendedby,
	COUNT(*) AS count
FROM
	cd.members
WHERE recommendedby IS NOT NULL
GROUP BY
	recommendedby
ORDER BY recommendedby
```
###### Question 18: fachours
```
SELECT bks.facid,
       Sum(bks.slots) AS "TOTAL SLOTS"
FROM   cd.facilities fac
       LEFT JOIN cd.bookings bks
              ON fac.facid = bks.facid
GROUP  BY bks.facid
ORDER  BY bks.facid 
```
###### Question 19: fachours by month
```
SELECT bookings.facid,
       Sum(bookings.slots) AS total_slots
FROM   cd.facilities facilities
       LEFT JOIN cd.bookings bookings
              ON facilities.facid = bookings.facid
WHERE  starttime >= '2012-09-01 00:00:00'
       AND starttime < '2012-10-01 00:00:00'
GROUP  BY bookings.facid
ORDER  BY total_slots 
```
###### Question 20: fachours by month 2
```
SELECT bookings.facid,
       Extract(month FROM bookings.starttime) AS month,
       Sum(bookings.slots)
FROM   cd.facilities facilities
       LEFT JOIN cd.bookings
              ON facilities.facid = bookings.facid
WHERE  bookings.starttime >= '2012-01-01 00:00:00'
       AND bookings.starttime < '2013-01-01 00:00:00'
GROUP  BY bookings.facid,
          month
ORDER  BY bookings.facid,
          month 
```
###### Question 21: members 1
```
SELECT Count (DISTINCT members.memid)
FROM   cd.members members
       INNER JOIN cd.bookings bookings
               ON members.memid = bookings.memid 
```
###### Question 22: nbooking
```
SELECT DISTINCT members.surname         AS surname,
                members.firstname       AS firstname,
                members.memid           AS memid,
                Min(bookings.starttime) AS starttime
FROM   cd.members members
       LEFT JOIN cd.bookings bookings
              ON members.memid = bookings.memid
WHERE  starttime > '2012-09-01 00:00:00'
GROUP  BY members.surname,
          members.firstname,
          members.memid
ORDER  BY memid 
```
###### Question 23: count members
```
SELECT (SELECT Count(memid)
        FROM   cd.members),
       firstname,
       surname
FROM   cd.members
ORDER  BY joindate 
```
###### Question 24: num members
```
SELECT
	ROW_NUMBER() OVER (ORDER BY joindate) AS row_number,
	firstname,
	surname
FROM
	cd.members
ORDER BY
	joindate
```
###### Question 25: fachours 4
```
SELECT facilities.facid,
       Sum(bookings.slots)
FROM   cd.facilities facilities
       LEFT JOIN cd.bookings bookings
              ON facilities.facid = bookings.facid
GROUP  BY facilities.facid
ORDER  BY sum DESC
LIMIT  1 
```
##### Strings
###### Question 26: Format String
```
SELECT 
	CONCAT(surname, ', ', firstname) AS name
FROM
	cd.members
```
###### Question 27: Reg
```
SELECT
	memid,
	telephone
FROM
	cd.members
WHERE
	telephone LIKE '(___) ___-____'
ORDER BY
	memid ASC
```
###### Question 28: Sub Str
```
SELECT
	SUBSTR(surname, 1, 1) AS letter,
	COUNT(*)
FROM
	cd.members
GROUP BY
	letter
ORDER BY 
	letter
```

