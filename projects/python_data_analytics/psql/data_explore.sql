-- Show table schema 
\d+ retail;

-- Show first 10 rows
SELECT * FROM retail limit 10;

-- Check # of records
SELECT count(*) From retail;

-- number of clients (e.g. unique client ID)
SELECT count(DISTINCT customer_id) From retail;

-- Invoice date range (e.g. max/min dates)
SELECT max(invoice_date), min(invoice_date) From retail;

-- Number of SKU/merchants (e.g. unique stock code)
SELECT count(DISTINCT stock_code) From retail;

-- Calculate average invoice amount excluding invoices with a negative amount (e.g. canceled orders have negative amount)
SELECT AVG(invoice_total) AS avg_invoice_value
FROM (
         SELECT
             invoice_no,
             SUM(quantity * unit_price) AS invoice_total
         FROM retail
         GROUP BY invoice_no
         HAVING SUM(quantity * unit_price) > 0
     ) t;
-- Calculate total revenue (e.g. sum of unit_price * quantity)
SELECT SUM(invoice_total) AS total_revenue
FROM (
         SELECT
             invoice_no,
             SUM(quantity * unit_price) AS invoice_total
         FROM retail
         GROUP BY invoice_no
     ) t;
-- Calculate total revenue by YYYYMM
SELECT
    month,
    SUM(invoice_total) AS sum
FROM (
    SELECT
    invoice_no,
    DATE_TRUNC('month', invoice_date) AS month,
    SUM(quantity * unit_price) AS invoice_total
    FROM retail
    GROUP BY invoice_no, DATE_TRUNC('month', invoice_date)
    ) sub
GROUP BY month
ORDER BY month
;