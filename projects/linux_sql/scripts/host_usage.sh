# Setup Arguments
psql_host=$1
psql_port=$2
db_name=$3
psql_user=$4
psql_password=$5

# Validate Arguments
if [ "$#" -ne 5 ]; then
    echo "Invalid Arguments, 5 Required: host port database_name username password"
    exit 1
fi

# Get Host Memory Info
vmstat_mb=$(vmstat --unit M)
hostname=$(hostname -f)

# Parsing Memory Usage Statistics
memory_free=$(echo "$vmstat_mb" | awk '{print $4}'| tail -n1 | xargs)
cpu_idle=$(echo "$vmstat_mb" | awk '{print $15}' | tail -n1 | xargs)
cpu_kernel=$(echo "$vmstat_mb" | awk '{print $14}' | tail -n1 | xargs)
disk_io=$(vmstat -d | awk '{print $10}' | tail -n1 | xargs)
disk_available=$(df -BM --total | grep "^total" | awk '{gsub(/M/,"",$4); print $4}')

# Get Timestamp
timestamp=$(vmstat -t | awk '{print $18}' | tail -n1 | xargs)

#set up env var for psql cmd
export PGPASSWORD=$psql_password

# Subquery to find corresponding host_info foreign key
host_id="(SELECT id FROM host_info WHERE hostname='$hostname')";

# Insert Data Statements
insert_stmt="INSERT INTO host_usage(
timestamp, host_id, memory_free, cpu_idle, cpu_kernel, disk_io, disk_available)
VALUES('$timestamp', $host_id, '$memory_free', '$cpu_idle', '$cpu_kernel', '$disk_io', '$disk_available')";

#Insert date into a database
psql -h $psql_host -p $psql_port -d $db_name -U $psql_user -c "$insert_stmt"
exit $?