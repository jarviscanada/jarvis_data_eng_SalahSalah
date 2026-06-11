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
# CPU Info
lscpu_out=$(lscpu)
# Parsing Host Data
hostname=$(hostname -f)
cpu_number=$(echo "$lscpu_out" | egrep "^CPU\(s\):" | awk '{print $2}' | xargs)
cpu_architecture=$(echo "$lscpu_out" | egrep "^Architecture:" | awk '{print $2}' | xargs)
cpu_model=$(echo "$lscpu_out" | egrep "^Model name:" | awk '{print $3,$4, $5, $6, $7}')
cpu_mhz=$(echo "$cpu_model" | awk '{print $5}'| sed 's/[Gg][Hh][Zz]//' | awk '{ printf "%.3f\n", $1 * 1000 }')
l2_cache=$(echo "$lscpu_out" | egrep "^L2 cache:" | awk '{print $3}')
total_mem=$(vmstat --unit M | tail -1 | awk '{print $4}')
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
# Create Insert Statements
insert_stmt="INSERT INTO host_info (
hostname, cpu_number, cpu_architecture, cpu_model, cpu_mhz, l2_cache, timestamp, total_mem)
VALUES('$hostname', '$cpu_number', '$cpu_architecture', '$cpu_model', '$cpu_mhz', '$l2_cache', '$timestamp', '$total_mem')";
# Execute Insert Into PSQL Database
export PGPASSWORD=$psql_password
psql -h $psql_host -p $psql_port -d $db_name -U $psql_user -c "$insert_stmt"
exit $?
