#!/usr/bin/env bash
# Resource monitoring script for EC2 workflows
# Shows CPU, RAM, top processes, and disk I/O

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default update interval (seconds)
INTERVAL=${1:-5}

# Function to print header
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}System Resource Monitor${NC} - $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BLUE}========================================${NC}"
}

# Function to get CPU usage
get_cpu_usage() {
    echo -e "\n${GREEN}CPU Usage:${NC}"
    # Overall CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
    echo -e "  Overall: ${YELLOW}${cpu_usage}${NC}"
    
    # Per-core CPU usage (if available)
    if command -v mpstat &> /dev/null; then
        echo -e "  Per-core:"
        mpstat -P ALL 1 1 | tail -n +4 | awk '{printf "    CPU %s: %5.1f%%\n", $2, 100-$12}'
    fi
    
    # Load average
    load_avg=$(uptime | awk -F'load average:' '{print $2}')
    echo -e "  Load Average:${YELLOW}${load_avg}${NC}"
}

# Function to get RAM usage
get_ram_usage() {
    echo -e "\n${GREEN}Memory Usage:${NC}"
    free -h | awk '
        /^Mem:/ {
            printf "  Total:     %s\n", $2
            printf "  Used:      %s (%s)\n", $3, $3/$2*100 "%"
            printf "  Free:      %s\n", $4
            printf "  Available: %s\n", $7
            printf "  Cached:    %s\n", $6
        }
        /^Swap:/ {
            if ($2 != "0B") {
                printf "  Swap Used: %s / %s\n", $3, $2
            }
        }
    '
    
    # Memory percentage
    mem_percent=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$mem_percent > 80" | bc -l) )); then
        echo -e "  ${RED}⚠ WARNING: Memory usage > 80%${NC}"
    elif (( $(echo "$mem_percent > 60" | bc -l) )); then
        echo -e "  ${YELLOW}⚠ Memory usage > 60%${NC}"
    fi
}

# Function to get top processes
get_top_processes() {
    echo -e "\n${GREEN}Top Processes by CPU:${NC}"
    ps aux --sort=-%cpu | head -n 6 | awk '
        NR==1 {printf "  %-8s %5s %5s %8s %s\n", "USER", "PID", "%CPU", "%MEM", "COMMAND"}
        NR>1 {printf "  %-8s %5s %5.1f %7.1f %s\n", $1, $2, $3, $4, substr($0, index($0,$11))}
    '
    
    echo -e "\n${GREEN}Top Processes by Memory:${NC}"
    ps aux --sort=-%mem | head -n 6 | awk '
        NR==1 {printf "  %-8s %5s %5s %8s %s\n", "USER", "PID", "%CPU", "%MEM", "COMMAND"}
        NR>1 {printf "  %-8s %5s %5.1f %7.1f %s\n", $1, $2, $3, $4, substr($0, index($0,$11))}
    '
}

# Function to get disk I/O
get_disk_io() {
    echo -e "\n${GREEN}Disk Usage:${NC}"
    df -h / /mnt/nvme 2>/dev/null | awk '
        NR==1 {printf "  %-12s %8s %8s %8s %5s %s\n", "Filesystem", "Size", "Used", "Avail", "Use%", "Mounted"}
        NR>1 {printf "  %-12s %8s %8s %8s %5s %s\n", $1, $2, $3, $4, $5, $6}
    '
    
    # Disk I/O stats (if iostat available)
    if command -v iostat &> /dev/null; then
        echo -e "\n${GREEN}Disk I/O (last 5s):${NC}"
        iostat -x 1 2 | tail -n +4 | head -n 10 | awk '
            NR==1 {printf "  %-10s %8s %8s %8s %8s %8s\n", "Device", "r/s", "w/s", "rMB/s", "wMB/s", "%util"}
            NR>1 {printf "  %-10s %8.1f %8.1f %8.2f %8.2f %7.1f%%\n", $1, $4, $5, $6/1024, $7/1024, $10}
        '
    fi
}

# Function to get Python processes (workflow related)
get_python_processes() {
    python_count=$(pgrep -c python3 2>/dev/null || echo "0")
    if [ "$python_count" -gt 0 ]; then
        echo -e "\n${GREEN}Python Processes:${NC}"
        echo -e "  Count: ${YELLOW}${python_count}${NC}"
        ps aux | grep -E "python3.*run_(cohort|mc_feature|final_model|shap|ffa)" | grep -v grep | awk '
            {printf "  PID %5s: %s %s %s\n", $2, $3"% CPU", $4"% MEM", substr($0, index($0,$11))}
        ' | head -n 10
    fi
}

# Main monitoring loop
while true; do
    clear
    print_header
    get_cpu_usage
    get_ram_usage
    get_python_processes
    get_top_processes
    get_disk_io
    
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "Press ${YELLOW}Ctrl+C${NC} to exit"
    echo -e "Updating every ${YELLOW}${INTERVAL}${NC} seconds..."
    sleep "$INTERVAL"
done

