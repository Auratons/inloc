#!/bin/bash

job_id=$1

# While loop that records resources
JobStatus="$(sacct -j $job_id | awk 'FNR == 3 {print $6}')"
JobNode="$(sacct -j $job_id --format=NodeList | sed -n 3p | tr -d ' ')"
#sleep time in seconds
STIME=5
while [ "$JobStatus" != "COMPLETED" ]; do
    #update job status
    JobStatus="$(sacct -j $job_id | awk 'FNR == 3 {print $6}')"
    if [ "$JobStatus" == "RUNNING" ]; then
        echo -n "$(date '+%H:%M:%S') " >> logs/inloc_algo_${job_id}_stats.txt
        ssh "${JobNode}" ps -u $USER -o pid,state,cputime,%cpu,rssize,command >> logs/inloc_algo_${job_id}_stats.txt
        sleep $STIME
    elif [ "$JobStatus" == "PENDING" ]; then
        sleep $STIME
    else
        echo -n "$(date '+%H:%M:%S') " >> logs/inloc_algo_${job_id}_stats.txt
        sacct -j ${job_id} --format=AllocCPUS,ReqMem,MaxRSS,AveRSS,AveDiskRead,AveDiskWrite,ReqCPUS,AllocCPUs,NTasks,Elapsed,State >> logs/inloc_algo_${job_id}_stats.txt
        JobStatus="COMPLETED"
        break
    fi
done
