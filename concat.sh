#!/usr/bin/env bash
F_STATE="48"
F_COUNTIES=(113)
F_START_YEAR=2000
F_END_YEAR=2016
F_PARAM=(42401 42603 43205 43206 43218 43233 43801 43802 43803 43804 43811 43813 43814 43815 43817 43818 43819 43820 43823 43824 43826 43829 43830 43831 43843 43860 44201 45109 45201 45202 45203 45204 45207 45208 45220 45801 61103 61104 62201 63301 68105 68108 68110 88101)

for county in "${F_COUNTIES[@]}"; do
    for param in "${F_PARAM[@]}"; do
        mkdir -p ${F_STATE}/${county}/"${F_START_YEAR}-${F_END_YEAR}"
        cd ${F_STATE}/${county}/
        sed '$d' ${F_START_YEAR}'-'${F_START_YEAR}/${param}.csv > "${F_START_YEAR}-${F_END_YEAR}/${param}.csv"
        for ((i=${F_START_YEAR}+1;i<=${F_END_YEAR};i++)); do
            # delete first and last lines here
            sed '1d;$d' ${i}'-'${i}/${param}.csv >> "${F_START_YEAR}-${F_END_YEAR}/${param}.csv"
        done
        cd ../..
    done
done


