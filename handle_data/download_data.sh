#!/usr/bin/env bash
M_USER="felipe.kpereira@live.com"
M_PASS="rubyheron82"
M_FORMAT="AQCSV"
F_STATE="48"
F_COUNTIES=(113)
F_YEAR=(2017)
F_PARAM=(42401 42603 43205 43206 43218 43233 43801 43802 43803 43804 43811 43813 43814 43815 43817 43818 43819 43820 43823 43824 43826 43829 43830 43831 43843 43860 44201 45109 45201 45202 45203 45204 45207 45208 45220 45801 61103 61104 62201 63301 68105 68108 68110 88101)

for county in "${F_COUNTIES[@]}"
do
    for year in "${F_YEAR[@]}"
    do
        # if directory doesn exist, create it
        if [ ! -d `echo ${F_STATE}/${county}/${year}'-'${year}` ]; then
            mkdir -p ${F_STATE}/${county}/${year}'-'${year}
        else
            echo 'Folder: '${F_STATE}/${county}/${year}'-'${year}' already exists'
        fi
        
        for param in "${F_PARAM[@]}"
        do
            # if file does not exist, download it
            if [ ! -e `echo ${F_STATE}/${county}/${year}'-'${year}/${param}.'csv'` ]; then
                touch ${F_STATE}/${county}/${year}'-'${year}/${param}.csv
                wget -O ${F_STATE}/${county}/${year}'-'${year}/${param}.csv "https://aqs.epa.gov/api/rawData?user="${M_USER}"&pw="${M_PASS}"&format="${M_FORMAT}"&param="${param}"&bdate="${year}"0101&edate="${year}"1231&state="${F_STATE}"&county="${county}"" --no-check-certificate
                sleep 5
            else
                echo 'file '${param}' already exists in '${year}'-'${year}
            fi

        done
    done
done
